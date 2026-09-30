import numpy as np
import onnx
from onnx import GraphProto, NodeProto, numpy_helper, shape_inference

from ml_runner_exporter.dtos import NodeContext
from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.model import export_model
from ml_runner_exporter.node_dispatch import OP_HANDLERS
from ml_runner_exporter.utils import dims_to_tensor_shape

_RECURRENT_OPS = ("RNN", "GRU", "LSTM")


def _compute_io_specs(graph: GraphProto) -> tuple[list[dict], list[dict]]:
    """Builds the model's IoSpec lists ({"name": ..., "shape": ...}),
    reusing the ONNX graph's own input/output tensor names - these are
    exactly the names `_build_node` uses for node wiring too, so a node
    consuming a graph input or producing a graph output lines up
    automatically without any extra bookkeeping.

    Whether a tensor uses the recurrent layout is decided per tensor, not
    per graph: a graph input is (seq_len, batch, features) only if an
    RNN/GRU/LSTM node reads it directly, and a graph output is a recurrent
    Y / Y_h only if such a node produces it. Anything else (e.g. the
    (batch, classes) output of a classifier head sitting after the LSTM)
    just has its leading batch dimension dropped.
    """
    recurrent_nodes = [n for n in graph.node if n.op_type in _RECURRENT_OPS]
    recurrent_inputs = {n.input[0] for n in recurrent_nodes}
    recurrent_outputs = {name for n in recurrent_nodes for name in n.output if name}

    inputs = []
    for inp in graph.input:
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        if inp.name in recurrent_inputs:
            seq_len, _batch, features = shape
            tensor_shape = dims_to_tensor_shape((seq_len, features))
        else:
            tensor_shape = dims_to_tensor_shape(tuple(shape[1:]))
        inputs.append({"name": inp.name, "shape": tensor_shape})

    outputs = []
    for out in graph.output:
        shape = tuple(d.dim_value for d in out.type.tensor_type.shape.dim)
        if out.name in recurrent_outputs:
            if len(shape) == 4:
                # Y: (seq_len, num_directions, batch, hidden) -> return_sequences=True
                seq_len, _num_directions, _batch, hidden = shape
                tensor_shape = dims_to_tensor_shape((seq_len, hidden))
            else:
                # Y_h: (num_directions, batch, hidden) -> final hidden state only
                _num_directions, _batch, hidden = shape
                tensor_shape = dims_to_tensor_shape((hidden,))
        else:
            tensor_shape = dims_to_tensor_shape(tuple(shape[1:]) if len(shape) > 1 else shape)
        outputs.append({"name": out.name, "shape": tensor_shape})

    return inputs, outputs


def _prune_unused_outputs(graph: GraphProto, skip_node_names: set[str]) -> None:
    """Blanks out (`""`) every output of a multi-output node that nothing
    reads - no other emitted node consumes it and it isn't a graph output.

    ONNX represents an omitted optional output as an empty-string
    placeholder, but not every exporter does that: PyTorch's legacy
    TorchScript exporter names all of an LSTM's `Y`, `Y_h` and `Y_c` even
    when only one is used downstream. Normalizing them here means both the
    layer parsers (which decide e.g. `return_sequences` from which of
    `Y`/`Y_h` is present) and `_build_node`'s single-output check see only
    the outputs that are actually wired up.
    """

    def is_skipped(i: int, node: NodeProto) -> bool:
        return (node.name or f"node_{i}") in skip_node_names

    used = {out.name for out in graph.output}
    for i, node in enumerate(graph.node):
        if not is_skipped(i, node):
            used.update(name for name in node.input if name)

    for i, node in enumerate(graph.node):
        if is_skipped(i, node) or len(node.output) < 2:
            continue
        for j, name in enumerate(node.output):
            if name and name not in used:
                node.output[j] = ""


def _resolve_constant_nodes(graph: GraphProto, weights: dict, tensor_shapes: dict) -> set[str]:
    """A `Constant` node bakes a literal tensor value directly into the
    node itself (via a `value`-style attribute) instead of referencing a
    graph-level initializer. Tracing exporters emit these liberally for
    anything from a folded scalar to an entire computed tensor - e.g. a
    shape-manipulation constant feeding `Reshape`/`Gather`, or a scalar
    operand of an `Add`/`Mul`.

    Like a constant-wrapping `Identity` (see `_resolve_constant_identities`),
    a `Constant` node has no Rust-side `Layer` equivalent: it's not
    forward-pass computation, just a literal value. This decodes each
    node's value into an ndarray, registers it in `weights`/`tensor_shapes`
    under the node's output name exactly like a real initializer, and
    returns the node names to drop from the emitted graph.
    """
    skip_node_names = set()
    for i, node in enumerate(graph.node):
        if node.op_type != "Constant":
            continue

        attrs = {a.name: a for a in node.attribute}
        if "value" in attrs:
            array = numpy_helper.to_array(attrs["value"].t)
        elif "value_float" in attrs:
            array = np.array(attrs["value_float"].f, dtype=np.float32)
        elif "value_floats" in attrs:
            array = np.array(list(attrs["value_floats"].floats), dtype=np.float32)
        elif "value_int" in attrs:
            array = np.array(attrs["value_int"].i, dtype=np.int64)
        elif "value_ints" in attrs:
            array = np.array(list(attrs["value_ints"].ints), dtype=np.int64)
        else:
            raise ValueError(
                f"Constant node '{node.name or i}' uses an unsupported attribute variant "
                f"{sorted(attrs)}; only 'value'/'value_float(s)'/'value_int(s)' are supported"
            )

        dst = node.output[0]
        weights[dst] = array
        tensor_shapes[dst] = array.shape
        skip_node_names.add(node.name or f"node_{i}")

    return skip_node_names


def _resolve_constant_identities(graph: GraphProto, weights: dict, tensor_shapes: dict) -> set[str]:
    """Some exporters (notably PyTorch's legacy TorchScript-based ONNX
    exporter) emit `Identity` nodes that just re-wrap a constant/parameter
    under a new name - e.g. a frozen BatchNorm buffer folded during
    tracing - rather than passing an actual activation tensor through.

    Such a node has no Rust-side equivalent: it isn't real forward-pass
    computation, just a rename of something that's already in `weights`.
    Treating it like a genuine data-flow `Identity` (as `IdentityLayerParser`
    does) breaks, because a raw parameter's shape has no batch dimension
    to drop in the first place - e.g. a `(64,)` BatchNorm scale, not a
    `(N, 64, H, W)` activation.

    This registers each such alias back into `weights` (and `tensor_shapes`,
    when known) so any downstream node reading the alias is treated exactly
    like it read the original initializer, and returns the set of node
    names that should be dropped from the emitted graph entirely, since
    they don't correspond to any runtime `Layer`.
    """
    skip_node_names = set()
    for i, node in enumerate(graph.node):
        if node.op_type != "Identity":
            continue
        src = node.input[0]
        if src not in weights:
            continue  # a genuine data-flow Identity - let IdentityLayerParser handle it

        dst = node.output[0]
        weights[dst] = weights[src]
        if src in tensor_shapes:
            tensor_shapes[dst] = tensor_shapes[src]
        skip_node_names.add(node.name or f"node_{i}")

    return skip_node_names


def _resolve_unsqueeze_axes(node: NodeProto, weights: dict) -> list[int]:
    """`axes` is an attribute on opset < 13, or a second (constant) input
    tensor on opset >= 13 - handle either, matching how the rest of this
    exporter already reads dual-representation ONNX attributes (see e.g.
    Conv2DLayerParser's kernel_shape/strides/pads).
    """
    for attr in node.attribute:
        if attr.name == "axes":
            return list(attr.ints)
    if len(node.input) > 1 and node.input[1] in weights:
        return [int(a) for a in weights[node.input[1]].flatten()]
    raise ValueError(f"Unsqueeze node '{node.name}' has no constant 'axes' to fold")


def _resolve_constant_unsqueezes(graph: GraphProto, weights: dict, tensor_shapes: dict) -> set[str]:
    """Same idea as `_resolve_constant_identities`: some exporters wrap a
    constant/parameter in an `Unsqueeze` - e.g. reshaping a frozen
    buffer's rank to line up with a downstream op, or the classic
    `Shape -> Gather -> Unsqueeze -> Concat -> Reshape` dynamic-shape
    chain once the gathered value happens to be statically known - rather
    than only ever applying it to a genuine activation.

    Unsqueezing an already-known constant is a compile-time reshape, not
    real forward-pass computation, so it's resolved here (via
    `np.expand_dims`) into `weights`/`tensor_shapes` under the node's
    output name, and the node is dropped from the emitted graph - same as
    the other constant-folding passes.
    """
    skip_node_names = set()
    for i, node in enumerate(graph.node):
        if node.op_type != "Unsqueeze":
            continue
        src = node.input[0]
        if src not in weights:
            continue  # a genuine data-flow Unsqueeze - let UnsqueezeLayerParser handle it

        array = weights[src]
        for axis in sorted(_resolve_unsqueeze_axes(node, weights)):
            array = np.expand_dims(array, axis=axis)

        dst = node.output[0]
        weights[dst] = array
        tensor_shapes[dst] = array.shape
        skip_node_names.add(node.name or f"node_{i}")

    return skip_node_names


def _resolve_constant_concats(graph: GraphProto, weights: dict, tensor_shapes: dict) -> set[str]:
    """Same idea as the other constant-folding passes: if every real input
    to a `Concat` node is already a known constant - e.g. assembling a
    literal shape/config vector from several `Constant`-folded pieces, the
    tail end of a `Shape -> Gather -> Unsqueeze -> Concat -> Reshape`
    dynamic-shape chain when nothing in it actually turned out to be
    dynamic - joining them is a compile-time operation, not real
    forward-pass computation.

    This folds such a node (via `np.concatenate`) into `weights`/
    `tensor_shapes` under its output name and returns the node names to
    drop from the emitted graph. A `Concat` with even one genuine
    data-flow input (e.g. real channel-wise feature map concatenation) is
    left alone for `ConcatLayerParser` to handle.
    """
    skip_node_names = set()
    for i, node in enumerate(graph.node):
        if node.op_type != "Concat":
            continue
        if not all(name in weights for name in node.input):
            continue  # at least one genuine data-flow input - a real Concat

        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), None)
        if axis is None:
            raise ValueError(f"Concat node '{node.name}' is missing the required 'axis' attribute")

        array = np.concatenate([weights[name] for name in node.input], axis=axis)

        dst = node.output[0]
        weights[dst] = array
        tensor_shapes[dst] = array.shape
        skip_node_names.add(node.name or f"node_{i}")

    return skip_node_names


def _resolve_constant_shapes(graph: GraphProto, weights: dict, tensor_shapes: dict) -> set[str]:
    """`Shape` of a tensor whose dims are statically known is a compile-time constant."""
    skip_node_names = set()
    for i, node in enumerate(graph.node):
        if node.op_type != "Shape":
            continue
        dims = tensor_shapes.get(node.input[0])
        if dims is None or any(d <= 0 for d in dims):
            continue  # unknown/dynamic dim - leave for ShapeLayerParser
        attrs = {a.name: a for a in node.attribute}
        start = attrs["start"].i if "start" in attrs else 0
        end = attrs["end"].i if "end" in attrs else len(dims)

        array = np.array(dims[start:end], dtype=np.int64)
        dst = node.output[0]
        weights[dst] = array
        tensor_shapes[dst] = array.shape
        skip_node_names.add(node.name or f"node_{i}")
    return skip_node_names


def _resolve_constant_gathers(graph: GraphProto, weights: dict, tensor_shapes: dict) -> set[str]:
    skip_node_names = set()
    for i, node in enumerate(graph.node):
        if node.op_type != "Gather":
            continue
        data, indices = node.input[0], node.input[1]
        if data not in weights or indices not in weights:
            continue
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        array = np.take(weights[data], weights[indices].astype(np.int64), axis=axis)

        dst = node.output[0]
        weights[dst] = array
        tensor_shapes[dst] = array.shape
        skip_node_names.add(node.name or f"node_{i}")
    return skip_node_names


def _resolve_constant_expands(graph: GraphProto, weights: dict, tensor_shapes: dict) -> set[str]:
    skip_node_names = set()
    for i, node in enumerate(graph.node):
        if node.op_type != "Expand":
            continue
        src, shape_name = node.input[0], node.input[1]
        if src not in weights or shape_name not in weights:
            continue
        target = tuple(int(d) for d in weights[shape_name].flatten())
        out_shape = np.broadcast_shapes(weights[src].shape, target)
        array = np.broadcast_to(weights[src], out_shape).copy()

        dst = node.output[0]
        weights[dst] = array
        tensor_shapes[dst] = array.shape
        skip_node_names.add(node.name or f"node_{i}")
    return skip_node_names


def _data_tensor_names(node: NodeProto, weights: dict) -> list[str]:
    """The subset of `node.input` that are actual data-flow tensors (an
    upstream node's output, or a graph input) rather than a weight/bias
    initializer that's already been folded into the parsed `LayerParser`
    (e.g. Gemm's W/B, Conv's weight/bias, RNN/GRU's W/R/B).
    """
    return [name for name in node.input if name and name not in weights]


def _data_output_names(node: NodeProto) -> list[str]:
    """The subset of `node.output` that are real produced tensors. ONNX
    represents an unused *optional* output (e.g. RNN/GRU's `Y` when only
    the final hidden state `Y_h` is consumed downstream) as an empty
    string placeholder occupying that position, rather than omitting the
    slot - so a plain `len(node.output)` overcounts for those ops.
    """
    return [name for name in node.output if name]


def _build_node(node: NodeProto, layer: LayerParser, weights: dict, index: int) -> dict:
    """Wraps a parsed `LayerParser` with the graph-wiring fields (`id`,
    `inputs`, `outputs`) the Rust `Node` struct expects.

    Every `Layer` variant on the Rust side is single-input/single-output
    except `add` and `concat`, which can take more than one real tensor
    input - a residual/skip connection (`x + shortcut(x)`) sums two
    computed activations rather than one tensor plus baked-in constants,
    and joining feature maps needs every operand at once. Anything else
    with more than one real data input still raises, the same restriction
    `Model::validate_shapes` enforces at load time.
    """
    data_inputs = _data_tensor_names(node, weights)
    allows_multiple_inputs = layer.layer_type in ("add", "concat")
    is_multi_input_ok = allows_multiple_inputs and len(data_inputs) > 1

    if len(data_inputs) < 1 or (not is_multi_input_ok and len(data_inputs) != 1):
        raise ValueError(
            f"Node '{node.name or index}' ({node.op_type}) has {len(data_inputs)} data "
            "inputs; only single-input layers (or a multi-input Add/Concat) are supported by the Rust runtime so far"
        )

    data_outputs = _data_output_names(node)
    if len(data_outputs) != 1:
        raise ValueError(
            f"Node '{node.name or index}' ({node.op_type}) has {len(data_outputs)} outputs; "
            "only single-output layers are supported by the Rust runtime so far"
        )

    return {
        "id": node.name or f"node_{index}",
        "inputs": data_inputs,
        "outputs": data_outputs,
        **layer.to_dict(),
    }


def _parse_node(node: NodeProto, tensor_shapes: dict, weights: dict, rust_dims: dict) -> LayerParser:
    node_weights = [weights[inp] for inp in node.input if inp in weights]
    weight_matrix = node_weights[0] if node_weights else None
    bias_vector = node_weights[1] if len(node_weights) > 1 else None

    handler = OP_HANDLERS.get(node.op_type)
    if handler is None:
        raise ValueError(f"Unsupported layer type: {node.op_type}")

    ctx = NodeContext(
        node=node,
        tensor_shapes=tensor_shapes,
        weights=weights,
        node_weights=node_weights,
        weight_matrix=weight_matrix,
        bias_vector=bias_vector,
        rust_dims=rust_dims,
    )
    return handler(ctx)


def export_onnx(model_path: str) -> dict:
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)  # This populates graph.value_info
    onnx.checker.check_model(model)
    graph = model.graph

    tensor_shapes = {}
    for init in graph.initializer:
        tensor_shapes[init.name] = numpy_helper.to_array(init).shape
    for inp in graph.input:
        tensor_shapes[inp.name] = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
    for info in graph.value_info:
        tensor_shapes[info.name] = tuple(d.dim_value for d in info.type.tensor_type.shape.dim)
    for out in graph.output:
        tensor_shapes[out.name] = tuple(d.dim_value for d in out.type.tensor_type.shape.dim)

    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    inputs, outputs = _compute_io_specs(graph)

    skip_node_names: set[str] = set()
    passes = (
        _resolve_constant_nodes,
        _resolve_constant_identities,
        _resolve_constant_shapes,
        _resolve_constant_gathers,
        _resolve_constant_unsqueezes,
        _resolve_constant_concats,
        _resolve_constant_expands,
    )
    while True:
        before = len(skip_node_names)
        for fold in passes:
            skip_node_names |= fold(graph, weights, tensor_shapes)
        if len(skip_node_names) == before:
            break

    _prune_unused_outputs(graph, skip_node_names)

    rust_dims: dict = {}
    nodes = [
        _build_node(node, _parse_node(node, tensor_shapes, weights, rust_dims), weights, i)
        for i, node in enumerate(graph.node)
        if (node.name or f"node_{i}") not in skip_node_names
    ]

    return export_model(nodes, inputs, outputs)
