import numpy as np
import onnx
from onnx import GraphProto, NodeProto, numpy_helper, shape_inference

from ml_runner_exporter.dtos import NodeContext
from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.model import export_model
from ml_runner_exporter.node_dispatch import OP_HANDLERS
from ml_runner_exporter.utils import dims_to_tensor_shape


def _compute_io_specs(graph: GraphProto) -> tuple[list[dict], list[dict]]:
    """Builds the model's IoSpec lists ({"name": ..., "shape": ...}),
    reusing the ONNX graph's own input/output tensor names - these are
    exactly the names `_build_node` uses for node wiring too, so a node
    consuming a graph input or producing a graph output lines up
    automatically without any extra bookkeeping.
    """
    is_recurrent = len(graph.node) > 0 and graph.node[0].op_type in ("RNN", "GRU")

    inputs = []
    for inp in graph.input:
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        if is_recurrent:
            seq_len, _batch, features = shape
            tensor_shape = dims_to_tensor_shape((seq_len, features))
        else:
            tensor_shape = dims_to_tensor_shape(tuple(shape[1:]))
        inputs.append({"name": inp.name, "shape": tensor_shape})

    outputs = []
    for out in graph.output:
        shape = tuple(d.dim_value for d in out.type.tensor_type.shape.dim)
        if is_recurrent:
            if len(shape) == 4:
                # Y: (seq_len, num_directions, batch, hidden) -> return_sequences=True
                seq_len, _num_directions, _batch, hidden = shape
                tensor_shape = dims_to_tensor_shape((seq_len, hidden))
            else:
                # Y_h: (num_directions, batch, hidden) -> final hidden state only
                _num_directions, _batch, hidden = shape
                tensor_shape = dims_to_tensor_shape((hidden,))
        else:
            tensor_shape = dims_to_tensor_shape(tuple(shape[1:]))
        outputs.append({"name": out.name, "shape": tensor_shape})

    return inputs, outputs


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


def _data_tensor_names(node: NodeProto, weights: dict) -> list[str]:
    """The subset of `node.input` that are actual data-flow tensors (an
    upstream node's output, or a graph input) rather than a weight/bias
    initializer that's already been folded into the parsed `LayerParser`
    (e.g. Gemm's W/B, Conv's weight/bias, RNN/GRU's W/R/B).
    """
    return [name for name in node.input if name not in weights]


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
    except `add`, which can take more than one real tensor input - a
    residual/skip connection (`x + shortcut(x)`) sums two computed
    activations, not just one tensor plus baked-in constants. Anything
    else with more than one real data input still raises, the same
    restriction `Model::validate_shapes` enforces at load time.
    """
    data_inputs = _data_tensor_names(node, weights)
    is_multi_input_add = layer.layer_type == "add" and len(data_inputs) > 1

    if len(data_inputs) < 1 or (not is_multi_input_add and len(data_inputs) != 1):
        raise ValueError(
            f"Node '{node.name or index}' ({node.op_type}) has {len(data_inputs)} data "
            "inputs; only single-input layers (or a multi-input Add) are supported by the Rust runtime so far"
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


def _parse_node(node: NodeProto, tensor_shapes: dict, weights: dict) -> LayerParser:
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

    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    inputs, outputs = _compute_io_specs(graph)

    # Order matters here: constants must be folded into `weights` first, so
    # that a Constant -> Identity or Constant -> Unsqueeze chain (rewrapping
    # a literal under another name/shape) is recognized regardless of where
    # each node sits in the graph.
    skip_node_names = _resolve_constant_nodes(graph, weights, tensor_shapes)
    skip_node_names |= _resolve_constant_identities(graph, weights, tensor_shapes)
    skip_node_names |= _resolve_constant_unsqueezes(graph, weights, tensor_shapes)

    nodes = [
        _build_node(node, _parse_node(node, tensor_shapes, weights), weights, i)
        for i, node in enumerate(graph.node)
        if (node.name or f"node_{i}") not in skip_node_names
    ]

    return export_model(nodes, inputs, outputs)
