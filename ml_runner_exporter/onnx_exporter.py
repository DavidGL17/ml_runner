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

    Every `Layer` variant on the Rust side is still single-input/single-
    output for now, so this raises rather than silently dropping extra
    tensor names if an ONNX node turns out to have more than one real
    data input/output - the same restriction `Model::validate_shapes`
    enforces at load time.
    """
    data_inputs = _data_tensor_names(node, weights)
    if len(data_inputs) != 1:
        raise ValueError(
            f"Node '{node.name or index}' ({node.op_type}) has {len(data_inputs)} data "
            "inputs; only single-input layers are supported by the Rust runtime so far"
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

    nodes = [_build_node(node, _parse_node(node, tensor_shapes, weights), weights, i) for i, node in enumerate(graph.node)]

    return export_model(nodes, inputs, outputs)
