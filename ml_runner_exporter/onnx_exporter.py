import onnx
from onnx import GraphProto, NodeProto, numpy_helper, shape_inference

from ml_runner_exporter.dtos import NodeContext
from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.model import export_model
from ml_runner_exporter.node_dispatch import OP_HANDLERS
from ml_runner_exporter.utils import dims_to_tensor_shape


def _compute_in_out_shapes(graph: GraphProto) -> tuple[dict, dict]:
    is_recurrent = len(graph.node) > 0 and graph.node[0].op_type in ("RNN", "GRU")

    in_shape: dict = {}
    for inp in graph.input:
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        if is_recurrent:
            seq_len, _batch, features = shape
            in_shape = dims_to_tensor_shape((seq_len, features))
        else:
            in_shape = dims_to_tensor_shape(tuple(shape[1:]))

    out_shape: dict = {}
    for out in graph.output:
        shape = tuple(d.dim_value for d in out.type.tensor_type.shape.dim)
        if is_recurrent:
            if len(shape) == 4:
                # Y: (seq_len, num_directions, batch, hidden) -> return_sequences=True
                seq_len, _num_directions, _batch, hidden = shape
                out_shape = dims_to_tensor_shape((seq_len, hidden))
            else:
                # Y_h: (num_directions, batch, hidden) -> final hidden state only
                _num_directions, _batch, hidden = shape
                out_shape = dims_to_tensor_shape((hidden,))
        else:
            out_shape = dims_to_tensor_shape(tuple(shape[1:]))

    return in_shape, out_shape


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
    in_shape, out_shape = _compute_in_out_shapes(graph)

    layers = [_parse_node(node, tensor_shapes, weights) for node in graph.node]

    return export_model(layers, in_shape, out_shape)
