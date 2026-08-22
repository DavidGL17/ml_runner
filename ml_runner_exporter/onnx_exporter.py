import onnx
from onnx import numpy_helper, shape_inference

from ml_runner_exporter.layers.activation import ActivationLayerParser
from ml_runner_exporter.layers.conv import Conv2DLayerParser
from ml_runner_exporter.layers.flatten import FlattenLayerParser
from ml_runner_exporter.layers.linear import LinearLayerParser
from ml_runner_exporter.layers.rnn import RNNLayerParser, GRULayerParser
from ml_runner_exporter.utils import onnx_shape_to_tensor_shape
from .model import export_model
from .layer import LayerParser


def export_onnx(model_path: str) -> dict:
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)  # This populates graph.value_info
    onnx.checker.check_model(model)
    graph = model.graph

    # --- Build a shape lookup for ALL tensors (weights AND intermediate outputs) ---
    # This maps tensor_name -> shape (tuple)
    tensor_shapes = {}

    # Add initializers (weights/biases)
    for init in graph.initializer:
        tensor_shapes[init.name] = numpy_helper.to_array(init).shape

    # Add graph inputs (needed so the *first* layer, e.g. a Conv, can look up its
    # own input shape the same way later layers look theirs up via value_info)
    for inp in graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        tensor_shapes[inp.name] = tuple(shape)

    # Add intermediate tensors (the outputs of layers like Gemm)
    for info in graph.value_info:
        shape = [d.dim_value for d in info.type.tensor_type.shape.dim]
        tensor_shapes[info.name] = tuple(shape)

    # --- Build a weights lookup for quick access (for Gemm/MatMul/Conv/RNN/GRU) ---
    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # Get in and out shape for model
    in_shape: dict = {}
    out_shape: dict = {}
    for inp in graph.input:
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        in_shape = onnx_shape_to_tensor_shape(shape)

    for out in graph.output:
        shape = tuple(d.dim_value for d in out.type.tensor_type.shape.dim)
        out_shape = onnx_shape_to_tensor_shape(shape)

    # --- Iterate over layers (nodes) ---
    layers: list[LayerParser] = []
    for i, node in enumerate(graph.node):
        # Extract weight and bias tensors from the node's inputs
        node_weights = [weights[inp] for inp in node.input if inp in weights]

        weight_matrix = node_weights[0] if len(node_weights) > 0 else None
        bias_vector = node_weights[1] if len(node_weights) > 1 else None

        if node.op_type == "Gemm":
            layer = LinearLayerParser.linear_layer_from_onnx(weight_matrix, bias_vector)
        elif node.op_type == "Conv":
            layer = Conv2DLayerParser.conv2d_layer_from_onnx(node, tensor_shapes, weight_matrix, bias_vector)
        elif node.op_type in ("Flatten", "Reshape"):
            layer = FlattenLayerParser.flatten_layer_from_onnx(node, tensor_shapes, weights)
        elif node.op_type in ["Relu", "Sigmoid", "Tanh", "Softmax"]:
            layer = ActivationLayerParser.activation_layer_from_onnx(node, tensor_shapes)
        elif node.op_type == "RNN":
            layer = RNNLayerParser.rnn_layer_from_onnx(node, tensor_shapes, weights)
        elif node.op_type == "GRU":
            layer = GRULayerParser.gru_layer_from_onnx(node, tensor_shapes, weights)
        else:
            raise ValueError(f"Unsupported layer type: {node.op_type}")

        layers.append(layer)

    return export_model(layers, in_shape, out_shape)
