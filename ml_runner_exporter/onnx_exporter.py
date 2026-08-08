import onnx
from onnx import numpy_helper, shape_inference
from .model import export_model
from .layer import LayerParser, LinearLayerParser, ActivationTypes, ActivationLayerParser


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

    # Add intermediate tensors (the outputs of layers like Gemm)
    for info in graph.value_info:
        shape = [d.dim_value for d in info.type.tensor_type.shape.dim]
        tensor_shapes[info.name] = tuple(shape)

    # --- Build a weights lookup for quick access (for Gemm/MatMul) ---
    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # Get in and out shape for model
    in_shape: int = 0
    out_shape: int = 0
    for inp in graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        in_shape = shape[1] if len(shape) > 1 else 1

    for out in graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        out_shape = shape[1] if len(shape) > 1 else 1

    # --- Iterate over layers (nodes) ---
    layers: list[LayerParser] = []
    for i, node in enumerate(graph.node):
        # Extract weight and bias tensors from the node's inputs
        node_weights = [weights[inp] for inp in node.input if inp in weights]

        weight_matrix = node_weights[0] if len(node_weights) > 0 else None
        bias_vector = node_weights[1] if len(node_weights) > 1 else None

        if node.op_type == "Gemm":
            input_size = weight_matrix.shape[1] if weight_matrix is not None else None
            output_size = weight_matrix.shape[0] if weight_matrix is not None else None
            layer = LinearLayerParser(
                layer_num=i,
                input_size=input_size,
                output_size=output_size,
                weights=weight_matrix.tolist() if weight_matrix is not None else [],
                bias=bias_vector.tolist() if bias_vector is not None else [],
            )

        elif node.op_type in ["Relu", "Sigmoid", "Tanh", "Softmax"]:
            # 1. Get the name of the input tensor to this ReLU
            input_tensor_name = node.input[0]

            # 2. Look up the shape in our tensor_shapes dictionary
            shape = tensor_shapes.get(input_tensor_name)

            # 3. Extract the feature dimension (following your logic: shape[1] or 1)
            if shape is not None:
                input_size = shape[1] if len(shape) > 1 else 1
            else:
                # Fallback if shape inference failed or name not found
                input_size = 1

            layer = ActivationLayerParser(i, ActivationTypes.from_onnx_type(node.op_type), input_size)
        else:
            raise ValueError(f"Unsupported layer type: {node.op_type}")

        layers.append(layer)

    return export_model(layers, in_shape, out_shape)
