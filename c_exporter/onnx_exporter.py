import onnx
from onnx import numpy_helper
from .model import export_model
from .layer import LayerParser, LinearLayerParser


def _get_attribute(node, name, default=None):
    """Return the first matching attribute value or a default."""
    for attr in node.attribute:
        if attr.name == name:
            # Depending on type we may need to pick ints, floats, etc.
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
    return default


def export_onnx(model_path: str, output_path: str) -> None:
    model = onnx.load(model_path)

    onnx.checker.check_model(model)
    graph = model.graph

    in_shape: int = 0
    out_shape: int = 0
    for inp in graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        in_shape = shape[1] if len(shape) > 1 else 1

    for out in graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        out_shape = shape[1] if len(shape) > 1 else 1

    # --- Build a weights lookup for quick access ---
    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # --- Iterate over layers (nodes) ---
    layers: list[LayerParser] = []
    for i, node in enumerate(graph.node):
        # Extract weight and bias tensors from the node's inputs
        node_weights = [weights[inp] for inp in node.input if inp in weights]

        weight_matrix = node_weights[0] if len(node_weights) > 0 else None
        bias_vector = node_weights[1] if len(node_weights) > 1 else None

        if node.op_type == "Gemm":
            # Default to linear layer for Gemm/MatMul etc.
            input_size = weight_matrix.shape[1] if weight_matrix is not None else None
            output_size = weight_matrix.shape[0] if weight_matrix is not None else None
            layer = LinearLayerParser(
                layer_num=i,
                input_size=input_size,
                output_size=output_size,
                weights=weight_matrix.T.tolist() if weight_matrix is not None else [],
                bias=bias_vector.T.tolist() if bias_vector is not None else [],
            )
        else:
            raise ValueError(f"Unsupported layer type: {node.op_type}")

        layers.append(layer)

    model_export = export_model(layers, in_shape, out_shape)

    with open(output_path, "w") as f:
        f.write(model_export)
