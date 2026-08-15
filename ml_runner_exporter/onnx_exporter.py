import onnx
from onnx import numpy_helper, shape_inference
from .model import export_model
from .layer import (
    LayerParser,
    LinearLayerParser,
    ActivationTypes,
    ActivationLayerParser,
    Conv2DLayerParser,
    FlattenLayerParser,
)


def _onnx_shape_to_tensor_shape(shape: tuple) -> dict:
    """Convert an ONNX (N, ...) shape into TensorShape's serde JSON representation.

    Drops the leading batch dimension. A single remaining dim becomes
    {"Flat": n}; three remaining dims (C, H, W) become
    {"CHW": {"channels": .., "height": .., "width": ..}}.
    """
    dims = tuple(shape[1:])

    if len(dims) == 1:
        return {"Flat": dims[0]}
    elif len(dims) == 3:
        channels, height, width = dims
        return {"CHW": {"channels": channels, "height": height, "width": width}}
    else:
        raise ValueError(
            f"Unsupported model input/output shape {shape}: expected a single " "feature dim (N, F) or a CHW dim (N, C, H, W) after the batch dimension"
        )


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

    # --- Build a weights lookup for quick access (for Gemm/MatMul/Conv) ---
    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # Get in and out shape for model
    in_shape: dict = {}
    out_shape: dict = {}
    for inp in graph.input:
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        in_shape = _onnx_shape_to_tensor_shape(shape)

    for out in graph.output:
        shape = tuple(d.dim_value for d in out.type.tensor_type.shape.dim)
        out_shape = _onnx_shape_to_tensor_shape(shape)

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

        elif node.op_type == "Conv":
            if weight_matrix is None:
                raise ValueError(f"Conv node {node.name} is missing its weight tensor")

            attrs = {a.name: a for a in node.attribute}

            group = attrs["group"].i if "group" in attrs else 1
            if group != 1:
                raise ValueError(f"Conv node {node.name} uses group={group}; grouped convolutions " "are not supported by the Rust Conv2DLayer")

            kernel_shape = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else list(weight_matrix.shape[2:])
            if len(set(kernel_shape)) != 1:
                raise ValueError(f"Conv node {node.name} has a non-square kernel {kernel_shape}; " "only square kernels are supported")

            strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
            if len(set(strides)) != 1:
                raise ValueError(f"Conv node {node.name} has non-uniform strides {strides}; " "only a single stride value is supported")

            pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
            if len(set(pads)) != 1:
                raise ValueError(f"Conv node {node.name} has asymmetric padding {pads}; " "only symmetric padding is supported")

            input_tensor_name = node.input[0]
            input_shape = tensor_shapes.get(input_tensor_name)
            if input_shape is None or len(input_shape) != 4:
                raise ValueError(f"Could not determine a 4D (N, C, H, W) input shape for Conv node {node.name}")

            _, in_channels, height, width = input_shape
            out_channels = weight_matrix.shape[0]

            layer = Conv2DLayerParser(
                layer_num=i,
                kernel_size=kernel_shape[0],
                stride=strides[0],
                padding=pads[0],
                input_channels=in_channels,
                output_channels=out_channels,
                height=height,
                width=width,
                weights=weight_matrix.tolist(),
                bias=bias_vector.tolist() if bias_vector is not None else [0.0] * out_channels,
            )

        elif node.op_type in ("Flatten", "Reshape"):
            # torch.onnx.export commonly constant-folds nn.Flatten into a plain
            # Reshape node (with a constant target-shape tensor) when the input
            # shape is fully static, rather than emitting an onnx::Flatten node.
            # Both represent the same operation here, so handle them together.
            input_tensor_name = node.input[0]
            input_shape = tensor_shapes.get(input_tensor_name)
            if input_shape is None or len(input_shape) != 4:
                raise ValueError(f"Could not determine a 4D (N, C, H, W) input shape for " f"{node.op_type} node {node.name}")

            _, channels, height, width = input_shape

            if node.op_type == "Flatten":
                attrs = {a.name: a for a in node.attribute}
                axis = attrs["axis"].i if "axis" in attrs else 1
                if axis != 1:
                    raise ValueError(f"Flatten node {node.name} uses axis={axis}; only axis=1 " "(flattening the full C, H, W into one dimension) is supported")
            else:  # Reshape
                # Verify the constant target shape actually matches a flatten
                # ((batch, features)) rather than some other reshape pattern.
                if len(node.input) < 2 or node.input[1] not in weights:
                    raise ValueError(f"Reshape node {node.name} has no constant target shape; " "cannot verify it represents a flatten")
                target_shape = weights[node.input[1]].tolist()
                expected_features = channels * height * width
                if not (len(target_shape) == 2 and target_shape[1] in (expected_features, -1)):
                    raise ValueError(f"Reshape node {node.name} has target shape {target_shape}; " "only a (batch, -1) flatten reshape is supported")

            layer = FlattenLayerParser(
                layer_num=i,
                channels=channels,
                height=height,
                width=width,
            )

        elif node.op_type in ["Relu", "Sigmoid", "Tanh", "Softmax"]:
            # 1. Get the name of the input tensor to this activation
            input_tensor_name = node.input[0]

            # 2. Look up the shape in our tensor_shapes dictionary
            shape = tensor_shapes.get(input_tensor_name)
            if shape is None:
                raise ValueError(f"Could not determine input shape for activation node {node.name}; " "shape inference may have failed")

            # 3. Convert to TensorShape's representation (Flat after a Dense
            #    layer, CHW after a Conv2D layer) - same helper used for the
            #    model's overall input/output shape.
            layer_shape = _onnx_shape_to_tensor_shape(shape)

            layer = ActivationLayerParser(i, ActivationTypes.from_onnx_type(node.op_type), layer_shape)
        else:
            raise ValueError(f"Unsupported layer type: {node.op_type}")

        layers.append(layer)

    return export_model(layers, in_shape, out_shape)
