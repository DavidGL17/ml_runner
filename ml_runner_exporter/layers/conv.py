from typing import Self

from numpy import ndarray
from onnx import NodeProto

from ml_runner_exporter.layer import LayerParser


class Conv2DLayerParser(LayerParser):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
        input_channels: int,
        output_channels: int,
        height: int,
        width: int,
        weights: list,
        bias: list,
    ):
        super().__init__("conv2d")
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.height = height
        self.width = width
        # Expected shape: weights is (output_channels, input_channels, kernel_size, kernel_size),
        # matching ONNX's Conv weight tensor layout directly.
        self.weights = weights
        self.bias = bias

    def to_dict(self) -> dict:
        # Flatten (output_channels, input_channels, kernel_size, kernel_size) in row-major
        # order, matching the indexing Conv2DLayer::weight_idx uses on the Rust side:
        # ((oc * input_channels + ic) * kernel_size + kh) * kernel_size + kw
        flat_weights = [float(w) for oc in self.weights for ic in oc for row in ic for w in row]
        flat_bias = [float(b) for b in self.bias]

        return {
            "type": self.layer_type,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "height": self.height,
            "width": self.width,
            "weights": flat_weights,
            "bias": flat_bias,
        }

    @classmethod
    def conv2d_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict, weight_matrix: ndarray | None, bias_vector: ndarray | None) -> Self:
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

        return Conv2DLayerParser(
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
