from typing import Self

from onnx import NodeProto

from ml_runner_exporter.layer import LayerParser


class AvgPool2DLayerParser(LayerParser):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
        channels: int,
        height: int,
        width: int,
        count_include_pad: bool,  # noqa: FBT001
    ) -> None:
        super().__init__("avgpool")
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channels = channels
        self.height = height
        self.width = width
        self.count_include_pad = count_include_pad

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "count_include_pad": self.count_include_pad,
        }

    @classmethod
    def avgpool_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict) -> Self:
        attrs = {a.name: a for a in node.attribute}

        dilations = list(attrs["dilations"].ints) if "dilations" in attrs else [1, 1]
        if any(d != 1 for d in dilations):
            raise ValueError(f"AveragePool node {node.name} uses dilations={dilations}; only dilation=1 is supported")

        if "ceil_mode" in attrs and attrs["ceil_mode"].i != 0:
            raise ValueError(f"AveragePool node {node.name} uses ceil_mode=1; only floor-mode output sizing is supported")

        if "auto_pad" in attrs and attrs["auto_pad"].s not in (b"NOTSET", b""):
            raise ValueError(f"AveragePool node {node.name} uses auto_pad={attrs['auto_pad'].s}; only explicit 'pads' is supported")

        if "kernel_shape" not in attrs:
            raise ValueError(f"AveragePool node {node.name} is missing required attribute 'kernel_shape'")
        kernel_shape = list(attrs["kernel_shape"].ints)
        if len(set(kernel_shape)) != 1:
            raise ValueError(f"AveragePool node {node.name} has a non-square kernel {kernel_shape}; only square kernels are supported")

        strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
        if len(set(strides)) != 1:
            raise ValueError(f"AveragePool node {node.name} has non-uniform strides {strides}; only a single stride value is supported")

        pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
        if len(set(pads)) != 1:
            raise ValueError(f"AveragePool node {node.name} has asymmetric padding {pads}; only symmetric padding is supported")

        count_include_pad = bool(attrs["count_include_pad"].i) if "count_include_pad" in attrs else False

        input_shape = tensor_shapes.get(node.input[0])
        if input_shape is None or len(input_shape) != 4:
            raise ValueError(f"Could not determine a 4D (N, C, H, W) input shape for AveragePool node {node.name}")

        _, channels, height, width = input_shape

        return cls(
            kernel_size=kernel_shape[0],
            stride=strides[0],
            padding=pads[0],
            channels=channels,
            height=height,
            width=width,
            count_include_pad=count_include_pad,
        )
