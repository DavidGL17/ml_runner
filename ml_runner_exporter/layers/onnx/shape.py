from typing import Any, Self

from ml_runner_exporter.layer import LayerParser


class ShapeLayerParser(LayerParser):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        super().__init__("shape")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

    @classmethod
    def shape_layer_from_onnx(cls, node: Any, tensor_shapes: dict) -> Self:
        """
        node: The ONNX node object.
        tensor_shapes: The dictionary containing shapes of all tensors in the graph.
        """
        # 1. Get the shape of the input tensor
        input_name = node.input[0]
        input_shape = tensor_shapes.get(input_name, ())

        # 2. Get the shape of the output tensor
        # The output of a Shape node is always a 1D tensor with length = rank of input
        output_name = node.output[0]
        output_shape = tensor_shapes.get(output_name, ())

        return cls(input_shape=input_shape, output_shape=output_shape)
