from typing import Any, Self

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


class ShapeLayerParser(LayerParser):
    def __init__(self, input_shape: dict, output_shape: dict) -> None:
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
        input_shape = dims_to_tensor_shape(tuple(tensor_shapes.get(input_name, ())[1:]))

        # 2. Get the shape of the output tensor
        # The output of a Shape node is always a 1D tensor with length = rank of input
        output_name = node.output[0]
        output_shape = dims_to_tensor_shape(tuple(tensor_shapes.get(output_name, ())[1:]))

        return cls(input_shape=input_shape, output_shape=output_shape)
