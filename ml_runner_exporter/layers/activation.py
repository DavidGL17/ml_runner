from enum import Enum
from typing import Self

from onnx import NodeProto

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import onnx_shape_to_tensor_shape


class ActivationTypes(Enum):
    ReLU = 1
    Sigmoid = 2
    Tanh = 3
    Softmax = 4

    def to_rust_id(self) -> str:
        if self == ActivationTypes.ReLU:
            return "relu"
        if self == ActivationTypes.Sigmoid:
            return "sigmoid"
        if self == ActivationTypes.Tanh:
            return "tanh"
        if self == ActivationTypes.Softmax:
            return "softmax"
        raise ValueError(f"Unknown activation type: {self}")

    @classmethod
    def from_onnx_type(cls, activation_type: str) -> Self:
        if activation_type == "Relu":
            return ActivationTypes.ReLU
        if activation_type == "Sigmoid":
            return ActivationTypes.Sigmoid
        if activation_type == "Tanh":
            return ActivationTypes.Tanh
        if activation_type == "Softmax":
            return ActivationTypes.Softmax
        raise ValueError(f"Unknown activation type: {activation_type}")


class ActivationLayerParser(LayerParser):
    def __init__(self, activation_type: ActivationTypes, shape: dict) -> None:
        super().__init__("activation")
        self.activation_type = activation_type
        self.shape = shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "activation_type": self.activation_type.to_rust_id(),
            "shape": self.shape,
        }

    @classmethod
    def activation_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict) -> Self:
        # 1. Get the name of the input tensor to this activation
        input_tensor_name = node.input[0]

        # 2. Look up the shape in our tensor_shapes dictionary
        shape = tensor_shapes.get(input_tensor_name)
        if shape is None:
            raise ValueError(f"Could not determine input shape for activation node {node.name}; shape inference may have failed")

        # 3. Convert to TensorShape's representation (Flat after a Dense
        #    layer, D3 after a Conv2D layer) - same helper used for the
        #    model's overall input/output shape.
        layer_shape = onnx_shape_to_tensor_shape(shape)

        return cls(ActivationTypes.from_onnx_type(node.op_type), layer_shape)
