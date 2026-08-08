from abc import ABC, abstractmethod
from enum import Enum


class LayerParser(ABC):
    def __init__(self, layer_type: str, layer_num: int):
        self.layer_type = layer_type
        self.layer_num = layer_num

    @abstractmethod
    def to_dict(self) -> dict:
        """Return this layer's JSON-serializable representation, matching the `Layer` enum the Rust runtime deserializes from."""
        pass


class LinearLayerParser(LayerParser):

    def __init__(self, layer_num: int, input_size: int, output_size: int, weights: list, bias: list):
        super().__init__("dense", layer_num)
        self.input_size = input_size
        self.output_size = output_size
        # Expected shape: weights is (output_size, input_size), i.e.
        # weights[i][j] is the weight connecting input j to output i.
        self.weights = weights
        self.bias = bias

    def to_dict(self) -> dict:
        flat_weights = [float(w) for row in self.weights for w in row]
        flat_bias = [float(b) for b in self.bias]

        return {
            "type": self.layer_type,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "weights": flat_weights,
            "bias": flat_bias,
        }


class ActivationTypes(Enum):
    ReLU = 1
    Sigmoid = 2
    Tanh = 3
    Softmax = 4

    def to_rust_id(self) -> str:
        if self == ActivationTypes.ReLU:
            return "relu"
        elif self == ActivationTypes.Sigmoid:
            return "sigmoid"
        elif self == ActivationTypes.Tanh:
            return "tanh"
        elif self == ActivationTypes.Softmax:
            return "softmax"
        else:
            raise ValueError(f"Unknown activation type: {self}")

    @staticmethod
    def from_onnx_type(type: str):
        if type == "Relu":
            return ActivationTypes.ReLU
        elif type == "Sigmoid":
            return ActivationTypes.Sigmoid
        elif type == "Tanh":
            return ActivationTypes.Tanh
        elif type == "Softmax":
            return ActivationTypes.Softmax
        else:
            raise ValueError(f"Unknown activation type: {type}")


class ActivationLayerParser(LayerParser):
    def __init__(self, layer_num: int, activation_type: ActivationTypes, input_size: int):
        super().__init__("activation", layer_num)
        self.activation_type = activation_type
        self.input_size = input_size

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "activation_type": self.activation_type.to_rust_id(),
            "input_size": self.input_size,
        }
