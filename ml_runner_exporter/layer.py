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


class Conv2DLayerParser(LayerParser):

    def __init__(
        self,
        layer_num: int,
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
        super().__init__("conv2d", layer_num)
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


class FlattenLayerParser(LayerParser):

    def __init__(self, layer_num: int, channels: int, height: int, width: int):
        super().__init__("flatten", layer_num)
        self.channels = channels
        self.height = height
        self.width = width

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            # Matches TensorShape's default (externally-tagged) serde representation:
            # the D3 variant serializes as {"D3": {"dim1": .., "dim2": .., "dim3": ..}}
            "shape": {
                "D3": {
                    "dim1": self.channels,
                    "dim2": self.height,
                    "dim3": self.width,
                }
            },
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
    def __init__(self, layer_num: int, activation_type: ActivationTypes, shape: dict):
        super().__init__("activation", layer_num)
        self.activation_type = activation_type
        self.shape = shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "activation_type": self.activation_type.to_rust_id(),
            "shape": self.shape,
        }
