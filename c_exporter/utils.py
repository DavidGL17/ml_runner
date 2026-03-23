from enum import Enum
from dataclasses import dataclass


class LayerTypes(Enum):
    LINEAR = ("DenseLayer", "Dense")

    def __new__(cls, layer_name, enum_name):
        obj = object.__new__(cls)
        obj._value_ = (layer_name, enum_name)
        obj.layer_name = layer_name
        obj.enum_name = enum_name
        return obj

    @staticmethod
    def from_string(name):
        mapping = {
            "Gemm": LayerTypes.LINEAR,
        }
        return mapping[name]


@dataclass
class Layer:
    layer_num: int
    layerType: LayerTypes
    input_size: int
    output_size: int
    weights: list
    bias: list
