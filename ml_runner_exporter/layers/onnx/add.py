from typing import Self

from numpy import ndarray

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


class AddLayerParser(LayerParser):
    def __init__(self, shape: tuple, constants: list[list[float]]) -> None:
        super().__init__("add")
        self.shape = dims_to_tensor_shape(tuple(shape[1:]))
        self.constants = constants

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "shape": self.shape,
            "constants": self.constants,
        }

    @classmethod
    def add_layer_from_onnx(cls, node_weights: list[ndarray], output_shape: tuple) -> Self:
        """
        node_weights: The list of ndarrays extracted from the node's inputs
                      that exist in the graph's initializers.
        output_shape: The shape of the resulting tensor after the addition.
        """
        # Convert each input ndarray into a flat list of floats
        constants = [[float(x) for x in arr.flatten()] for arr in node_weights]

        return cls(shape=output_shape, constants=constants)
