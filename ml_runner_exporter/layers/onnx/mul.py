from typing import Self

import numpy as np
from numpy import ndarray

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


class MulLayerParser(LayerParser):
    def __init__(self, shape: tuple, constants: list[list[float]]) -> None:
        super().__init__("mul")
        self.shape = dims_to_tensor_shape(tuple(shape[1:]))
        self.constants = constants

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "shape": self.shape,
            "constants": self.constants,
        }

    @classmethod
    def mul_layer_from_onnx(cls, node_weights: list[ndarray], output_shape: tuple) -> Self:
        """
        node_weights: The constant operands of the Mul (initializers or folded constants).
        output_shape: The ONNX (batch-included) shape of the Mul's output.

        The Rust `MulLayer` only takes full-size or scalar constants, so any
        other constant (e.g. a per-channel BatchNorm scale of shape (64,)
        against a (1, 1, 64) activation) is broadcast to the output shape
        here, following ONNX's numpy-style broadcasting rules.
        """
        constants = []
        for arr in node_weights:
            if arr.size == 1:
                constants.append([float(arr.flatten()[0])])
            else:
                broadcast = np.broadcast_to(arr, output_shape)
                constants.append([float(x) for x in broadcast.flatten()])

        return cls(shape=output_shape, constants=constants)
