from typing import Self

import numpy as np
from numpy import ndarray

from ml_runner_exporter.layer import LayerParser


class LinearLayerParser(LayerParser):

    def __init__(self, input_size: int, output_size: int, weights: list, bias: list) -> None:
        super().__init__("dense")
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

    @classmethod
    def linear_layer_from_onnx(cls, weight_matrix: ndarray | None, bias_vector: ndarray | None) -> Self:
        if weight_matrix is None:
            raise ValueError("Linear node is missing it's weight tensor")
        if bias_vector is None:
            raise ValueError("Linear node is missing it's bias vector")
        input_size = int(weight_matrix.shape[1])
        output_size = int(weight_matrix.shape[0])
        return cls(
            input_size=input_size,
            output_size=output_size,
            weights=weight_matrix.tolist() if weight_matrix is not None else [],
            bias=bias_vector.tolist() if bias_vector is not None else [],
        )

    @classmethod
    def linear_layer_from_onnx_matmul(cls, input_shape: tuple, weight_matrix: ndarray | None) -> Self:
        """A `MatMul` of a (batch, in) activation with a constant (in, out)
        matrix is a bias-less Dense layer (TensorFlow's `Tensordot` lowers
        to exactly this, with the bias as a separate `Add`).
        """
        if weight_matrix is None or weight_matrix.ndim != 2:
            raise ValueError("MatMul node needs a constant 2D right-hand operand to be exported as a dense layer")
        if len(input_shape) != 2:
            raise ValueError(f"MatMul node has a {len(input_shape)}D input {input_shape}; only a (batch, features) input is supported")
        weights = weight_matrix.T
        return cls(
            input_size=int(weights.shape[1]),
            output_size=int(weights.shape[0]),
            weights=weights.tolist(),
            bias=np.zeros(weights.shape[0], dtype=np.float32).tolist(),
        )
