from typing import Self

from numpy import ndarray

from ml_runner_exporter.layer import LayerParser


class LinearLayerParser(LayerParser):

    def __init__(self, input_size: int, output_size: int, weights: list, bias: list):
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
        input_size = weight_matrix.shape[1] if weight_matrix is not None else None
        output_size = weight_matrix.shape[0] if weight_matrix is not None else None
        return cls(
            input_size=input_size,
            output_size=output_size,
            weights=weight_matrix.tolist() if weight_matrix is not None else [],
            bias=bias_vector.tolist() if bias_vector is not None else [],
        )
