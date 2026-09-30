from typing import Self

import numpy as np
from onnx import NodeProto

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


class ExpandLayerParser(LayerParser):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        super().__init__("expand")
        # Both already batch-stripped (see expand_layer_from_onnx).
        self.input_shape = input_shape
        self.output_shape = output_shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "input_shape": dims_to_tensor_shape(self.input_shape),
            "output_shape": dims_to_tensor_shape(self.output_shape),
        }

    @classmethod
    def expand_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict, weights: dict) -> Self:
        """ONNX `Expand(input, shape)` broadcasts `input` numpy-style to `shape`.
        The target `shape` must be a compile-time constant (an initializer or a
        folded Constant/Concat chain), since the Rust layer stores the resulting
        output shape statically instead of reading a second tensor at runtime.
        """
        shape_name = node.input[1]
        if shape_name not in weights:
            raise ValueError(
                f"Expand node '{node.name}' has a non-constant target shape '{shape_name}'; only constant (statically foldable) shapes are supported"
            )

        input_shape = tensor_shapes.get(node.input[0])
        if input_shape is None:
            raise ValueError(f"Could not determine the input shape for Expand node '{node.name}'")

        target = [int(d) for d in np.asarray(weights[shape_name]).flatten()]
        try:
            full_output = tuple(int(d) for d in np.broadcast_shapes(tuple(input_shape), tuple(target)))
        except ValueError as e:
            raise ValueError(f"Expand node '{node.name}': input shape {tuple(input_shape)} can't be broadcast to {tuple(target)}") from e

        # Drop the leading batch dimension from both, like every other layer
        # parser. Broadcasting aligns trailing dims, so this stays valid even
        # when the input has lower rank than the output.
        if len(input_shape) < 2 or len(full_output) < 2:
            raise ValueError(f"Expand node '{node.name}': need a batch dimension to strip, got input {tuple(input_shape)} -> output {full_output}")
        return cls(tuple(input_shape[1:]), full_output[1:])
