from typing import Any, Self

from numpy import ndarray

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


class GatherLayerParser(LayerParser):
    def __init__(self, input_shape: dict, output_shape: dict, axis: int, indices: list[float] | None) -> None:
        super().__init__("gather")
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.axis = axis
        # indices will be a flat list of floats if constant, or None if dynamic
        self.indices = indices

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "axis": self.axis,
            "indices": self.indices,
        }

    @classmethod
    def gather_layer_from_onnx(cls, node: Any, tensor_shapes: dict, weights: dict[str, ndarray]) -> Self:
        """
        node: The ONNX node object (needed for 'axis' attribute).
        tensor_shapes: Dictionary of all tensor shapes in the graph.
        weights: The dictionary of constant tensors (initializers).
        """
        # 1. Identify inputs
        input_name = node.input[0]
        indices_name = node.input[1]

        # 2. Get shapes
        input_shape = dims_to_tensor_shape(tuple(tensor_shapes.get(input_name, ())[1:]))
        output_shape = dims_to_tensor_shape(tuple(tensor_shapes.get(node.output[0], ())[1:]))

        # 3. Extract 'axis' attribute
        axis = 0  # Default fallback
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
                break

        # 4. Try to extract indices if they are constant
        indices_values = None
        if indices_name in weights:
            # If the indices are in the weights dict, they are constant
            indices_values = [float(x) for x in weights[indices_name].flatten()]

        return cls(input_shape=input_shape, output_shape=output_shape, axis=axis, indices=indices_values)
