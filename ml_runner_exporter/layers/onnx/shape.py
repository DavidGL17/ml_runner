from typing import Any, Self

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


class ShapeLayerParser(LayerParser):
    def __init__(self, input_shape: dict, output_shape: dict) -> None:
        super().__init__("shape")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

    @classmethod
    def shape_layer_from_onnx(cls, node: Any, tensor_shapes: dict) -> Self:
        """
        node: The ONNX node object.
        tensor_shapes: The dictionary containing shapes of all tensors in the graph.
        """
        # The input is a genuine activation, so its recorded ONNX shape
        # carries a leading batch dimension - drop it to get the per-sample
        # shape the Rust runtime works with.
        input_name = node.input[0]
        input_dims = tuple(tensor_shapes.get(input_name, ())[1:])
        input_shape = dims_to_tensor_shape(input_dims)

        # Unlike input_shape, this is NOT a batch-carrying activation shape
        # to drop a dimension from. ONNX's Shape op always outputs a 1-D
        # vector whose *length* equals the input's rank (e.g. an
        # (N, C, H, W) input -> a 4-element output vector) - that vector
        # has no batch dimension of its own. Since the Rust runtime works
        # per-sample (no batch dim in its tensors at all), the layer
        # reports the per-sample rank: the number of dims left in
        # `input_dims` above, not whatever ONNX/shape_inference recorded
        # for the output tensor's own shape (which describes the same
        # thing under a different, batch-inclusive convention).
        output_shape = dims_to_tensor_shape((len(input_dims),))

        return cls(input_shape=input_shape, output_shape=output_shape)
