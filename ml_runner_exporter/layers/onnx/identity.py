from typing import Self

from onnx import NodeProto

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


class IdentityLayerParser(LayerParser):

    def __init__(self, shape: dict) -> None:
        super().__init__("identity")
        # Already in TensorShape's serde representation (e.g. {"Flat": 10}
        # or {"D3": {...}}) - see `dims_to_tensor_shape` - so it's stored
        # as-is rather than as separate dims, unlike Conv2D/Flatten which
        # are hard-coded to the 4D (N, C, H, W) conv shape.
        self.shape = shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "shape": self.shape,
        }

    @classmethod
    def identity_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict) -> Self:
        input_tensor_name = node.input[0]
        input_shape = tensor_shapes.get(input_tensor_name)
        if input_shape is None:
            raise ValueError(f"Could not determine input shape for Identity node {node.name}")

        # Drop the leading batch dimension, same as `_compute_io_specs`
        # does for graph-level inputs/outputs. Unlike Conv/Flatten, Identity
        # isn't restricted to 4D (N, C, H, W) tensors - it can sit anywhere
        # in the graph - so this passes through whatever rank is left
        # (1-3 dims) to `dims_to_tensor_shape` rather than assuming D3.
        tensor_shape = dims_to_tensor_shape(tuple(input_shape[1:]))
        return cls(shape=tensor_shape)
