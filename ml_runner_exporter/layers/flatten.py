from typing import Self

from numpy import ndarray
from onnx import NodeProto

from ml_runner_exporter.layer import LayerParser


class FlattenLayerParser(LayerParser):

    def __init__(self, channels: int, height: int, width: int):
        super().__init__("flatten")
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

    @classmethod
    def flatten_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict, weights: dict[str, ndarray]) -> Self:
        # torch.onnx.export commonly constant-folds nn.Flatten into a plain Reshape node (with a constant target-shape tensor)
        # when the input shape is fully static, rather than emitting an onnx::Flatten node.
        # Both represent the same operation here, so handle them together.
        input_tensor_name = node.input[0]
        input_shape = tensor_shapes.get(input_tensor_name)
        if input_shape is None or len(input_shape) != 4:
            raise ValueError(f"Could not determine a 4D (N, C, H, W) input shape for " f"{node.op_type} node {node.name}")

        _, channels, height, width = input_shape

        if node.op_type == "Flatten":
            attrs = {a.name: a for a in node.attribute}
            axis = attrs["axis"].i if "axis" in attrs else 1
            if axis != 1:
                raise ValueError(f"Flatten node {node.name} uses axis={axis}; only axis=1 " "(flattening the full C, H, W into one dimension) is supported")
        else:  # Reshape
            # Verify the constant target shape actually matches a flatten
            # ((batch, features)) rather than some other reshape pattern.
            if len(node.input) < 2 or node.input[1] not in weights:
                raise ValueError(f"Reshape node {node.name} has no constant target shape; " "cannot verify it represents a flatten")
            target_shape = weights[node.input[1]].tolist()
            expected_features = channels * height * width
            if not (len(target_shape) == 2 and target_shape[1] in (expected_features, -1)):
                raise ValueError(f"Reshape node {node.name} has target shape {target_shape}; " "only a (batch, -1) flatten reshape is supported")

        return cls(channels=channels, height=height, width=width)
