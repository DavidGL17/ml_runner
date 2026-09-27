from typing import Any, Self

from numpy import ndarray

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


def _strip_batch_dim(dims: tuple) -> tuple:
    """Most tensors this exporter looks up in `tensor_shapes` are genuine
    per-sample activations recorded with a leading batch dimension - since
    the Rust runtime only supports 1-3 dim per-sample tensors, a real
    activation's ONNX-recorded (batch-included) shape always has rank 2-4.

    `Gather` is one of the few ops whose input can just as easily be a
    small 1-D "meta" tensor instead - most commonly the output of a
    `Shape` node (e.g. the `Shape -> Gather(indices=[0])` pattern used to
    pull out a dynamic batch size), which was never batched to begin with.
    A tensor already at rank <= 1 can't be a real batched activation under
    this codebase's constraints, so it's used as-is rather than chopping
    off its only dimension.
    """
    return dims[1:] if len(dims) > 1 else dims


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

        # 2. Get shapes - see `_strip_batch_dim` for why this isn't a
        # blind `[1:]` the way a Conv/Flatten input can be.
        input_dims = _strip_batch_dim(tuple(tensor_shapes.get(input_name, ())))
        output_dims = _strip_batch_dim(tuple(tensor_shapes.get(node.output[0], ())))
        input_shape = dims_to_tensor_shape(input_dims)

        if len(output_dims) == 0:
            # Per ONNX semantics, gathering with a scalar (rank-0) index
            # removes the gathered axis entirely rather than keeping it as
            # size 1 - e.g. pulling a single dynamic batch size out of a
            # Shape node's output leaves a genuine 0-D scalar. TensorShape
            # has no scalar variant, so it's represented as a single-
            # element Flat vector instead, matching how the rest of this
            # codebase already represents standalone scalar values (e.g.
            # AddLayer's broadcast constant).
            output_shape = dims_to_tensor_shape((1,))
        else:
            output_shape = dims_to_tensor_shape(output_dims)

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
