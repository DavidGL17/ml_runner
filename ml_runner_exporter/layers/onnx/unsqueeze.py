from typing import Any, Self

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


def _strip_batch_dim(dims: tuple) -> tuple:
    """See gather.py for the full reasoning: a tensor already at rank <= 1
    can't be a real batched activation under this codebase's constraints
    (the Rust runtime only supports 1-3 dim per-sample tensors, so a real
    activation's ONNX-recorded, batch-included shape always has rank 2-4),
    so it's used as-is instead of dropping its only dimension.
    """
    return dims[1:] if len(dims) > 1 else dims


def _dims_or_scalar_fallback(dims: tuple) -> tuple:
    """A genuine 0-D scalar (e.g. produced by a Gather with a scalar index
    - see gather.py) has no dims for `dims_to_tensor_shape` to describe.
    This runtime has no true scalar type though - standalone scalar values
    are represented as a single-element Flat vector instead (again, see
    gather.py) - so an empty dims tuple is mapped the same way here, for
    consistency with whatever upstream node actually produced it. This
    matters a lot for Unsqueeze specifically, since "turn a scalar into a
    1-element vector" is exactly what the common
    `Shape -> Gather(scalar index) -> Unsqueeze` pattern does.
    """
    return dims if len(dims) > 0 else (1,)


class UnsqueezeLayerParser(LayerParser):
    def __init__(self, input_shape: dict, output_shape: dict) -> None:
        super().__init__("unsqueeze")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

    @classmethod
    def unsqueeze_layer_from_onnx(cls, node: Any, tensor_shapes: dict) -> Self:
        """
        node: The ONNX node object.
        tensor_shapes: The dictionary containing shapes of all tensors in the graph.

        Deliberately doesn't look at the `axes` attribute/input at all:
        ONNX's own shape_inference has already computed the correct output
        shape (with the new size-1 axis in the right place) and recorded
        it in `tensor_shapes`, so re-deriving that from `axes` here would
        just be a second, more error-prone way to compute something we
        already have - see also ShapeLayerParser/GatherLayerParser, which
        take the same approach.
        """
        input_name = node.input[0]
        output_name = node.output[0]

        input_dims = _dims_or_scalar_fallback(_strip_batch_dim(tuple(tensor_shapes.get(input_name, ()))))
        output_dims = _dims_or_scalar_fallback(_strip_batch_dim(tuple(tensor_shapes.get(output_name, ()))))

        input_shape = dims_to_tensor_shape(input_dims)
        output_shape = dims_to_tensor_shape(output_dims)

        return cls(input_shape=input_shape, output_shape=output_shape)
