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


class ConcatLayerParser(LayerParser):
    def __init__(self, axis: int, input_shapes: list[dict], output_shape: dict) -> None:
        super().__init__("concat")
        self.axis = axis
        self.input_shapes = input_shapes
        self.output_shape = output_shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "axis": self.axis,
            "input_shapes": self.input_shapes,
            "output_shape": self.output_shape,
        }

    @classmethod
    def concat_layer_from_onnx(cls, node: Any, tensor_shapes: dict) -> Self:
        """
        node: The ONNX node object (needs the 'axis' attribute, and the
              full list of real tensor inputs being joined).
        tensor_shapes: The dictionary containing shapes of all tensors in
              the graph.

        Unlike ShapeLayerParser/UnsqueezeLayerParser, this can't just read
        an already-inferred output shape and call it done: the Rust
        ConcatLayer needs an explicit per-sample `axis` to actually
        perform the join at runtime, since the operands aren't just
        reshaped versions of each other.
        """
        if len(node.input) < 1:
            raise ValueError(f"Concat node '{node.name}' has no inputs")

        raw_input_dims = [tuple(tensor_shapes.get(name, ())) for name in node.input]

        # ONNX requires every Concat operand to share the same rank, so
        # checking the first is enough to know whether these are genuine
        # batch-carrying activations (rank > 1) or batch-less "meta"
        # tensors (rank <= 1, e.g. shape-vectors) - see _strip_batch_dim.
        onnx_rank = len(raw_input_dims[0])
        had_batch_dim = onnx_rank > 1

        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), None)
        if axis is None:
            raise ValueError(f"Concat node '{node.name}' is missing the required 'axis' attribute")
        axis = axis % onnx_rank if onnx_rank else axis

        # Convert from the ONNX axis (expressed against the full,
        # batch-included rank) to the per-sample axis this runtime's
        # TensorShape actually uses.
        per_sample_axis = axis - 1 if had_batch_dim and axis > 0 else axis

        input_shapes = [dims_to_tensor_shape(_strip_batch_dim(dims)) for dims in raw_input_dims]

        output_dims = _strip_batch_dim(tuple(tensor_shapes.get(node.output[0], ())))
        output_shape = dims_to_tensor_shape(output_dims)

        return cls(axis=per_sample_axis, input_shapes=input_shapes, output_shape=output_shape)
