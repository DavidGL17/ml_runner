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
    def gather_layer_from_onnx(cls, node: Any, tensor_shapes: dict, weights: dict[str, ndarray], rust_dims: dict | None = None) -> Self:  # noqa: PLR0912
        """
        node: The ONNX node object (needed for 'axis' attribute).
        tensor_shapes: Dictionary of all tensor shapes in the graph.
        weights: The dictionary of constant tensors (initializers).
        rust_dims: tensors (by name) whose true runtime shape isn't
            "ONNX shape minus the leading batch dim" - currently populated
            for an RNN/GRU/LSTM's Y/Y_h and anything descended from one
            through a chain of Squeeze ops (see squeeze.py). ONNX's own
            shape_inference can't always be trusted for such a tensor (it
            runs once, up front, before this exporter's own constant
            folding, and doesn't know the Rust runtime never materializes
            the num_directions/batch axes to begin with) - when a name is
            registered here, its recorded shape is used as-is instead of
            re-deriving it from `tensor_shapes`.

        `axis` is used throughout this function exactly as it's stored on
        the returned layer (`self.axis = axis`, unmodified) - i.e. it
        indexes directly into `input_dims` (already batch-stripped for an
        ordinary tensor, or already batch-free for a tracked one), matching
        whatever space Rust's `GatherLayer::forward` expects. There's no
        separate "ONNX axis" vs "Rust axis" here; this file has only ever
        used one numbering, consistently, for both.
        """
        rust_dims = rust_dims or {}
        input_name = node.input[0]
        indices_name = node.input[1]

        # 2. Get shapes - see `_strip_batch_dim` for why this isn't a
        # blind `[1:]` the way a Conv/Flatten input can be.
        entry = rust_dims.get(input_name)
        if entry is not None:
            input_dims = entry["shape"]
        else:
            input_dims = _strip_batch_dim(tuple(tensor_shapes.get(input_name, ())))
        input_shape = dims_to_tensor_shape(input_dims)

        # 3. Extract 'axis' attribute
        axis = 0  # Default fallback
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
                break

        if entry is None and len(tensor_shapes.get(input_name, ())) > 1:
            rank = len(tensor_shapes[input_name])
            if axis < 0:
                axis += rank
            if axis == 0:
                raise ValueError(f"Gather node '{node.name}': gathering along the batch axis isn't supported")
            axis -= 1

        # 4. Try to extract indices if they are constant.
        # ONNX (like numpy) allows negative indices, counting back from the
        # end of the gathered axis - e.g. PyTorch exports `out[:, -1, :]`
        # as a literal `-1` index. The Rust GatherLayer doesn't implement
        # that convention, so it's normalized here against the axis's
        # statically-known size (in `input_dims`'s own numbering, the same
        # one `axis` already indexes into), while it's still cheap to do so.
        indices_values = None
        if indices_name in weights:
            axis_size = input_dims[axis] if 0 <= axis < len(input_dims) else None
            indices_values = []
            for x in weights[indices_name].flatten():
                idx = float(x)
                if idx < 0:
                    if axis_size is None:
                        raise ValueError(f"Gather node '{node.name}': can't resolve negative index {idx} without a known size for axis {axis}")
                    idx += axis_size
                indices_values.append(idx)

        # 5. Output shape.
        # The Rust GatherLayer only supports single-index selection: it
        # always drops the gathered axis entirely (see its `output_shape`
        # test: D2{3,2} + axis=0 + a single index -> Flat(2)), regardless
        # of whether ONNX's `indices` tensor was rank-0 or a length-1
        # rank-1 tensor. When we know the input's real shape (`entry` above,
        # or indices are a known single constant) we derive the output the
        # same way Rust will actually compute it, rather than trusting a
        # second `tensor_shapes` lookup that's just as likely to be stale
        # as the input one was.
        can_derive_directly = entry is not None or (indices_values is not None and len(indices_values) == 1)
        if can_derive_directly:
            if not 0 <= axis < len(input_dims):
                raise ValueError(f"Gather node '{node.name}': axis {axis} is out of range for input shape {input_dims}")
            output_dims = tuple(d for i, d in enumerate(input_dims) if i != axis)
            if not output_dims:
                # Gathering the only axis of a 1-D tensor leaves a scalar;
                # represented as a single-element Flat, same as the
                # existing rank-0 special case below.
                output_dims = (1,)
        else:
            output_dims = _strip_batch_dim(tuple(tensor_shapes.get(node.output[0], ())))
            if len(output_dims) == 0:
                # Per ONNX semantics, gathering with a scalar (rank-0) index
                # removes the gathered axis entirely rather than keeping it as
                # size 1 - e.g. pulling a single dynamic batch size out of a
                # Shape node's output leaves a genuine 0-D scalar. TensorShape
                # has no scalar variant, so it's represented as a single-
                # element Flat vector instead, matching how the rest of this
                # codebase already represents standalone scalar values (e.g.
                # AddLayer's broadcast constant).
                output_dims = (1,)

        output_shape = dims_to_tensor_shape(output_dims)

        # Propagate, in case something else downstream also needs this
        # tensor's true runtime shape (e.g. another Gather, or a Squeeze).
        out_name = node.output[0] if node.output else ""
        if out_name and can_derive_directly:
            rust_dims[out_name] = {"shape": output_dims, "phantom_axes": set()}

        return cls(input_shape=input_shape, output_shape=output_shape, axis=axis, indices=indices_values)
