import math
from typing import Self

from onnx import NodeProto

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


def resolve_squeeze_axes(node: NodeProto, weights: dict) -> list[int] | None:
    """`axes` is an attribute on opset < 13, or an optional second (constant)
    input tensor on opset >= 13 - handle either, like `_resolve_unsqueeze_axes`.
    Returns None when no axes are given, which in ONNX means "squeeze every
    size-1 dimension".
    """
    for attr in node.attribute:
        if attr.name == "axes":
            return list(attr.ints)
    if len(node.input) > 1 and node.input[1]:
        if node.input[1] not in weights:
            raise ValueError(f"Squeeze node '{node.name}' has non-constant 'axes' '{node.input[1]}'")
        return [int(a) for a in weights[node.input[1]].flatten()]
    return None


def squeezed_shape(shape: tuple, axes: list[int] | None, node_name: str) -> tuple:
    """The ONNX-level shape after squeezing `axes` (or every size-1 dim if None)."""
    rank = len(shape)
    if axes is None:
        return tuple(d for d in shape if d != 1)
    normalized = {a + rank if a < 0 else a for a in axes}
    for a in normalized:
        if not 0 <= a < rank:
            raise ValueError(f"Squeeze node '{node_name}': axis {a} is out of range for shape {shape}")
        if shape[a] != 1:
            raise ValueError(f"Squeeze node '{node_name}': can't squeeze axis {a} of shape {shape} (size {shape[a]} != 1)")
    return tuple(d for i, d in enumerate(shape) if i not in normalized)


class SqueezeLayerParser(LayerParser):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        super().__init__("squeeze")
        # Both already in the Rust runtime's representation (see squeeze_layer_from_onnx).
        self.input_shape = input_shape
        self.output_shape = output_shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "input_shape": dims_to_tensor_shape(self.input_shape),
            "output_shape": dims_to_tensor_shape(self.output_shape),
        }

    @classmethod
    def squeeze_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict, weights: dict, rust_dims: dict | None = None) -> Self:
        """The Rust runtime has no batch dimension, so a tensor's runtime shape is
        its ONNX shape minus the leading batch dim - except for an RNN/GRU/LSTM's
        `Y` / `Y_h`, whose ONNX shapes carry extra num_directions/batch dims that
        the Rust layers don't produce (`Y_h` is (1, batch, hidden) in ONNX but a
        `Flat(hidden)` at runtime). `rust_dims` maps those tensor names to their
        true runtime dims; anything not in it uses the leading-dim rule.

        The squeezed output follows the leading-dim rule too, because that's what
        every downstream parser will apply to it.
        """
        name = node.name or "Squeeze"
        in_name = node.input[0]
        onnx_in = tensor_shapes.get(in_name)
        if onnx_in is None:
            raise ValueError(f"Could not determine the input shape for Squeeze node '{name}'")
        onnx_in = tuple(onnx_in)

        onnx_out = squeezed_shape(onnx_in, resolve_squeeze_axes(node, weights), name)

        rust_in = tuple((rust_dims or {}).get(in_name, onnx_in[1:] if len(onnx_in) > 1 else onnx_in))
        rust_out = onnx_out[1:] if len(onnx_out) > 1 else onnx_out

        if not rust_in or not rust_out or math.prod(rust_in) != math.prod(rust_out):
            raise ValueError(
                f"Squeeze node '{name}': runtime shapes {rust_in} -> {rust_out} don't hold the same number of elements "
                f"(ONNX {onnx_in} -> {onnx_out}). Squeezing the batch axis of an ordinary tensor isn't supported."
            )
        return cls(rust_in, rust_out)
