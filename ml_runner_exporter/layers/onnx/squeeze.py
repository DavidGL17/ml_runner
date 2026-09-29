from onnx import NodeProto
from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape

RECURRENT_OUTPUT_PHANTOM_AXES = {"Y": (1, 2), "Y_h": (0, 1), "Y_c": (0, 1)}


def register_recurrent_output(rust_dims: dict, output_name: str, onnx_role: str, rust_shape: tuple) -> None:
    if not output_name:
        return
    rust_dims[output_name] = {"shape": tuple(rust_shape), "phantom_axes": set(RECURRENT_OUTPUT_PHANTOM_AXES[onnx_role])}


def resolve_squeeze_axes(node: NodeProto, weights: dict) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == "axes":
            return list(attr.ints)
    if len(node.input) > 1 and node.input[1]:
        if node.input[1] not in weights:
            raise ValueError(f"Squeeze node '{node.name}' has non-constant 'axes' '{node.input[1]}'")
        return [int(a) for a in weights[node.input[1]].flatten()]
    return None


def _normalize_axes(shape: tuple, axes: list[int] | None, node_name: str) -> set[int]:
    rank = len(shape)
    if axes is None:
        return {i for i, d in enumerate(shape) if d == 1}
    normalized = {a + rank if a < 0 else a for a in axes}
    for a in normalized:
        if not 0 <= a < rank:
            raise ValueError(f"Squeeze node '{node_name}': axis {a} is out of range for shape {shape}")
        if shape[a] != 1:
            raise ValueError(f"Squeeze node '{node_name}': can't squeeze axis {a} of shape {shape} (size {shape[a]} != 1)")
    return normalized


def squeezed_shape(shape: tuple, axes: set[int]) -> tuple:
    return tuple(d for i, d in enumerate(shape) if i not in axes)


class SqueezeLayerParser(LayerParser):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        super().__init__("squeeze")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "input_shape": dims_to_tensor_shape(self.input_shape),
            "output_shape": dims_to_tensor_shape(self.output_shape),
        }

    @classmethod
    def squeeze_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict, weights: dict, rust_dims: dict | None = None) -> "SqueezeLayerParser":
        name = node.name or "Squeeze"
        in_name = node.input[0]
        onnx_in = tensor_shapes.get(in_name)
        if onnx_in is None:
            raise ValueError(f"Could not determine the input shape for Squeeze node '{name}'")
        onnx_in = tuple(onnx_in)
        requested = _normalize_axes(onnx_in, resolve_squeeze_axes(node, weights), name)

        rust_dims = rust_dims if rust_dims is not None else {}
        entry = rust_dims.get(in_name)
        if entry is not None:
            phantom = entry["phantom_axes"]
            rust_in = entry["shape"]
            real_requested = requested - phantom
            rust_axes = {i - sum(1 for p in phantom if p < i) for i in real_requested}
            for a in rust_axes:
                if not (0 <= a < len(rust_in)) or rust_in[a] != 1:
                    raise ValueError(
                        f"Squeeze node '{name}': can't squeeze runtime axis {a} of shape {rust_in} " f"(from ONNX shape {onnx_in}, axes {sorted(requested)})"
                    )
            rust_out = squeezed_shape(rust_in, rust_axes)

            # Propagate: this op's OWN output may still carry phantom axes
            # that this particular Squeeze didn't remove (PyTorch's legacy
            # exporter often squeezes num_directions and batch in two
            # separate ops). Re-index the survivors into the output
            # tensor's own axis numbering so a later Squeeze (or anything
            # else that consults rust_dims) sees the truth instead of
            # falling back to the "leading axis is batch" assumption.
            remaining_phantom = phantom - requested
            new_phantom = {p - sum(1 for r in requested if r < p) for p in remaining_phantom}
            out_name = node.output[0] if node.output else ""
            if out_name:
                rust_dims[out_name] = {"shape": rust_out, "phantom_axes": new_phantom}
        else:
            if len(onnx_in) < 2:
                raise ValueError(f"Squeeze node '{name}': can't drop a batch dim from shape {onnx_in}")
            rust_in = onnx_in[1:]
            if 0 in requested:
                raise ValueError(f"Squeeze node '{name}': squeezing the batch axis isn't supported")
            rust_axes = {a - 1 for a in requested}
            rust_out = squeezed_shape(rust_in, rust_axes)

        if not rust_out:
            raise ValueError(f"Squeeze node '{name}': result would be a 0-D scalar, which isn't supported")
        return cls(rust_in, rust_out)
