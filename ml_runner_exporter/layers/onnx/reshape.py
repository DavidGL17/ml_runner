from onnx import NodeProto

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.utils import dims_to_tensor_shape


class ReshapeLayerParser(LayerParser):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        super().__init__("reshape")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "input_shape": dims_to_tensor_shape(self.input_shape),
            "output_shape": dims_to_tensor_shape(self.output_shape),
        }

    @classmethod
    def reshape_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict, rust_dims: dict) -> "ReshapeLayerParser":
        """A general row-major reshape. Like UnsqueezeLayerParser, the target
        shape is read from ONNX's own shape inference rather than the
        `shape` input (which may contain -1/0 placeholders).

        Only reshapes that keep the leading batch dimension in place are
        supported, since the batch dim is dropped on the Rust side. The
        input's runtime shape comes from `rust_dims` when an upstream
        recurrent layer registered one (see squeeze.py).
        """
        name = node.name or "Reshape"
        onnx_in = tuple(tensor_shapes.get(node.input[0], ()))
        onnx_out = tuple(tensor_shapes.get(node.output[0], ()))
        if len(onnx_in) < 2 or len(onnx_out) < 2:
            raise ValueError(f"Reshape node '{name}' maps {onnx_in} -> {onnx_out}; both sides need a leading batch dim")
        if onnx_in[0] != onnx_out[0]:
            raise ValueError(f"Reshape node '{name}' maps {onnx_in} -> {onnx_out}; reshapes that move the batch dim aren't supported")

        entry = rust_dims.get(node.input[0])
        rust_in = tuple(entry["shape"]) if entry is not None else onnx_in[1:]
        return cls(rust_in, onnx_out[1:])
