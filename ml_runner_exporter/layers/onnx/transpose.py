from typing import Self

from onnx import NodeProto

from ml_runner_exporter.layer import LayerParser


class TransposeLayerParser(LayerParser):
    def __init__(self, input_shape: tuple, perm: list[int]) -> None:
        super().__init__("transpose")
        self.input_shape = input_shape
        self.perm = perm

    def to_dict(self) -> dict:
        return {
            "type": self.layer_type,
            "input_shape": self.input_shape,
            "perm": self.perm,
        }

    @classmethod
    def transpose_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict) -> Self:
        """
        node: The ONNX node object (needed to access attributes like 'perm')
        tensor_shapes: The dictionary containing shapes of all tensors in the graph
        """
        # 1. Get the shape of the input tensor. Transpose always has exactly one input
        input_name = node.input[0]
        input_shape = tensor_shapes.get(input_name, ())

        # 2. Extract the 'perm' attribute.  In ONNX, 'perm' is an attribute of the node, not a tensor in the graph
        perm = []
        for attr in node.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
                break

        # Note: If 'perm' is not provided in ONNX, it defaults to reversing the axes, but in most exported models, it is explicitly present.

        return cls(input_shape=input_shape, perm=perm)
