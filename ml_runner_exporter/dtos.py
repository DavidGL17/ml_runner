from dataclasses import dataclass

from numpy import ndarray
from onnx import NodeProto


@dataclass
class NodeContext:
    """Everything a per-op-type handler might need."""

    node: NodeProto
    tensor_shapes: dict
    weights: dict
    node_weights: list
    weight_matrix: ndarray | None
    bias_vector: ndarray | None
