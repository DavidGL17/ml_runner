import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import helper, numpy_helper, TensorProto

# ---------------------------------------------------------------------------
# PyTorch modules (traced export gives exactly the node we want)
# ---------------------------------------------------------------------------


# Residual connection: Gemm(x) + x -> Add with two dynamic inputs.
class AddModel(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size)

    def forward(self, x):
        return self.linear(x) + x


# Concat of the raw input with a transformed copy along the feature axis.
# Output features = input_size + output_size.
class ConcatModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.cat([x, self.linear(x)], dim=1)


# ---------------------------------------------------------------------------
# Hand-built ONNX models
# ---------------------------------------------------------------------------


def _int64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def _build_single_node_model(name, node, in_shape, out_shape, initializers=(), out_type=TensorProto.FLOAT):
    graph = helper.make_graph(
        [node],
        name,
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("Y", out_type, out_shape)],
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# (1, 10) + constant of shape (1, 10): same-shape constant add.
def make_add_constant_model():
    rng = np.random.default_rng()
    const = rng.standard_normal((1, 10)).astype(np.float32)
    node = helper.make_node("Add", ["X", "C"], ["Y"], name="add")
    return _build_single_node_model(
        "add_constant_model",
        node,
        [1, 10],
        [1, 10],
        [numpy_helper.from_array(const, name="C")],
    )


# (1, 10) + constant of shape (10,): constant broadcast over the batch dim.
def make_add_broadcast_model():
    rng = np.random.default_rng()
    const = rng.standard_normal((10,)).astype(np.float32)
    node = helper.make_node("Add", ["X", "C"], ["Y"], name="add")
    return _build_single_node_model(
        "add_broadcast_model",
        node,
        [1, 10],
        [1, 10],
        [numpy_helper.from_array(const, name="C")],
    )


# (1, 3, 1) -> (1, 3, 4)
def make_expand_model():
    node = helper.make_node("Expand", ["X", "shape"], ["Y"], name="expand")
    return _build_single_node_model("expand_model", node, [1, 3, 1], [1, 3, 4], [_int64("shape", [1, 3, 4])])


def _make_gather_model(name, indices, out_shape):
    node = helper.make_node("Gather", ["X", "indices"], ["Y"], axis=1, name="gather")
    return _build_single_node_model(
        name,
        node,
        [1, 5, 3],
        out_shape,
        [numpy_helper.from_array(np.asarray(indices, dtype=np.int64), name="indices")],
    )


def make_gather_model():
    return _make_gather_model("gather_model", [4, 0, 2], [1, 3, 3])


def make_gather_scalar_index_model():
    return _make_gather_model("gather_scalar_index_model", 2, [1, 3])


def make_gather_negative_index_model():
    return _make_gather_model("gather_negative_index_model", -1, [1, 3])


# (1, 3, 1, 4) -> (1, 3, 4)
def make_squeeze_model():
    node = helper.make_node("Squeeze", ["X", "axes"], ["Y"], name="squeeze")
    return _build_single_node_model("squeeze_model", node, [1, 3, 1, 4], [1, 3, 4], [_int64("axes", [2])])


# (1, 3, 4) -> (1, 3, 1, 4)
def make_unsqueeze_model():
    node = helper.make_node("Unsqueeze", ["X", "axes"], ["Y"], name="unsqueeze")
    return _build_single_node_model("unsqueeze_model", node, [1, 3, 4], [1, 3, 1, 4], [_int64("axes", [2])])


# (1, 2, 3, 4) with perm [0, 3, 1, 2] -> (1, 4, 2, 3).
def make_transpose_model():
    node = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 3, 1, 2], name="transpose")
    return _build_single_node_model("transpose_model", node, [1, 2, 3, 4], [1, 4, 2, 3])
