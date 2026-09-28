from ml_runner_exporter.dtos import NodeContext
from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.layers.activation import ActivationLayerParser
from ml_runner_exporter.layers.conv import Conv2DLayerParser
from ml_runner_exporter.layers.flatten import FlattenLayerParser
from ml_runner_exporter.layers.onnx.identity import IdentityLayerParser
from ml_runner_exporter.layers.linear import LinearLayerParser
from ml_runner_exporter.layers.maxpool import MaxPool2DLayerParser
from ml_runner_exporter.layers.onnx.add import AddLayerParser
from ml_runner_exporter.layers.onnx.expand import ExpandLayerParser
from ml_runner_exporter.layers.onnx.concat import ConcatLayerParser
from ml_runner_exporter.layers.onnx.gather import GatherLayerParser
from ml_runner_exporter.layers.onnx.shape import ShapeLayerParser
from ml_runner_exporter.layers.onnx.transpose import TransposeLayerParser
from ml_runner_exporter.layers.onnx.squeeze import SqueezeLayerParser
from ml_runner_exporter.layers.onnx.unsqueeze import UnsqueezeLayerParser
from ml_runner_exporter.layers.rnn import GRULayerParser, LSTMLayerParser, RNNLayerParser


def _handle_add(ctx: NodeContext) -> LayerParser:
    output_shape = ctx.tensor_shapes.get(ctx.node.output[0], ())
    return AddLayerParser.add_layer_from_onnx(ctx.node_weights, output_shape)


def _handle_concat(ctx: NodeContext) -> LayerParser:
    return ConcatLayerParser.concat_layer_from_onnx(ctx.node, ctx.tensor_shapes)


def _handle_expand(ctx: NodeContext) -> LayerParser:
    return ExpandLayerParser.expand_layer_from_onnx(ctx.node, ctx.tensor_shapes, ctx.weights)


def _handle_transpose(ctx: NodeContext) -> LayerParser:
    return TransposeLayerParser.transpose_layer_from_onnx(ctx.node, ctx.tensor_shapes)


def _handle_squeeze(ctx: NodeContext) -> LayerParser:
    return SqueezeLayerParser.squeeze_layer_from_onnx(ctx.node, ctx.tensor_shapes, ctx.node_weights)


def _handle_unsqueeze(ctx: NodeContext) -> LayerParser:
    return UnsqueezeLayerParser.unsqueeze_layer_from_onnx(ctx.node, ctx.tensor_shapes)


def _handle_shape(ctx: NodeContext) -> LayerParser:
    return ShapeLayerParser.shape_layer_from_onnx(ctx.node, ctx.tensor_shapes)


def _handle_gather(ctx: NodeContext) -> LayerParser:
    return GatherLayerParser.gather_layer_from_onnx(ctx.node, ctx.tensor_shapes, ctx.weights)


def _handle_gemm(ctx: NodeContext) -> LayerParser:
    return LinearLayerParser.linear_layer_from_onnx(ctx.weight_matrix, ctx.bias_vector)


def _handle_conv(ctx: NodeContext) -> LayerParser:
    return Conv2DLayerParser.conv2d_layer_from_onnx(ctx.node, ctx.tensor_shapes, ctx.weight_matrix, ctx.bias_vector)


def _handle_maxpool(ctx: NodeContext) -> LayerParser:
    return MaxPool2DLayerParser.maxpool_layer_from_onnx(ctx.node, ctx.tensor_shapes)


def _handle_flatten_reshape(ctx: NodeContext) -> LayerParser:
    return FlattenLayerParser.flatten_layer_from_onnx(ctx.node, ctx.tensor_shapes, ctx.weights)


def _handle_identity(ctx: NodeContext) -> LayerParser:
    return IdentityLayerParser.identity_layer_from_onnx(ctx.node, ctx.tensor_shapes)


def _handle_activation(ctx: NodeContext) -> LayerParser:
    return ActivationLayerParser.activation_layer_from_onnx(ctx.node, ctx.tensor_shapes)


def _handle_rnn(ctx: NodeContext) -> LayerParser:
    return RNNLayerParser.rnn_layer_from_onnx(ctx.node, ctx.tensor_shapes, ctx.weights)


def _handle_gru(ctx: NodeContext) -> LayerParser:
    return GRULayerParser.gru_layer_from_onnx(ctx.node, ctx.tensor_shapes, ctx.weights)


def _handle_lstm(ctx: NodeContext) -> LayerParser:
    return LSTMLayerParser.lstm_layer_from_onnx(ctx.node, ctx.tensor_shapes, ctx.weights)


OP_HANDLERS = {
    "Add": _handle_add,
    "Concat": _handle_concat,
    "Expand": _handle_expand,
    "Transpose": _handle_transpose,
    "Unsqueeze": _handle_unsqueeze,
    "Squeeze": _handle_squeeze,
    "Shape": _handle_shape,
    "Gather": _handle_gather,
    "Gemm": _handle_gemm,
    "Conv": _handle_conv,
    "MaxPool": _handle_maxpool,
    "Flatten": _handle_flatten_reshape,
    "Reshape": _handle_flatten_reshape,
    "Identity": _handle_identity,
    "Relu": _handle_activation,
    "Sigmoid": _handle_activation,
    "Tanh": _handle_activation,
    "Softmax": _handle_activation,
    "RNN": _handle_rnn,
    "GRU": _handle_gru,
    "LSTM": _handle_lstm,
}
