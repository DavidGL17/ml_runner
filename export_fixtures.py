import json
import os
from typing import Union

import torch
import torch.nn as nn
import torch.onnx as torch_onnx

import onnxruntime as ort
import onnx

from ml_runner_exporter.onnx_exporter import export_onnx
from python_fixtures.dense_fixtures import SimpleLinearModel
from python_fixtures.conv_fixtures import SimpleConvOnlyModel, ConvFlattenModel, FullConvModel, MaxPoolOnlyModel, MaxPoolPaddedModel, ConvPoolModel
from python_fixtures.activation_fixtures import ActivationModel
from python_fixtures.rnn_fixtures import (
    make_simple_rnn_only_model,
    make_simple_gru_only_model,
    make_simple_gru_return_sequences_model,
    make_simple_rnn_return_sequences_model,
    make_simple_lstm_only_model,
    make_simple_lstm_return_sequences_model,
)

from python_fixtures.tensor_op_fixtures import (
    ConcatModel,
    AddModel,
    make_add_broadcast_model,
    make_add_constant_model,
    make_expand_model,
    make_gather_model,
    make_gather_scalar_index_model,
    make_gather_negative_index_model,
    make_squeeze_model,
    make_unsqueeze_model,
    make_transpose_model,
)

fixtures_path = "tests/fixtures"

if not os.path.exists(fixtures_path):
    os.makedirs(fixtures_path)


def export_model(name: str, model: Union[nn.Module, onnx.ModelProto], input_shape):
    tmp_model_path = "temporary_model.onnx"

    # input_shape is one of:
    #   int              -> flat feature count (dense-only models); batch
    #                        prepended as dim 0: (1, features)
    #   (C, H, W) tuple   -> conv models; batch prepended as dim 0:
    #                        (1, C, H, W)
    #   (seq_len, F) tuple -> RNN/GRU models; batch goes in the *middle*,
    #                        matching nn.RNN/nn.GRU's batch_first=False (and
    #                        the ONNX RNN/GRU op's own) layout: (seq_len, 1, F)
    if isinstance(input_shape, list):
        #   list -> exact tensor shape, used as-is (no batch dim added);
        #           for generic tensor-op models
        dummy_input_data = torch.randn(*input_shape)
    elif isinstance(input_shape, tuple) and len(input_shape) == 3:
        dummy_input_data = torch.randn(1, *input_shape)
    elif isinstance(input_shape, tuple) and len(input_shape) == 2:
        seq_len, feature_size = input_shape
        dummy_input_data = torch.randn(seq_len, 1, feature_size)
    else:
        dummy_input_data = torch.randn(1, input_shape)

    if isinstance(model, nn.Module):
        # Avoids the "exporting a model while it is in training mode"
        # warning; doesn't change behavior here since none of these models
        # use dropout/batchnorm, but it's the right default for exported
        # fixtures.
        model.eval()
        torch_onnx.export(model, dummy_input_data, tmp_model_path, export_params=True, opset_version=17, dynamo=False)
        with torch.no_grad():
            output = model(dummy_input_data).numpy()
    else:
        # Already an ONNX model (e.g. hand-built via onnx.helper for ops
        # torch.onnx.export doesn't cover yet) - save it as-is and run it
        # through onnxruntime to get the reference output instead of a
        # PyTorch forward pass. From here on both paths are identical.
        onnx.save(model, tmp_model_path)
        input_name = model.graph.input[0].name
        sess = ort.InferenceSession(tmp_model_path)
        output = sess.run(None, {input_name: dummy_input_data.numpy()})[0]

    output_model = export_onnx(tmp_model_path)
    model_output = {
        "model": output_model,
        # Tensor::data on the Rust side is always flat (row-major) regardless
        # of TensorShape, so flatten both input and output fully here rather
        # than relying on tolist()[0], which would leave singleton/batch dims
        # in for conv and RNN/GRU shapes.
        "test_input": dummy_input_data.flatten().tolist(),
        "test_output": output.flatten().tolist(),
    }

    with open(os.path.join(fixtures_path, name), "w") as f:
        print(f"Exporting model: {name}")
        json.dump(model_output, f, indent=2)


fixtures = [
    {
        "name": "dense_simple_model.json",
        "model": SimpleLinearModel(10, 5, 1),
        "input_shape": 10,
    },
    {
        "name": "dense_long_model.json",
        "model": SimpleLinearModel(10, 5, 20),
        "input_shape": 10,
    },
    {
        "name": "dense_large_model.json",
        "model": SimpleLinearModel(100, 100, 5),
        "input_shape": 100,
    },
    {
        "name": "activation_all_types_model.json",
        "model": ActivationModel(10, 5),
        "input_shape": 10,
    },
    {
        "name": "conv_simple_model.json",
        "model": SimpleConvOnlyModel(),
        "input_shape": (1, 4, 4),
    },
    {
        "name": "conv_flatten_model.json",
        "model": ConvFlattenModel(),
        "input_shape": (1, 4, 4),
    },
    {
        "name": "conv_flatten_dense_activation_model.json",
        "model": FullConvModel(5),
        "input_shape": (1, 4, 4),
    },
    {
        "name": "rnn_simple_model.json",
        "model": make_simple_rnn_only_model(input_size=3, hidden_size=5, seq_len=4),
        "input_shape": (4, 3),  # (seq_len, input_size)
    },
    {
        "name": "rnn_return_sequences_model.json",
        "model": make_simple_rnn_return_sequences_model(input_size=3, hidden_size=5, seq_len=4),
        "input_shape": (4, 3),
    },
    {
        "name": "gru_simple_model.json",
        "model": make_simple_gru_only_model(input_size=3, hidden_size=5, seq_len=4),
        "input_shape": (4, 3),
    },
    {
        "name": "gru_return_sequences_model.json",
        "model": make_simple_gru_return_sequences_model(input_size=3, hidden_size=5, seq_len=4),
        "input_shape": (4, 3),
    },
    {
        "name": "lstm_simple_model.json",
        "model": make_simple_lstm_only_model(input_size=3, hidden_size=5, seq_len=4),
        "input_shape": (4, 3),
    },
    {
        "name": "lstm_return_sequences_model.json",
        "model": make_simple_lstm_return_sequences_model(input_size=3, hidden_size=5, seq_len=4),
        "input_shape": (4, 3),
    },
    {"name": "add_model.json", "model": AddModel(10), "input_shape": 10},
    {"name": "add_constant_model.json", "model": make_add_constant_model(), "input_shape": [1, 10]},
    {"name": "add_broadcast_model.json", "model": make_add_broadcast_model(), "input_shape": [1, 10]},
    {"name": "concat_model.json", "model": ConcatModel(10, 5), "input_shape": 10},
    {"name": "expand_model.json", "model": make_expand_model(), "input_shape": [1, 3, 1]},
    {"name": "gather_model.json", "model": make_gather_model(), "input_shape": [1, 5, 3]},
    {"name": "gather_scalar_index_model.json", "model": make_gather_scalar_index_model(), "input_shape": [1, 5, 3]},
    {"name": "gather_negative_index_model.json", "model": make_gather_negative_index_model(), "input_shape": [1, 5, 3]},
    {"name": "squeeze_model.json", "model": make_squeeze_model(), "input_shape": [1, 3, 1, 4]},
    {"name": "unsqueeze_model.json", "model": make_unsqueeze_model(), "input_shape": [1, 3, 4]},
    {"name": "transpose_model.json", "model": make_transpose_model(), "input_shape": [1, 2, 3, 4]},
    {
        "name": "maxpool_simple_model.json",
        "model": MaxPoolOnlyModel(),
        "input_shape": (2, 4, 4),
    },
    {
        "name": "maxpool_padded_model.json",
        "model": MaxPoolPaddedModel(),
        "input_shape": (2, 6, 6),
    },
    {
        "name": "conv_maxpool_flatten_dense_model.json",
        "model": ConvPoolModel(5),
        "input_shape": (1, 4, 4),
    },
]


for fixture in fixtures:
    export_model(fixture["name"], fixture["model"], fixture["input_shape"])
