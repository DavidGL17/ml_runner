import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def _init_uniform(shape, hidden_size, rng):
    stdv = 1.0 / np.sqrt(hidden_size)
    return rng.uniform(-stdv, stdv, size=shape).astype(np.float32)


def _reorder_gru_gates(w: np.ndarray) -> np.ndarray:
    """PyTorch gate order [r, z, n] -> ONNX gate order [z, r, n]."""
    r, z, n = np.split(w, 3, axis=0)
    return np.concatenate([z, r, n], axis=0)


def _build_rnn_or_gru_model(op_type, weight_ih, weight_hh, bias_ih, bias_hh, seq_len, input_size, hidden_size, return_sequences):
    if op_type == "GRU":
        weight_ih = _reorder_gru_gates(weight_ih)
        weight_hh = _reorder_gru_gates(weight_hh)
        bias_ih = _reorder_gru_gates(bias_ih)
        bias_hh = _reorder_gru_gates(bias_hh)

    W = weight_ih[np.newaxis, :, :]
    R = weight_hh[np.newaxis, :, :]
    B = np.concatenate([bias_ih, bias_hh])[np.newaxis, :]

    initializers = [
        numpy_helper.from_array(W, name="W"),
        numpy_helper.from_array(R, name="R"),
        numpy_helper.from_array(B, name="B"),
    ]

    # No Squeeze node here: keep the RNN/GRU op's native output as-is and
    # use it directly as the graph output. That's exactly what
    # torch.onnx.export() would emit too (same underlying ONNX op), and
    # your exporter's node-type parser only knows RNN/GRU/etc, not Squeeze.
    # The extra num_directions axis on Y is harmless since test_input/
    # test_output get flattened before being written to the fixture.
    y_out = "Y" if return_sequences else ""
    yh_out = "" if return_sequences else "Y_h"
    node_kwargs = {"hidden_size": hidden_size, "name": op_type.lower()}
    if op_type == "RNN":
        node_kwargs["activations"] = ["Tanh"]
    else:
        node_kwargs["linear_before_reset"] = 1  # matches PyTorch's GRU formula

    main_node = helper.make_node(op_type, inputs=["X", "W", "R", "B"], outputs=[y_out, yh_out], **node_kwargs)
    nodes = [main_node]

    if return_sequences:
        out_shape = [seq_len, 1, 1, hidden_size]  # (seq_len, num_directions, batch, hidden)
        out_name = "Y"
    else:
        out_shape = [1, 1, hidden_size]  # (num_directions, batch, hidden)
        out_name = "Y_h"

    graph = helper.make_graph(
        nodes,
        f"{op_type.lower()}_model",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [seq_len, 1, input_size])],
        outputs=[helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _random_rnn_weights(input_size, hidden_size, rng):
    return (
        _init_uniform((hidden_size, input_size), hidden_size, rng),
        _init_uniform((hidden_size, hidden_size), hidden_size, rng),
        _init_uniform((hidden_size,), hidden_size, rng),
        _init_uniform((hidden_size,), hidden_size, rng),
    )


def _random_gru_weights(input_size, hidden_size, rng):
    return (
        _init_uniform((3 * hidden_size, input_size), hidden_size, rng),
        _init_uniform((3 * hidden_size, hidden_size), hidden_size, rng),
        _init_uniform((3 * hidden_size,), hidden_size, rng),
        _init_uniform((3 * hidden_size,), hidden_size, rng),
    )


def make_simple_rnn_only_model(input_size, hidden_size, seq_len, seed=None):
    rng = np.random.default_rng(seed)
    weight_ih, weight_hh, bias_ih, bias_hh = _random_rnn_weights(input_size, hidden_size, rng)
    return _build_rnn_or_gru_model("RNN", weight_ih, weight_hh, bias_ih, bias_hh, seq_len, input_size, hidden_size, return_sequences=False)


def make_simple_rnn_return_sequences_model(input_size, hidden_size, seq_len, seed=None):
    rng = np.random.default_rng(seed)
    weight_ih, weight_hh, bias_ih, bias_hh = _random_rnn_weights(input_size, hidden_size, rng)
    return _build_rnn_or_gru_model("RNN", weight_ih, weight_hh, bias_ih, bias_hh, seq_len, input_size, hidden_size, return_sequences=True)


def make_simple_gru_only_model(input_size, hidden_size, seq_len, seed=None):
    rng = np.random.default_rng(seed)
    weight_ih, weight_hh, bias_ih, bias_hh = _random_gru_weights(input_size, hidden_size, rng)
    return _build_rnn_or_gru_model("GRU", weight_ih, weight_hh, bias_ih, bias_hh, seq_len, input_size, hidden_size, return_sequences=False)


def make_simple_gru_return_sequences_model(input_size, hidden_size, seq_len, seed=None):
    rng = np.random.default_rng(seed)
    weight_ih, weight_hh, bias_ih, bias_hh = _random_gru_weights(input_size, hidden_size, rng)
    return _build_rnn_or_gru_model("GRU", weight_ih, weight_hh, bias_ih, bias_hh, seq_len, input_size, hidden_size, return_sequences=True)
