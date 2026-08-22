from typing import Self

import numpy as np
from onnx import NodeProto

from ml_runner_exporter.layer import LayerParser
from ml_runner_exporter.layers.activation import ActivationTypes


class RNNLayerParser(LayerParser):

    def __init__(
        self,
        seq_len: int,
        input_size: int,
        hidden_size: int,
        weights_ih: list,
        weights_hh: list,
        bias_ih: list,
        bias_hh: list,
        activation_type: ActivationTypes,
        return_sequences: bool = False,
    ):
        super().__init__("rnn")
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Expected shape: weights_ih is (hidden_size, input_size),
        # weights_hh is (hidden_size, hidden_size).
        self.weights_ih = weights_ih
        self.weights_hh = weights_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.activation_type = activation_type
        self.return_sequences = return_sequences

    def to_dict(self) -> dict:
        flat_weights_ih = [float(w) for row in self.weights_ih for w in row]
        flat_weights_hh = [float(w) for row in self.weights_hh for w in row]

        return {
            "type": self.layer_type,
            "seq_len": self.seq_len,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "weights_ih": flat_weights_ih,
            "weights_hh": flat_weights_hh,
            "bias_ih": [float(b) for b in self.bias_ih],
            "bias_hh": [float(b) for b in self.bias_hh],
            "activation_type": self.activation_type.to_rust_id(),
            "return_sequences": self.return_sequences,
        }

    @classmethod
    def rnn_layer_from_onnx(cls, node, tensor_shapes: dict, weights: dict) -> Self:
        attrs = {a.name: a for a in node.attribute}

        direction = attrs["direction"].s.decode() if "direction" in attrs else "forward"
        if direction != "forward":
            raise ValueError(f"RNN node {node.name} uses direction={direction!r}; only 'forward' " "(single-direction) RNNs are supported")

        if "hidden_size" not in attrs:
            raise ValueError(f"RNN node {node.name} is missing the required hidden_size attribute")
        hidden_size = attrs["hidden_size"].i

        # ONNX default activation for RNN is Tanh.
        if "activations" in attrs and len(attrs["activations"].strings) > 0:
            activation_name = attrs["activations"].strings[0].decode()
        else:
            activation_name = "Tanh"
        activation_type = ActivationTypes.from_onnx_type(activation_name)

        # X: (seq_length, batch_size, input_size)
        input_tensor_name = node.input[0]
        input_shape = tensor_shapes.get(input_tensor_name)
        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"Could not determine a 3D (seq_length, batch, input_size) input shape " f"for RNN node {node.name}")
        seq_len, _batch, input_size = input_shape

        if len(node.input) < 3 or node.input[1] not in weights or node.input[2] not in weights:
            raise ValueError(f"RNN node {node.name} is missing its W/R weight tensors")

        # Drop the leading num_directions=1 dim.
        w = weights[node.input[1]][0]  # (hidden_size, input_size)
        r = weights[node.input[2]][0]  # (hidden_size, hidden_size)

        bias_ih = np.zeros(hidden_size, dtype=np.float32)
        bias_hh = np.zeros(hidden_size, dtype=np.float32)
        if len(node.input) > 3 and node.input[3] and node.input[3] in weights:
            # B: (2 * hidden_size,) = [Wb, Rb]
            b = weights[node.input[3]][0]
            bias_ih = b[:hidden_size]
            bias_hh = b[hidden_size:]

        # Y is the first output (every timestep's hidden state); Y_h is the
        # second (final hidden state only). ONNX leaves an output name empty
        # ("") when that particular output isn't consumed downstream.
        return_sequences = len(node.output) > 0 and node.output[0] != ""

        return cls(
            seq_len=seq_len,
            input_size=input_size,
            hidden_size=hidden_size,
            weights_ih=w.tolist(),
            weights_hh=r.tolist(),
            bias_ih=bias_ih.tolist(),
            bias_hh=bias_hh.tolist(),
            activation_type=activation_type,
            return_sequences=return_sequences,
        )


class GRULayerParser(LayerParser):

    def __init__(
        self,
        seq_len: int,
        input_size: int,
        hidden_size: int,
        weights_ir: list,
        weights_hr: list,
        bias_ir: list,
        bias_hr: list,
        weights_iz: list,
        weights_hz: list,
        bias_iz: list,
        bias_hz: list,
        weights_in: list,
        weights_hn: list,
        bias_in: list,
        bias_hn: list,
        recurrent_activation_type: ActivationTypes,
        activation_type: ActivationTypes,
        return_sequences: bool = False,
    ):
        super().__init__("gru")
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Reset gate
        self.weights_ir = weights_ir
        self.weights_hr = weights_hr
        self.bias_ir = bias_ir
        self.bias_hr = bias_hr

        # Update gate
        self.weights_iz = weights_iz
        self.weights_hz = weights_hz
        self.bias_iz = bias_iz
        self.bias_hz = bias_hz

        # Candidate state
        self.weights_in = weights_in
        self.weights_hn = weights_hn
        self.bias_in = bias_in
        self.bias_hn = bias_hn

        self.recurrent_activation_type = recurrent_activation_type
        self.activation_type = activation_type
        self.return_sequences = return_sequences

    def to_dict(self) -> dict:
        def flatten(matrix: list) -> list:
            return [float(w) for row in matrix for w in row]

        return {
            "type": self.layer_type,
            "seq_len": self.seq_len,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "weights_ir": flatten(self.weights_ir),
            "weights_hr": flatten(self.weights_hr),
            "bias_ir": [float(b) for b in self.bias_ir],
            "bias_hr": [float(b) for b in self.bias_hr],
            "weights_iz": flatten(self.weights_iz),
            "weights_hz": flatten(self.weights_hz),
            "bias_iz": [float(b) for b in self.bias_iz],
            "bias_hz": [float(b) for b in self.bias_hz],
            "weights_in": flatten(self.weights_in),
            "weights_hn": flatten(self.weights_hn),
            "bias_in": [float(b) for b in self.bias_in],
            "bias_hn": [float(b) for b in self.bias_hn],
            "recurrent_activation_type": self.recurrent_activation_type.to_rust_id(),
            "activation_type": self.activation_type.to_rust_id(),
            "return_sequences": self.return_sequences,
        }

    @classmethod
    def gru_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict, weights: dict) -> Self:
        attrs = {a.name: a for a in node.attribute}

        direction = attrs["direction"].s.decode() if "direction" in attrs else "forward"
        if direction != "forward":
            raise ValueError(f"GRU node {node.name} uses direction={direction!r}; only 'forward' " "(single-direction) GRUs are supported")

        # PyTorch's nn.GRU applies the reset gate after the hidden-side linear
        # transform (ONNX's linear_before_reset=1). If a GRU was exported with
        # linear_before_reset=0 the underlying math differs, and this parser's
        # gate wiring would silently produce a layer that doesn't match.
        linear_before_reset = attrs["linear_before_reset"].i if "linear_before_reset" in attrs else 0
        if linear_before_reset != 1:
            raise ValueError(
                f"GRU node {node.name} has linear_before_reset={linear_before_reset}; only " "linear_before_reset=1 (PyTorch's GRU semantics) is supported"
            )

        if "hidden_size" not in attrs:
            raise ValueError(f"GRU node {node.name} is missing the required hidden_size attribute")
        hidden_size = attrs["hidden_size"].i

        # ONNX default activations for GRU are [f, g] = [Sigmoid, Tanh].
        if "activations" in attrs and len(attrs["activations"].strings) >= 2:
            recurrent_activation_name = attrs["activations"].strings[0].decode()
            activation_name = attrs["activations"].strings[1].decode()
        else:
            recurrent_activation_name = "Sigmoid"
            activation_name = "Tanh"
        recurrent_activation_type = ActivationTypes.from_onnx_type(recurrent_activation_name)
        activation_type = ActivationTypes.from_onnx_type(activation_name)

        # X: (seq_length, batch_size, input_size)
        input_tensor_name = node.input[0]
        input_shape = tensor_shapes.get(input_tensor_name)
        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"Could not determine a 3D (seq_length, batch, input_size) input shape " f"for GRU node {node.name}")
        seq_len, _batch, input_size = input_shape

        if len(node.input) < 3 or node.input[1] not in weights or node.input[2] not in weights:
            raise ValueError(f"GRU node {node.name} is missing its W/R weight tensors")

        # W: (3*hidden_size, input_size), R: (3*hidden_size, hidden_size),
        # both concatenated in ONNX's gate order [z, r, h] (update, reset,
        # candidate) - NOT the same order as this struct's field names, which
        # follow PyTorch's r/z/n convention. Slice by ONNX order, then assign
        # to the correspondingly-named fields below.
        w = weights[node.input[1]][0]  # leading num_directions=1 dim dropped
        r = weights[node.input[2]][0]

        w_z, w_r, w_h = w[:hidden_size], w[hidden_size : 2 * hidden_size], w[2 * hidden_size :]
        r_z, r_r, r_h = r[:hidden_size], r[hidden_size : 2 * hidden_size], r[2 * hidden_size :]

        bias_iz = np.zeros(hidden_size, dtype=np.float32)
        bias_ir = np.zeros(hidden_size, dtype=np.float32)
        bias_in = np.zeros(hidden_size, dtype=np.float32)
        bias_hz = np.zeros(hidden_size, dtype=np.float32)
        bias_hr = np.zeros(hidden_size, dtype=np.float32)
        bias_hn = np.zeros(hidden_size, dtype=np.float32)
        if len(node.input) > 3 and node.input[3] and node.input[3] in weights:
            # B: (6*hidden_size,) = [Wbz, Wbr, Wbh, Rbz, Rbr, Rbh]
            b = weights[node.input[3]][0]
            bias_iz = b[:hidden_size]
            bias_ir = b[hidden_size : 2 * hidden_size]
            bias_in = b[2 * hidden_size : 3 * hidden_size]
            bias_hz = b[3 * hidden_size : 4 * hidden_size]
            bias_hr = b[4 * hidden_size : 5 * hidden_size]
            bias_hn = b[5 * hidden_size : 6 * hidden_size]

        # Y is the first output (every timestep's hidden state); Y_h is the
        # second (final hidden state only). ONNX leaves an output name empty
        # ("") when that particular output isn't consumed downstream.
        return_sequences = len(node.output) > 0 and node.output[0] != ""

        return cls(
            seq_len=seq_len,
            input_size=input_size,
            hidden_size=hidden_size,
            weights_ir=w_r.tolist(),
            weights_hr=r_r.tolist(),
            bias_ir=bias_ir.tolist(),
            bias_hr=bias_hr.tolist(),
            weights_iz=w_z.tolist(),
            weights_hz=r_z.tolist(),
            bias_iz=bias_iz.tolist(),
            bias_hz=bias_hz.tolist(),
            weights_in=w_h.tolist(),
            weights_hn=r_h.tolist(),
            bias_in=bias_in.tolist(),
            bias_hn=bias_hn.tolist(),
            recurrent_activation_type=recurrent_activation_type,
            activation_type=activation_type,
            return_sequences=return_sequences,
        )
