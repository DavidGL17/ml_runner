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
        return_sequences: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
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
    def rnn_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict, weights: dict) -> Self:
        attrs = {a.name: a for a in node.attribute}

        direction = attrs["direction"].s.decode() if "direction" in attrs else "forward"
        if direction != "forward":
            raise ValueError(f"RNN node {node.name} uses direction={direction!r}; only 'forward' (single-direction) RNNs are supported")

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
            raise ValueError(f"Could not determine a 3D (seq_length, batch, input_size) input shape for RNN node {node.name}")
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
        return_sequences: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
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
            raise ValueError(f"GRU node {node.name} uses direction={direction!r}; only 'forward' (single-direction) GRUs are supported")

        # PyTorch's nn.GRU applies the reset gate after the hidden-side linear
        # transform (ONNX's linear_before_reset=1). If a GRU was exported with
        # linear_before_reset=0 the underlying math differs, and this parser's
        # gate wiring would silently produce a layer that doesn't match.
        linear_before_reset = attrs["linear_before_reset"].i if "linear_before_reset" in attrs else 0
        if linear_before_reset != 1:
            raise ValueError(
                f"GRU node {node.name} has linear_before_reset={linear_before_reset}; only linear_before_reset=1 (PyTorch's GRU semantics) is supported"
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
            raise ValueError(f"Could not determine a 3D (seq_length, batch, input_size) input shape for GRU node {node.name}")
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


_ONNX_ACTIVATIONS = {"sigmoid": "sigmoid", "tanh": "tanh", "relu": "relu"}


def _split_gates(matrix: np.ndarray, hidden_size: int) -> dict[str, np.ndarray]:
    """ONNX stacks the four gates along the first axis in (i, o, f, c) order.
    Split them into the per-gate fields the Rust `LSTMLayer` uses (input,
    output, forget, cell candidate -> i, o, f, g), like `GRULayerParser`
    does for its gates.
    """
    h = hidden_size
    return {"i": matrix[0:h], "o": matrix[h : 2 * h], "f": matrix[2 * h : 3 * h], "g": matrix[3 * h : 4 * h]}


class LSTMLayerParser(LayerParser):
    def __init__(
        self,
        seq_len: int,
        input_size: int,
        hidden_size: int,
        weights_ih: np.ndarray,
        weights_hh: np.ndarray,
        bias_ih: np.ndarray,
        bias_hh: np.ndarray,
        recurrent_activation_type: str,
        activation_type: str,
        cell_activation_type: str,
        return_sequences: bool,
    ) -> None:
        super().__init__("lstm")
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        # ONNX layout: weights_ih (4*hidden, input), weights_hh (4*hidden, hidden),
        # biases (4*hidden), gates stacked in (i, o, f, c) order.
        self.weights_ih = weights_ih
        self.weights_hh = weights_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.recurrent_activation_type = recurrent_activation_type
        self.activation_type = activation_type
        self.cell_activation_type = cell_activation_type
        self.return_sequences = return_sequences

    def to_dict(self) -> dict:
        w_ih = _split_gates(self.weights_ih, self.hidden_size)
        w_hh = _split_gates(self.weights_hh, self.hidden_size)
        b_ih = _split_gates(self.bias_ih, self.hidden_size)
        b_hh = _split_gates(self.bias_hh, self.hidden_size)

        def flat(a: np.ndarray) -> list[float]:
            return [float(v) for v in a.flatten()]

        out: dict = {
            "type": self.layer_type,
            "seq_len": self.seq_len,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
        }
        # Rust field names: weights_i{i,f,g,o} / weights_h{i,f,g,o} / bias_i{...} / bias_h{...}
        for gate in ("i", "f", "g", "o"):
            out[f"weights_i{gate}"] = flat(w_ih[gate])
            out[f"weights_h{gate}"] = flat(w_hh[gate])
            out[f"bias_i{gate}"] = flat(b_ih[gate])
            out[f"bias_h{gate}"] = flat(b_hh[gate])
        out["recurrent_activation_type"] = self.recurrent_activation_type
        out["activation_type"] = self.activation_type
        out["cell_activation_type"] = self.cell_activation_type
        out["return_sequences"] = self.return_sequences
        return out

    @classmethod
    def lstm_layer_from_onnx(cls, node: NodeProto, tensor_shapes: dict, weights: dict) -> Self:
        name = node.name or "LSTM"
        attrs = {a.name: a for a in node.attribute}

        direction = attrs["direction"].s.decode() if "direction" in attrs else "forward"
        if direction != "forward":
            raise ValueError(f"LSTM node {name} uses direction='{direction}'; only 'forward' is supported")
        if "layout" in attrs and attrs["layout"].i != 0:
            raise ValueError(f"LSTM node {name} uses layout=1 (batch-first); only layout=0 is supported")
        if "input_forget" in attrs and attrs["input_forget"].i != 0:
            raise ValueError(f"LSTM node {name} uses input_forget; not supported")
        if "clip" in attrs:
            raise ValueError(f"LSTM node {name} uses clip; not supported")

        # ONNX order is [f, g, h]: gates, candidate cell state, output-side cell activation.
        onnx_acts = ["sigmoid", "tanh", "tanh"]
        if "activations" in attrs:
            onnx_acts = [a.decode().lower() for a in attrs["activations"].strings]
            if len(onnx_acts) != 3:
                raise ValueError(f"LSTM node {name} has activations {onnx_acts}; expected exactly 3 for a forward LSTM")
        unsupported = [a for a in onnx_acts if a not in _ONNX_ACTIVATIONS]
        if unsupported:
            raise ValueError(f"LSTM node {name} uses unsupported activation(s) {unsupported}; supported: {sorted(_ONNX_ACTIVATIONS)}")
        recurrent_act, candidate_act, cell_act = (_ONNX_ACTIVATIONS[a] for a in onnx_acts)

        # Positional inputs: X, W, R, B, sequence_lens, initial_h, initial_c, P ("" = omitted)
        inputs = list(node.input) + [""] * (8 - len(node.input))
        x_name, w_name, r_name, b_name, _seq_lens, h0_name, c0_name, p_name = inputs[:8]

        if p_name:
            raise ValueError(f"LSTM node {name} uses peephole weights (P); not supported")
        if w_name not in weights or r_name not in weights:
            raise ValueError(f"LSTM node {name} needs constant W and R weight tensors")

        # The Rust layer always starts from zero hidden/cell state.
        for label, state_name in (("initial_h", h0_name), ("initial_c", c0_name)):
            if not state_name:
                continue
            if state_name not in weights:
                raise ValueError(f"LSTM node {name} has a non-constant {label} '{state_name}'; only zero initial states are supported")
            if np.any(weights[state_name] != 0):
                raise ValueError(f"LSTM node {name} has a non-zero {label}; only zero initial states are supported")

        w = weights[w_name]  # (num_directions, 4*hidden, input)
        r = weights[r_name]  # (num_directions, 4*hidden, hidden)
        if w.shape[0] != 1:
            raise ValueError(f"LSTM node {name} has num_directions={w.shape[0]}; only unidirectional is supported")
        input_size = w.shape[2]
        hidden_size = r.shape[2]

        if b_name:
            if b_name not in weights:
                raise ValueError(f"LSTM node {name} has a non-constant bias B")
            b = weights[b_name][0]  # (8*hidden): Wb[iofc] then Rb[iofc]
            bias_ih, bias_hh = b[: 4 * hidden_size], b[4 * hidden_size :]
        else:
            bias_ih = bias_hh = np.zeros(4 * hidden_size, dtype=np.float32)

        x_shape = tensor_shapes.get(x_name)
        if x_shape is None or len(x_shape) != 3:
            raise ValueError(f"Could not determine a 3D (seq_len, batch, features) input shape for LSTM node {name}")
        seq_len = x_shape[0]

        # Outputs are Y (all steps), Y_h (final hidden), Y_c (final cell). Exactly
        # one may be wired downstream (enforced by _build_node).
        outputs = list(node.output) + [""] * (3 - len(node.output))
        y, y_h, _y_c = outputs[:3]
        if y:
            return_sequences = True
        elif y_h:
            return_sequences = False
        else:
            raise ValueError(f"LSTM node {name} only exposes Y_c; expose Y or Y_h instead")

        return cls(
            seq_len,
            input_size,
            hidden_size,
            w[0],
            r[0],
            bias_ih,
            bias_hh,
            recurrent_act,
            candidate_act,
            cell_act,
            return_sequences,
        )
