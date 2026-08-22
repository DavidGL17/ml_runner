//! Vanilla (Elman) RNN and GRU layers.
//!
//! Both process a fixed-length sequence timestep by timestep, starting
//! from a zero hidden state, and - depending on `return_sequences` -
//! output either every timestep's hidden state or just the final one.
//!
//! `TensorShape` is fully static - `Model::validate_shapes` relies on that
//! to check a layer chain without running any data through it - so unlike
//! most RNN implementations, the sequence length here is a fixed part of
//! each layer's config (`seq_len`), not something inferred per call.

use crate::activation::ActivationType;
use crate::tensor::{Tensor, TensorShape};
use ndarray::{Array1, ArrayView1, ArrayView2, Ix1, Ix2};
use serde::{Deserialize, Serialize};

/// Applies an `ActivationType` to a 1-D array, going through `ArrayD`
/// (the type `ActivationType::apply_array` operates on) and back. Shared
/// by `RNNLayer` and `GRULayer`'s per-timestep computations.
fn activate(activation: &ActivationType, z: Array1<f32>) -> Array1<f32> {
    let mut z_dyn = z.into_dyn();
    activation.apply_array(&mut z_dyn);
    z_dyn
        .into_dimensionality::<Ix1>()
        .expect("activation output is not 1-D")
}

fn default_recurrent_activation() -> ActivationType {
    ActivationType::Sigmoid
}

fn default_candidate_activation() -> ActivationType {
    ActivationType::Tanh
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RNNLayer {
    pub seq_len: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    /// Input-to-hidden weights, flattened (hidden_size x input_size), row-major.
    pub weights_ih: Vec<f32>,
    /// Hidden-to-hidden weights, flattened (hidden_size x hidden_size), row-major.
    pub weights_hh: Vec<f32>,
    pub bias_ih: Vec<f32>,
    pub bias_hh: Vec<f32>,
    pub activation_type: ActivationType,
    /// If true, the output is every timestep's hidden state (a `seq_len x hidden_size` D2 tensor). 
    /// If false (the default), the output is just the final hidden state (a `hidden_size` Flat tensor)
    #[serde(default)]
    pub return_sequences: bool,
}

impl RNNLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::D2 {
            dim1: self.seq_len,
            dim2: self.input_size,
        }
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        if self.return_sequences {
            TensorShape::D2 {
                dim1: self.seq_len,
                dim2: self.hidden_size,
            }
        } else {
            TensorShape::Flat(self.hidden_size)
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in RNNLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let weights_ih =
            ArrayView2::from_shape((self.hidden_size, self.input_size), &self.weights_ih)
                .expect("RNNLayer weights_ih length doesn't match hidden_size * input_size");
        let weights_hh =
            ArrayView2::from_shape((self.hidden_size, self.hidden_size), &self.weights_hh)
                .expect("RNNLayer weights_hh length doesn't match hidden_size * hidden_size");
        let bias_ih = ArrayView1::from(&self.bias_ih);
        let bias_hh = ArrayView1::from(&self.bias_hh);

        let input_seq: ArrayView2<f32> = input
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("RNNLayer input is not 2-D");

        let mut hidden = Array1::<f32>::zeros(self.hidden_size);
        let mut outputs: Vec<f32> = if self.return_sequences {
            Vec::with_capacity(self.seq_len * self.hidden_size)
        } else {
            Vec::new()
        };

        for t in 0..self.seq_len {
            let x_t = input_seq.row(t);

            let mut z = weights_ih.dot(&x_t) + &bias_ih;
            z = z + weights_hh.dot(&hidden) + &bias_hh;

            hidden = activate(&self.activation_type, z);

            if self.return_sequences {
                outputs.extend(hidden.iter());
            }
        }

        if self.return_sequences {
            Tensor::new(outputs, self.output_shape())
        } else {
            Tensor::from_array(hidden.into_dyn())
        }
    }
}

/// A GRU (Gated Recurrent Unit) layer.
///
/// At each timestep t:
/// ```text
/// r_t = recurrent_activation(W_ir . x_t + b_ir + W_hr . h_{t-1} + b_hr)   // reset gate
/// z_t = recurrent_activation(W_iz . x_t + b_iz + W_hz . h_{t-1} + b_hz)   // update gate
/// n_t = activation(W_in . x_t + b_in + r_t * (W_hn . h_{t-1} + b_hn))    // candidate state
/// h_t = (1 - z_t) * n_t + z_t * h_{t-1}
/// ```
/// with `h_0` the zero vector and `*` elementwise multiplication.
/// `recurrent_activation` defaults to sigmoid and `activation` to tanh -
/// the conventional GRU - but both are configurable per layer.
///
/// Unlike `RNNLayer`'s combined `weights_ih`/`weights_hh`, each gate gets
/// its own explicit weight matrices and biases rather than one matrix
/// stacked across gates: there's no fixed external weight-import format
/// to match here, and explicit per-gate fields are far easier to hand-write
/// correct JSON for than guessing a stacking convention.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GRULayer {
    pub seq_len: usize,
    pub input_size: usize,
    pub hidden_size: usize,

    /// Reset gate: input-to-hidden weights, flattened (hidden_size x input_size), row-major.
    pub weights_ir: Vec<f32>,
    /// Reset gate: hidden-to-hidden weights, flattened (hidden_size x hidden_size), row-major.
    pub weights_hr: Vec<f32>,
    pub bias_ir: Vec<f32>,
    pub bias_hr: Vec<f32>,

    /// Update gate: input-to-hidden weights, flattened (hidden_size x input_size), row-major.
    pub weights_iz: Vec<f32>,
    /// Update gate: hidden-to-hidden weights, flattened (hidden_size x hidden_size), row-major.
    pub weights_hz: Vec<f32>,
    pub bias_iz: Vec<f32>,
    pub bias_hz: Vec<f32>,

    /// Candidate state: input-to-hidden weights, flattened (hidden_size x input_size), row-major.
    pub weights_in: Vec<f32>,
    /// Candidate state: hidden-to-hidden weights, flattened (hidden_size x hidden_size), row-major.
    pub weights_hn: Vec<f32>,
    pub bias_in: Vec<f32>,
    pub bias_hn: Vec<f32>,

    /// Activation for the reset and update gates. Defaults to sigmoid.
    #[serde(default = "default_recurrent_activation")]
    pub recurrent_activation_type: ActivationType,
    /// Activation for the candidate hidden state. Defaults to tanh.
    #[serde(default = "default_candidate_activation")]
    pub activation_type: ActivationType,

    /// If true, the output is every timestep's hidden state (a `seq_len x hidden_size` D2 tensor). 
    /// If false (the default), the output is just the final hidden state (a `hidden_size` Flat tensor).
    #[serde(default)]
    pub return_sequences: bool,
}

impl GRULayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::D2 {
            dim1: self.seq_len,
            dim2: self.input_size,
        }
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        if self.return_sequences {
            TensorShape::D2 {
                dim1: self.seq_len,
                dim2: self.hidden_size,
            }
        } else {
            TensorShape::Flat(self.hidden_size)
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in GRULayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let weights_ir =
            ArrayView2::from_shape((self.hidden_size, self.input_size), &self.weights_ir)
                .expect("GRULayer weights_ir length doesn't match hidden_size * input_size");
        let weights_hr =
            ArrayView2::from_shape((self.hidden_size, self.hidden_size), &self.weights_hr)
                .expect("GRULayer weights_hr length doesn't match hidden_size * hidden_size");
        let weights_iz =
            ArrayView2::from_shape((self.hidden_size, self.input_size), &self.weights_iz)
                .expect("GRULayer weights_iz length doesn't match hidden_size * input_size");
        let weights_hz =
            ArrayView2::from_shape((self.hidden_size, self.hidden_size), &self.weights_hz)
                .expect("GRULayer weights_hz length doesn't match hidden_size * hidden_size");
        let weights_in =
            ArrayView2::from_shape((self.hidden_size, self.input_size), &self.weights_in)
                .expect("GRULayer weights_in length doesn't match hidden_size * input_size");
        let weights_hn =
            ArrayView2::from_shape((self.hidden_size, self.hidden_size), &self.weights_hn)
                .expect("GRULayer weights_hn length doesn't match hidden_size * hidden_size");

        let bias_ir = ArrayView1::from(&self.bias_ir);
        let bias_hr = ArrayView1::from(&self.bias_hr);
        let bias_iz = ArrayView1::from(&self.bias_iz);
        let bias_hz = ArrayView1::from(&self.bias_hz);
        let bias_in = ArrayView1::from(&self.bias_in);
        let bias_hn = ArrayView1::from(&self.bias_hn);

        let input_seq: ArrayView2<f32> = input
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("GRULayer input is not 2-D");

        let mut hidden = Array1::<f32>::zeros(self.hidden_size);
        let mut outputs: Vec<f32> = if self.return_sequences {
            Vec::with_capacity(self.seq_len * self.hidden_size)
        } else {
            Vec::new()
        };

        for t in 0..self.seq_len {
            let x_t = input_seq.row(t);

            let mut r_pre = weights_ir.dot(&x_t) + &bias_ir;
            r_pre = r_pre + weights_hr.dot(&hidden) + &bias_hr;
            let r = activate(&self.recurrent_activation_type, r_pre);

            let mut z_pre = weights_iz.dot(&x_t) + &bias_iz;
            z_pre = z_pre + weights_hz.dot(&hidden) + &bias_hz;
            let z = activate(&self.recurrent_activation_type, z_pre);

            let hn_term = weights_hn.dot(&hidden) + &bias_hn;
            let mut n_pre = weights_in.dot(&x_t) + &bias_in;
            n_pre = n_pre + &r * &hn_term;
            let n = activate(&self.activation_type, n_pre);

            let one_minus_z = z.mapv(|v| 1.0 - v);
            hidden = &one_minus_z * &n + &z * &hidden;

            if self.return_sequences {
                outputs.extend(hidden.iter());
            }
        }

        if self.return_sequences {
            Tensor::new(outputs, self.output_shape())
        } else {
            Tensor::from_array(hidden.into_dyn())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::ActivationType;
    use approx::assert_abs_diff_eq;

    /// seq_len = 1 means the hidden-to-hidden term is multiplied by the
    /// zero initial state, so this reduces to a single Dense-like step:
    /// h_1 = 1*1 + 1*2 + 0 = 3.
    #[test]
    fn test_single_step_matches_dense_like_computation() {
        let layer = RNNLayer {
            seq_len: 1,
            input_size: 2,
            hidden_size: 1,
            weights_ih: vec![1.0, 1.0],
            weights_hh: vec![0.0],
            bias_ih: vec![0.0],
            bias_hh: vec![0.0],
            activation_type: ActivationType::Linear,
            return_sequences: false,
        };
        let input = Tensor::new(vec![1.0, 2.0], TensorShape::D2 { dim1: 1, dim2: 2 });
        let output = layer.forward(&input);

        assert_eq!(output.shape(), TensorShape::Flat(1));
        assert_eq!(output.to_vec(), vec![3.0]);
    }

    /// seq_len = 2, scalar input/hidden, all weights = 1, all biases = 0:
    /// h_0 = 0
    /// h_1 = 1*x_0 + 1*h_0 = 1*1 + 1*0 = 1
    /// h_2 = 1*x_1 + 1*h_1 = 1*2 + 1*1 = 3
    #[test]
    fn test_hidden_state_carries_over_return_sequences() {
        let layer = RNNLayer {
            seq_len: 2,
            input_size: 1,
            hidden_size: 1,
            weights_ih: vec![1.0],
            weights_hh: vec![1.0],
            bias_ih: vec![0.0],
            bias_hh: vec![0.0],
            activation_type: ActivationType::Linear,
            return_sequences: true,
        };
        let input = Tensor::new(vec![1.0, 2.0], TensorShape::D2 { dim1: 2, dim2: 1 });
        let output = layer.forward(&input);

        assert_eq!(output.shape(), TensorShape::D2 { dim1: 2, dim2: 1 });
        assert_eq!(output.to_vec(), vec![1.0, 3.0]);
    }

    /// Same layer as above but return_sequences = false: only the final
    /// hidden state (3.0) should come back, as a Flat(1) tensor.
    #[test]
    fn test_hidden_state_carries_over_final_only() {
        let layer = RNNLayer {
            seq_len: 2,
            input_size: 1,
            hidden_size: 1,
            weights_ih: vec![1.0],
            weights_hh: vec![1.0],
            bias_ih: vec![0.0],
            bias_hh: vec![0.0],
            activation_type: ActivationType::Linear,
            return_sequences: false,
        };
        let input = Tensor::new(vec![1.0, 2.0], TensorShape::D2 { dim1: 2, dim2: 1 });
        let output = layer.forward(&input);

        assert_eq!(output.shape(), TensorShape::Flat(1));
        assert_eq!(output.to_vec(), vec![3.0]);
    }

    /// Exercises a non-linear activation and a bias-only layer (zero
    /// weights) so the expected value is just activation(bias) at every
    /// step, independent of the input or hidden state.
    #[test]
    fn test_tanh_activation() {
        let layer = RNNLayer {
            seq_len: 1,
            input_size: 1,
            hidden_size: 1,
            weights_ih: vec![0.0],
            weights_hh: vec![0.0],
            bias_ih: vec![0.5],
            bias_hh: vec![0.0],
            activation_type: ActivationType::Tanh,
            return_sequences: false,
        };
        let input = Tensor::new(vec![100.0], TensorShape::D2 { dim1: 1, dim2: 1 });
        let output = layer.forward(&input);

        assert_eq!(output.to_vec(), vec![0.5f32.tanh()]);
    }

    #[test]
    fn test_multi_dimensional_hidden_and_input() {
        // input_size = 2, hidden_size = 2, seq_len = 1.
        // weights_ih row-major (hidden_size x input_size):
        //   [1, 0]
        //   [0, 1]
        // so with zero bias and zero initial hidden state, h_1 = x_0.
        let layer = RNNLayer {
            seq_len: 1,
            input_size: 2,
            hidden_size: 2,
            weights_ih: vec![1.0, 0.0, 0.0, 1.0],
            weights_hh: vec![0.0, 0.0, 0.0, 0.0],
            bias_ih: vec![0.0, 0.0],
            bias_hh: vec![0.0, 0.0],
            activation_type: ActivationType::Linear,
            return_sequences: false,
        };
        let input = Tensor::new(vec![3.0, 4.0], TensorShape::D2 { dim1: 1, dim2: 2 });
        let output = layer.forward(&input);

        assert_eq!(output.to_vec(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_input_output_shape_return_sequences() {
        let layer = RNNLayer {
            seq_len: 5,
            input_size: 3,
            hidden_size: 4,
            weights_ih: vec![0.0; 4 * 3],
            weights_hh: vec![0.0; 4 * 4],
            bias_ih: vec![0.0; 4],
            bias_hh: vec![0.0; 4],
            activation_type: ActivationType::Linear,
            return_sequences: true,
        };

        assert_eq!(layer.input_shape(), TensorShape::D2 { dim1: 5, dim2: 3 });
        assert_eq!(layer.output_shape(), TensorShape::D2 { dim1: 5, dim2: 4 });
    }

    #[test]
    fn test_input_output_shape_final_only() {
        let layer = RNNLayer {
            seq_len: 5,
            input_size: 3,
            hidden_size: 4,
            weights_ih: vec![0.0; 4 * 3],
            weights_hh: vec![0.0; 4 * 4],
            bias_ih: vec![0.0; 4],
            bias_hh: vec![0.0; 4],
            activation_type: ActivationType::Linear,
            return_sequences: false,
        };

        assert_eq!(layer.input_shape(), TensorShape::D2 { dim1: 5, dim2: 3 });
        assert_eq!(layer.output_shape(), TensorShape::Flat(4));
    }

    #[test]
    #[should_panic(expected = "Shape mismatch in RNNLayer")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = RNNLayer {
            seq_len: 2,
            input_size: 3,
            hidden_size: 1,
            weights_ih: vec![0.0; 3],
            weights_hh: vec![0.0],
            bias_ih: vec![0.0],
            bias_hh: vec![0.0],
            activation_type: ActivationType::Linear,
            return_sequences: false,
        };
        let wrong_input = Tensor::new(vec![0.0; 4], TensorShape::Flat(4));
        let _ = layer.forward(&wrong_input);
    }

    /// seq_len = 1, all weights/biases zero except weights_in = [1.0], so:
    /// r = sigmoid(0) = 0.5, z = sigmoid(0) = 0.5, hn_term = 0
    /// n = tanh(1*2.0 + 0.5*0) = tanh(2.0)
    /// h_1 = (1 - 0.5)*tanh(2.0) + 0.5*0 = 0.5*tanh(2.0)
    #[test]
    fn test_gru_single_step_matches_manual_computation() {
        let layer = GRULayer {
            seq_len: 1,
            input_size: 1,
            hidden_size: 1,
            weights_ir: vec![0.0],
            weights_hr: vec![0.0],
            bias_ir: vec![0.0],
            bias_hr: vec![0.0],
            weights_iz: vec![0.0],
            weights_hz: vec![0.0],
            bias_iz: vec![0.0],
            bias_hz: vec![0.0],
            weights_in: vec![1.0],
            weights_hn: vec![0.0],
            bias_in: vec![0.0],
            bias_hn: vec![0.0],
            recurrent_activation_type: ActivationType::Sigmoid,
            activation_type: ActivationType::Tanh,
            return_sequences: false,
        };
        let input = Tensor::new(vec![2.0], TensorShape::D2 { dim1: 1, dim2: 1 });
        let output = layer.forward(&input);

        assert_eq!(output.shape(), TensorShape::Flat(1));
        assert_eq!(output.to_vec(), vec![0.5 * 2.0f32.tanh()]);
    }

    /// seq_len = 2, r and z gates held at a constant 0.5 (all their
    /// weights/biases are zero), weights_in = weights_hn = [1.0]:
    /// h_0 = 0
    /// n_0 = tanh(1*x_0 + 0.5*(1*h_0)) = tanh(1.0)
    /// h_1 = 0.5*n_0 + 0.5*h_0 = 0.5*tanh(1.0)
    /// n_1 = tanh(1*x_1 + 0.5*(1*h_1)) = tanh(2 + 0.5*h_1)
    /// h_2 = 0.5*n_1 + 0.5*h_1
    #[test]
    fn test_gru_hidden_state_carries_over() {
        let layer = GRULayer {
            seq_len: 2,
            input_size: 1,
            hidden_size: 1,
            weights_ir: vec![0.0],
            weights_hr: vec![0.0],
            bias_ir: vec![0.0],
            bias_hr: vec![0.0],
            weights_iz: vec![0.0],
            weights_hz: vec![0.0],
            bias_iz: vec![0.0],
            bias_hz: vec![0.0],
            weights_in: vec![1.0],
            weights_hn: vec![1.0],
            bias_in: vec![0.0],
            bias_hn: vec![0.0],
            recurrent_activation_type: ActivationType::Sigmoid,
            activation_type: ActivationType::Tanh,
            return_sequences: true,
        };
        let input = Tensor::new(vec![1.0, 2.0], TensorShape::D2 { dim1: 2, dim2: 1 });
        let output = layer.forward(&input);

        let h1 = 0.5 * 1.0f32.tanh();
        let n1 = (2.0 + 0.5 * h1).tanh();
        let h2 = 0.5 * n1 + 0.5 * h1;

        assert_eq!(output.shape(), TensorShape::D2 { dim1: 2, dim2: 1 });
        let got = output.to_vec();
        assert_abs_diff_eq!(got[0], h1, epsilon = 1e-6);
        assert_abs_diff_eq!(got[1], h2, epsilon = 1e-6);
    }

    #[test]
    fn test_gru_recurrent_and_candidate_activations_default() {
        let json = r#"
        {
            "seq_len": 1,
            "input_size": 1,
            "hidden_size": 1,
            "weights_ir": [0.0], "weights_hr": [0.0], "bias_ir": [0.0], "bias_hr": [0.0],
            "weights_iz": [0.0], "weights_hz": [0.0], "bias_iz": [0.0], "bias_hz": [0.0],
            "weights_in": [1.0], "weights_hn": [0.0], "bias_in": [0.0], "bias_hn": [0.0]
        }
        "#;
        let layer: GRULayer = serde_json::from_str(json).unwrap();

        assert_eq!(layer.recurrent_activation_type, ActivationType::Sigmoid);
        assert_eq!(layer.activation_type, ActivationType::Tanh);
        assert!(!layer.return_sequences);
    }

    #[test]
    fn test_gru_input_output_shape_return_sequences() {
        let layer = GRULayer {
            seq_len: 5,
            input_size: 3,
            hidden_size: 4,
            weights_ir: vec![0.0; 4 * 3],
            weights_hr: vec![0.0; 4 * 4],
            bias_ir: vec![0.0; 4],
            bias_hr: vec![0.0; 4],
            weights_iz: vec![0.0; 4 * 3],
            weights_hz: vec![0.0; 4 * 4],
            bias_iz: vec![0.0; 4],
            bias_hz: vec![0.0; 4],
            weights_in: vec![0.0; 4 * 3],
            weights_hn: vec![0.0; 4 * 4],
            bias_in: vec![0.0; 4],
            bias_hn: vec![0.0; 4],
            recurrent_activation_type: ActivationType::Sigmoid,
            activation_type: ActivationType::Tanh,
            return_sequences: true,
        };

        assert_eq!(layer.input_shape(), TensorShape::D2 { dim1: 5, dim2: 3 });
        assert_eq!(layer.output_shape(), TensorShape::D2 { dim1: 5, dim2: 4 });
    }

    #[test]
    fn test_gru_input_output_shape_final_only() {
        let layer = GRULayer {
            seq_len: 5,
            input_size: 3,
            hidden_size: 4,
            weights_ir: vec![0.0; 4 * 3],
            weights_hr: vec![0.0; 4 * 4],
            bias_ir: vec![0.0; 4],
            bias_hr: vec![0.0; 4],
            weights_iz: vec![0.0; 4 * 3],
            weights_hz: vec![0.0; 4 * 4],
            bias_iz: vec![0.0; 4],
            bias_hz: vec![0.0; 4],
            weights_in: vec![0.0; 4 * 3],
            weights_hn: vec![0.0; 4 * 4],
            bias_in: vec![0.0; 4],
            bias_hn: vec![0.0; 4],
            recurrent_activation_type: ActivationType::Sigmoid,
            activation_type: ActivationType::Tanh,
            return_sequences: false,
        };

        assert_eq!(layer.input_shape(), TensorShape::D2 { dim1: 5, dim2: 3 });
        assert_eq!(layer.output_shape(), TensorShape::Flat(4));
    }

    #[test]
    #[should_panic(expected = "Shape mismatch in GRULayer")]
    fn test_gru_forward_rejects_wrong_input_shape() {
        let layer = GRULayer {
            seq_len: 2,
            input_size: 3,
            hidden_size: 1,
            weights_ir: vec![0.0; 3],
            weights_hr: vec![0.0],
            bias_ir: vec![0.0],
            bias_hr: vec![0.0],
            weights_iz: vec![0.0; 3],
            weights_hz: vec![0.0],
            bias_iz: vec![0.0],
            bias_hz: vec![0.0],
            weights_in: vec![0.0; 3],
            weights_hn: vec![0.0],
            bias_in: vec![0.0],
            bias_hn: vec![0.0],
            recurrent_activation_type: ActivationType::Sigmoid,
            activation_type: ActivationType::Tanh,
            return_sequences: false,
        };
        let wrong_input = Tensor::new(vec![0.0; 4], TensorShape::Flat(4));
        let _ = layer.forward(&wrong_input);
    }
}