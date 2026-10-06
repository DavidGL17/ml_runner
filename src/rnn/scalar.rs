//! Default forward-pass implementations for `RNNLayer`, `GRULayer` and
//! `LSTMLayer`, via `ndarray`'s `.dot()` (portable `matrixmultiply`, or a
//! linked system BLAS with the `blas` feature). Weights/biases stay as flat
//! `Vec<f32>` fields (so the JSON model format is unaffected) and are
//! borrowed here as `ndarray` views - no copying. Compiled in whenever the
//! `simd` feature is *not* enabled - see `rnn/simd.rs` for the alternative.
//! Input and weight validation is shared and lives in each layer's
//! `validate` (`rnn/layer.rs`).

use super::{GRULayer, LSTMLayer, RNNLayer};
use crate::activation::ActivationType;
use crate::tensor::Tensor;
use ndarray::{Array1, ArrayView1, ArrayView2, Ix1, Ix2};

/// Applies an `ActivationType` to a 1-D array, going through `ArrayD`
/// (the type `ActivationType::apply_array` operates on) and back. Shared
/// by all three layers' per-timestep computations.
fn activate(activation: &ActivationType, z: Array1<f32>) -> Array1<f32> {
    let mut z_dyn = z.into_dyn();
    activation.apply_array(&mut z_dyn);
    z_dyn
        .into_dimensionality::<Ix1>()
        .expect("activation output is not 1-D")
}

impl RNNLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.validate(input);

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

            let mut z = weights_ih.dot(&x_t) + bias_ih;
            z = z + weights_hh.dot(&hidden) + bias_hh;

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

impl GRULayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.validate(input);

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

            let mut r_pre = weights_ir.dot(&x_t) + bias_ir;
            r_pre = r_pre + weights_hr.dot(&hidden) + bias_hr;
            let r = activate(&self.recurrent_activation_type, r_pre);

            let mut z_pre = weights_iz.dot(&x_t) + bias_iz;
            z_pre = z_pre + weights_hz.dot(&hidden) + bias_hz;
            let z = activate(&self.recurrent_activation_type, z_pre);

            let hn_term = weights_hn.dot(&hidden) + bias_hn;
            let mut n_pre = weights_in.dot(&x_t) + bias_in;
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

impl LSTMLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.validate(input);

        let weights_ii =
            ArrayView2::from_shape((self.hidden_size, self.input_size), &self.weights_ii)
                .expect("LSTMLayer weights_ii length doesn't match hidden_size * input_size");
        let weights_hi =
            ArrayView2::from_shape((self.hidden_size, self.hidden_size), &self.weights_hi)
                .expect("LSTMLayer weights_hi length doesn't match hidden_size * hidden_size");
        let weights_if =
            ArrayView2::from_shape((self.hidden_size, self.input_size), &self.weights_if)
                .expect("LSTMLayer weights_if length doesn't match hidden_size * input_size");
        let weights_hf =
            ArrayView2::from_shape((self.hidden_size, self.hidden_size), &self.weights_hf)
                .expect("LSTMLayer weights_hf length doesn't match hidden_size * hidden_size");
        let weights_ig =
            ArrayView2::from_shape((self.hidden_size, self.input_size), &self.weights_ig)
                .expect("LSTMLayer weights_ig length doesn't match hidden_size * input_size");
        let weights_hg =
            ArrayView2::from_shape((self.hidden_size, self.hidden_size), &self.weights_hg)
                .expect("LSTMLayer weights_hg length doesn't match hidden_size * hidden_size");
        let weights_io =
            ArrayView2::from_shape((self.hidden_size, self.input_size), &self.weights_io)
                .expect("LSTMLayer weights_io length doesn't match hidden_size * input_size");
        let weights_ho =
            ArrayView2::from_shape((self.hidden_size, self.hidden_size), &self.weights_ho)
                .expect("LSTMLayer weights_ho length doesn't match hidden_size * hidden_size");

        let bias_ii = ArrayView1::from(&self.bias_ii);
        let bias_hi = ArrayView1::from(&self.bias_hi);
        let bias_if = ArrayView1::from(&self.bias_if);
        let bias_hf = ArrayView1::from(&self.bias_hf);
        let bias_ig = ArrayView1::from(&self.bias_ig);
        let bias_hg = ArrayView1::from(&self.bias_hg);
        let bias_io = ArrayView1::from(&self.bias_io);
        let bias_ho = ArrayView1::from(&self.bias_ho);

        let input_seq: ArrayView2<f32> = input
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("LSTMLayer input is not 2-D");

        let mut hidden = Array1::<f32>::zeros(self.hidden_size);
        let mut cell = Array1::<f32>::zeros(self.hidden_size);
        let mut outputs: Vec<f32> = if self.return_sequences {
            Vec::with_capacity(self.seq_len * self.hidden_size)
        } else {
            Vec::new()
        };

        for t in 0..self.seq_len {
            let x_t = input_seq.row(t);

            let mut i_pre = weights_ii.dot(&x_t) + bias_ii;
            i_pre = i_pre + weights_hi.dot(&hidden) + bias_hi;
            let i = activate(&self.recurrent_activation_type, i_pre);

            let mut f_pre = weights_if.dot(&x_t) + bias_if;
            f_pre = f_pre + weights_hf.dot(&hidden) + bias_hf;
            let f = activate(&self.recurrent_activation_type, f_pre);

            let mut g_pre = weights_ig.dot(&x_t) + bias_ig;
            g_pre = g_pre + weights_hg.dot(&hidden) + bias_hg;
            let g = activate(&self.activation_type, g_pre);

            let mut o_pre = weights_io.dot(&x_t) + bias_io;
            o_pre = o_pre + weights_ho.dot(&hidden) + bias_ho;
            let o = activate(&self.recurrent_activation_type, o_pre);

            cell = &f * &cell + &i * &g;
            hidden = &o * &activate(&self.cell_activation_type, cell.clone());

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