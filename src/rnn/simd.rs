//! SIMD-accelerated forward-pass implementations for `RNNLayer`,
//! `GRULayer` and `LSTMLayer`, via the `wide` crate (portable SIMD chosen at
//! compile time from the build's target features, no runtime CPU
//! detection). Only compiled in when the `simd` Cargo feature is enabled.
//! Input and weight validation is shared and lives in each layer's
//! `validate` (`rnn/layer.rs`).
//!
//! Like `dense/simd.rs`, this bypasses `ndarray`'s `.dot()`/BLAS, so nothing
//! needs linking. Each timestep is dominated by matrix-vector products
//! (`hidden_size x (input_size + hidden_size)` per gate), so those are the
//! SIMD part: every output row is an 8-lane dot product with a scalar tail,
//! and the input and recurrent halves of a gate are fused into one pass
//! that writes the pre-activation directly.
//!
//! Two deliberate choices:
//!   - Activations go through the project's own `ActivationType::apply_array`
//!     (via a reusable scratch buffer), exactly as the scalar backend does,
//!     so tanh/sigmoid/etc. are numerically identical across backends. They
//!     cost O(hidden_size) per gate against O(hidden_size * (input + hidden))
//!     for the matvec.
//!   - The cheap elementwise state updates (`n + r * hn`, `(1 - z) * n + z *
//!     h`, `f * c + i * g`, ...) are plain loops over slices, which the
//!     compiler vectorises on its own.
//!
//! Buffers are allocated once per `forward` call and reused every timestep.

use super::{GRULayer, LSTMLayer, RNNLayer};
use crate::activation::ActivationType;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use wide::f32x8;

const LANES: usize = 8;

/// 8-lane dot product with a scalar tail (same scheme as `dense/simd.rs`).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let chunks = a.len() / LANES;
    let mut acc = f32x8::ZERO;
    for c in 0..chunks {
        let base = c * LANES;
        let x = f32x8::from(<[f32; LANES]>::try_from(&a[base..base + LANES]).unwrap());
        let y = f32x8::from(<[f32; LANES]>::try_from(&b[base..base + LANES]).unwrap());
        acc += x * y;
    }
    let mut sum = acc.reduce_add();
    for j in chunks * LANES..a.len() {
        sum += a[j] * b[j];
    }
    sum
}

/// `out[i] = W[i] . v + b[i]`, with `W` row-major `(out.len() x v.len())`.
fn affine(w: &[f32], b: &[f32], v: &[f32], out: &mut [f32]) {
    let cols = v.len();
    debug_assert_eq!(w.len(), out.len() * cols);
    debug_assert_eq!(b.len(), out.len());
    for (i, o) in out.iter_mut().enumerate() {
        *o = dot(&w[i * cols..(i + 1) * cols], v) + b[i];
    }
}

/// A gate pre-activation, fused:
/// `out[i] = ((Wx[i] . x + bx[i]) + Wh[i] . h) + bh[i]`.
/// `Wx` is `(out.len() x x.len())` and `Wh` is `(out.len() x h.len())`,
/// both row-major.
fn gate_preact(
    wx: &[f32],
    wh: &[f32],
    bx: &[f32],
    bh: &[f32],
    x: &[f32],
    h: &[f32],
    out: &mut [f32],
) {
    let (in_n, hid) = (x.len(), h.len());
    debug_assert_eq!(wx.len(), out.len() * in_n);
    debug_assert_eq!(wh.len(), out.len() * hid);
    debug_assert_eq!(bx.len(), out.len());
    debug_assert_eq!(bh.len(), out.len());
    for (i, o) in out.iter_mut().enumerate() {
        let zx = dot(&wx[i * in_n..(i + 1) * in_n], x) + bx[i];
        *o = (zx + dot(&wh[i * hid..(i + 1) * hid], h)) + bh[i];
    }
}

/// Applies an `ActivationType` to a slice in place by round-tripping
/// through one reusable `ArrayD` (the type `apply_array` operates on).
struct Activator {
    scratch: ArrayD<f32>,
}

impl Activator {
    fn new(len: usize) -> Self {
        Self {
            scratch: ArrayD::zeros(IxDyn(&[len])),
        }
    }

    fn apply(&mut self, activation: &ActivationType, buf: &mut [f32]) {
        self.scratch
            .as_slice_mut()
            .expect("activation scratch is contiguous")
            .copy_from_slice(buf);
        activation.apply_array(&mut self.scratch);
        buf.copy_from_slice(
            self.scratch
                .as_slice()
                .expect("activation scratch is contiguous"),
        );
    }
}

fn contiguous(t: &Tensor) -> &[f32] {
    t.data
        .as_slice()
        .expect("recurrent layer input must be a contiguous tensor")
}

impl RNNLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.validate(input);

        let (n_in, n_h) = (self.input_size, self.hidden_size);
        let x = contiguous(input);

        let mut hidden = vec![0.0f32; n_h];
        let mut z = vec![0.0f32; n_h];
        let mut act = Activator::new(n_h);
        let mut outputs: Vec<f32> = if self.return_sequences {
            Vec::with_capacity(self.seq_len * n_h)
        } else {
            Vec::new()
        };

        for t in 0..self.seq_len {
            let x_t = &x[t * n_in..(t + 1) * n_in];

            gate_preact(
                &self.weights_ih,
                &self.weights_hh,
                &self.bias_ih,
                &self.bias_hh,
                x_t,
                &hidden,
                &mut z,
            );
            act.apply(&self.activation_type, &mut z);
            std::mem::swap(&mut hidden, &mut z);

            if self.return_sequences {
                outputs.extend_from_slice(&hidden);
            }
        }

        if self.return_sequences {
            Tensor::new(outputs, self.output_shape())
        } else {
            Tensor::new(hidden, self.output_shape())
        }
    }
}

impl GRULayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.validate(input);

        let (n_in, n_h) = (self.input_size, self.hidden_size);
        let x = contiguous(input);

        let mut hidden = vec![0.0f32; n_h];
        let mut r = vec![0.0f32; n_h];
        let mut z = vec![0.0f32; n_h];
        let mut n = vec![0.0f32; n_h];
        let mut hn_term = vec![0.0f32; n_h];
        let mut act = Activator::new(n_h);
        let mut outputs: Vec<f32> = if self.return_sequences {
            Vec::with_capacity(self.seq_len * n_h)
        } else {
            Vec::new()
        };

        for t in 0..self.seq_len {
            let x_t = &x[t * n_in..(t + 1) * n_in];

            gate_preact(
                &self.weights_ir,
                &self.weights_hr,
                &self.bias_ir,
                &self.bias_hr,
                x_t,
                &hidden,
                &mut r,
            );
            act.apply(&self.recurrent_activation_type, &mut r);

            gate_preact(
                &self.weights_iz,
                &self.weights_hz,
                &self.bias_iz,
                &self.bias_hz,
                x_t,
                &hidden,
                &mut z,
            );
            act.apply(&self.recurrent_activation_type, &mut z);

            // n_pre = (W_in . x + b_in) + r * (W_hn . h + b_hn)
            affine(&self.weights_hn, &self.bias_hn, &hidden, &mut hn_term);
            affine(&self.weights_in, &self.bias_in, x_t, &mut n);
            for k in 0..n_h {
                n[k] += r[k] * hn_term[k];
            }
            act.apply(&self.activation_type, &mut n);

            // h_t = (1 - z) * n + z * h_{t-1}   (each k is independent, so in place)
            for k in 0..n_h {
                hidden[k] = (1.0 - z[k]) * n[k] + z[k] * hidden[k];
            }

            if self.return_sequences {
                outputs.extend_from_slice(&hidden);
            }
        }

        if self.return_sequences {
            Tensor::new(outputs, self.output_shape())
        } else {
            Tensor::new(hidden, self.output_shape())
        }
    }
}

impl LSTMLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.validate(input);

        let (n_in, n_h) = (self.input_size, self.hidden_size);
        let x = contiguous(input);

        let mut hidden = vec![0.0f32; n_h];
        let mut cell = vec![0.0f32; n_h];
        let mut i_gate = vec![0.0f32; n_h];
        let mut f_gate = vec![0.0f32; n_h];
        let mut g_gate = vec![0.0f32; n_h];
        let mut o_gate = vec![0.0f32; n_h];
        let mut cell_act = vec![0.0f32; n_h];
        let mut act = Activator::new(n_h);
        let mut outputs: Vec<f32> = if self.return_sequences {
            Vec::with_capacity(self.seq_len * n_h)
        } else {
            Vec::new()
        };

        for t in 0..self.seq_len {
            let x_t = &x[t * n_in..(t + 1) * n_in];

            gate_preact(
                &self.weights_ii,
                &self.weights_hi,
                &self.bias_ii,
                &self.bias_hi,
                x_t,
                &hidden,
                &mut i_gate,
            );
            act.apply(&self.recurrent_activation_type, &mut i_gate);

            gate_preact(
                &self.weights_if,
                &self.weights_hf,
                &self.bias_if,
                &self.bias_hf,
                x_t,
                &hidden,
                &mut f_gate,
            );
            act.apply(&self.recurrent_activation_type, &mut f_gate);

            gate_preact(
                &self.weights_ig,
                &self.weights_hg,
                &self.bias_ig,
                &self.bias_hg,
                x_t,
                &hidden,
                &mut g_gate,
            );
            act.apply(&self.activation_type, &mut g_gate);

            gate_preact(
                &self.weights_io,
                &self.weights_ho,
                &self.bias_io,
                &self.bias_ho,
                x_t,
                &hidden,
                &mut o_gate,
            );
            act.apply(&self.recurrent_activation_type, &mut o_gate);

            // c_t = f * c_{t-1} + i * g
            for k in 0..n_h {
                cell[k] = f_gate[k] * cell[k] + i_gate[k] * g_gate[k];
            }

            // h_t = o * cell_activation(c_t)
            cell_act.copy_from_slice(&cell);
            act.apply(&self.cell_activation_type, &mut cell_act);
            for k in 0..n_h {
                hidden[k] = o_gate[k] * cell_act[k];
            }

            if self.return_sequences {
                outputs.extend_from_slice(&hidden);
            }
        }

        if self.return_sequences {
            Tensor::new(outputs, self.output_shape())
        } else {
            Tensor::new(hidden, self.output_shape())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- naive references, independent of the SIMD code path ----------

    fn matvec(w: &[f32], b: &[f32], v: &[f32]) -> Vec<f32> {
        let cols = v.len();
        (0..b.len())
            .map(|i| (0..cols).map(|j| w[i * cols + j] * v[j]).sum::<f32>() + b[i])
            .collect()
    }

    fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        // Small deterministic values so gates don't saturate.
        (0..n)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed * 7919)) >> 20;
                ((v % 17) as f32 / 16.0 - 0.5) * 0.5
            })
            .collect()
    }

    // Sizes chosen to hit both the 8-lane body and the scalar tail:
    // input 11 = 8 + 3, hidden 10 = 8 + 2.
    const SEQ: usize = 4;
    const N_IN: usize = 11;
    const N_H: usize = 10;

    fn check_outputs(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-4);
        }
    }

    #[test]
    fn dot_covers_simd_chunks_and_tail() {
        let a: Vec<f32> = (0..19).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..19).map(|x| 1.0 + x as f32).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert_eq!(dot(&a, &b), expected);
    }

    #[test]
    fn gate_preact_matches_naive_with_tails() {
        let wx = pseudo(N_H * N_IN, 1);
        let wh = pseudo(N_H * N_H, 2);
        let bx = pseudo(N_H, 3);
        let bh = pseudo(N_H, 4);
        let x = pseudo(N_IN, 5);
        let h = pseudo(N_H, 6);

        let expected = add(&matvec(&wx, &bx, &x), &matvec(&wh, &bh, &h));
        let mut got = vec![0.0; N_H];
        gate_preact(&wx, &wh, &bx, &bh, &x, &h, &mut got);
        check_outputs(&got, &expected);
    }

    #[test]
    fn rnn_matches_reference_with_tails() {
        let layer = RNNLayer {
            seq_len: SEQ,
            input_size: N_IN,
            hidden_size: N_H,
            weights_ih: pseudo(N_H * N_IN, 1),
            weights_hh: pseudo(N_H * N_H, 2),
            bias_ih: pseudo(N_H, 3),
            bias_hh: pseudo(N_H, 4),
            activation_type: ActivationType::Tanh,
            return_sequences: true,
        };
        let data = pseudo(SEQ * N_IN, 9);

        let mut h = vec![0.0f32; N_H];
        let mut expected = Vec::new();
        for t in 0..SEQ {
            let x_t = &data[t * N_IN..(t + 1) * N_IN];
            let z = add(
                &matvec(&layer.weights_ih, &layer.bias_ih, x_t),
                &matvec(&layer.weights_hh, &layer.bias_hh, &h),
            );
            h = z.iter().map(|v| v.tanh()).collect();
            expected.extend_from_slice(&h);
        }

        let got = layer
            .forward(&Tensor::new(data, layer.input_shape()))
            .to_vec();
        check_outputs(&got, &expected);
    }

    #[test]
    fn gru_matches_reference_with_tails() {
        let mut s = 0u32;
        let mut w = |n: usize| {
            s += 1;
            pseudo(n, s)
        };
        let layer = GRULayer {
            seq_len: SEQ,
            input_size: N_IN,
            hidden_size: N_H,
            weights_ir: w(N_H * N_IN),
            weights_hr: w(N_H * N_H),
            bias_ir: w(N_H),
            bias_hr: w(N_H),
            weights_iz: w(N_H * N_IN),
            weights_hz: w(N_H * N_H),
            bias_iz: w(N_H),
            bias_hz: w(N_H),
            weights_in: w(N_H * N_IN),
            weights_hn: w(N_H * N_H),
            bias_in: w(N_H),
            bias_hn: w(N_H),
            recurrent_activation_type: ActivationType::Sigmoid,
            activation_type: ActivationType::Tanh,
            return_sequences: true,
        };
        let data = pseudo(SEQ * N_IN, 99);

        let mut h = vec![0.0f32; N_H];
        let mut expected = Vec::new();
        for t in 0..SEQ {
            let x_t = &data[t * N_IN..(t + 1) * N_IN];
            let r: Vec<f32> = add(
                &matvec(&layer.weights_ir, &layer.bias_ir, x_t),
                &matvec(&layer.weights_hr, &layer.bias_hr, &h),
            )
            .into_iter()
            .map(sigmoid)
            .collect();
            let z: Vec<f32> = add(
                &matvec(&layer.weights_iz, &layer.bias_iz, x_t),
                &matvec(&layer.weights_hz, &layer.bias_hz, &h),
            )
            .into_iter()
            .map(sigmoid)
            .collect();
            let xin = matvec(&layer.weights_in, &layer.bias_in, x_t);
            let hn = matvec(&layer.weights_hn, &layer.bias_hn, &h);
            let n: Vec<f32> = (0..N_H).map(|k| (xin[k] + r[k] * hn[k]).tanh()).collect();
            h = (0..N_H).map(|k| (1.0 - z[k]) * n[k] + z[k] * h[k]).collect();
            expected.extend_from_slice(&h);
        }

        let got = layer
            .forward(&Tensor::new(data, layer.input_shape()))
            .to_vec();
        check_outputs(&got, &expected);
    }

    #[test]
    fn lstm_matches_reference_with_tails() {
        let mut s = 0u32;
        let mut w = |n: usize| {
            s += 1;
            pseudo(n, s)
        };
        let layer = LSTMLayer {
            seq_len: SEQ,
            input_size: N_IN,
            hidden_size: N_H,
            weights_ii: w(N_H * N_IN),
            weights_hi: w(N_H * N_H),
            bias_ii: w(N_H),
            bias_hi: w(N_H),
            weights_if: w(N_H * N_IN),
            weights_hf: w(N_H * N_H),
            bias_if: w(N_H),
            bias_hf: w(N_H),
            weights_ig: w(N_H * N_IN),
            weights_hg: w(N_H * N_H),
            bias_ig: w(N_H),
            bias_hg: w(N_H),
            weights_io: w(N_H * N_IN),
            weights_ho: w(N_H * N_H),
            bias_io: w(N_H),
            bias_ho: w(N_H),
            recurrent_activation_type: ActivationType::Sigmoid,
            activation_type: ActivationType::Tanh,
            cell_activation_type: ActivationType::Tanh,
            return_sequences: true,
        };
        let data = pseudo(SEQ * N_IN, 77);

        let mut h = vec![0.0f32; N_H];
        let mut c = vec![0.0f32; N_H];
        let mut expected = Vec::new();
        for t in 0..SEQ {
            let x_t = &data[t * N_IN..(t + 1) * N_IN];
            let i: Vec<f32> = add(
                &matvec(&layer.weights_ii, &layer.bias_ii, x_t),
                &matvec(&layer.weights_hi, &layer.bias_hi, &h),
            )
            .into_iter()
            .map(sigmoid)
            .collect();
            let f: Vec<f32> = add(
                &matvec(&layer.weights_if, &layer.bias_if, x_t),
                &matvec(&layer.weights_hf, &layer.bias_hf, &h),
            )
            .into_iter()
            .map(sigmoid)
            .collect();
            let g: Vec<f32> = add(
                &matvec(&layer.weights_ig, &layer.bias_ig, x_t),
                &matvec(&layer.weights_hg, &layer.bias_hg, &h),
            )
            .into_iter()
            .map(|v| v.tanh())
            .collect();
            let o: Vec<f32> = add(
                &matvec(&layer.weights_io, &layer.bias_io, x_t),
                &matvec(&layer.weights_ho, &layer.bias_ho, &h),
            )
            .into_iter()
            .map(sigmoid)
            .collect();
            c = (0..N_H).map(|k| f[k] * c[k] + i[k] * g[k]).collect();
            h = (0..N_H).map(|k| o[k] * c[k].tanh()).collect();
            expected.extend_from_slice(&h);
        }

        let got = layer
            .forward(&Tensor::new(data, layer.input_shape()))
            .to_vec();
        check_outputs(&got, &expected);
    }

    #[test]
    fn final_only_returns_last_timestep_of_sequence() {
        // return_sequences = false must equal the last row of the full run.
        let make = |return_sequences| RNNLayer {
            seq_len: SEQ,
            input_size: N_IN,
            hidden_size: N_H,
            weights_ih: pseudo(N_H * N_IN, 1),
            weights_hh: pseudo(N_H * N_H, 2),
            bias_ih: pseudo(N_H, 3),
            bias_hh: pseudo(N_H, 4),
            activation_type: ActivationType::Tanh,
            return_sequences,
        };
        let data = pseudo(SEQ * N_IN, 9);
        let full = make(true);
        let last = make(false);

        let all = full
            .forward(&Tensor::new(data.clone(), full.input_shape()))
            .to_vec();
        let final_h = last
            .forward(&Tensor::new(data, last.input_shape()))
            .to_vec();

        assert_eq!(final_h, all[(SEQ - 1) * N_H..].to_vec());
    }
}