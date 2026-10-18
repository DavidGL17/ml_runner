//! SIMD-accelerated forward-pass implementation for `LinearLayer`, via the
//! `wide` crate, which picks portable SIMD instructions (SSE/AVX on
//! x86_64, NEON on aarch64, etc.) at compile time based on the build's
//! target features - no runtime CPU detection involved. Only compiled in
//! when the `simd` Cargo feature is enabled.
//!
//! Unlike the default backend (`dense/scalar.rs`), this doesn't route
//! through `ndarray`'s `.dot()`/BLAS at all, so nothing needs to be linked
//! at build time - this is the option for embedded/microcontroller
//! targets that support SIMD instructions but have no BLAS port available.
//!
//! On aarch64 the `wide` kernel is replaced by hand-written NEON
//! intrinsics (`core::arch::aarch64`, NEON being baseline there). NEON
//! registers are 128-bit, so `wide`'s `f32x8` is emulated as two `f32x4`
//! halves with a single dependency chain; the NEON kernel instead computes
//! 4 output rows per pass (each input load is reused across 4 rows) with 8
//! independent fused multiply-add accumulators.

use super::LinearLayer;
use crate::tensor::Tensor;
#[cfg(not(target_arch = "aarch64"))]
use wide::f32x8;

#[cfg(not(target_arch = "aarch64"))]
const LANES: usize = 8;

/// Computes `out = weights * input + bias` for a fully connected layer,
/// processing 8 elements of each dot product at a time.
#[cfg(not(target_arch = "aarch64"))]
fn dense_forward(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    input_size: usize,
    output_size: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), input_size);
    debug_assert_eq!(bias.len(), output_size);
    debug_assert_eq!(weights.len(), input_size * output_size);
    debug_assert_eq!(out.len(), output_size);

    let chunks = input_size / LANES;
    let remainder_start = chunks * LANES;

    for i in 0..output_size {
        let row = &weights[i * input_size..(i + 1) * input_size];

        let mut acc = f32x8::ZERO;
        for c in 0..chunks {
            let base = c * LANES;
            let a = f32x8::from(<[f32; LANES]>::try_from(&input[base..base + LANES]).unwrap());
            let b = f32x8::from(<[f32; LANES]>::try_from(&row[base..base + LANES]).unwrap());
            acc += a * b;
        }

        let mut sum: f32 = acc.reduce_add();
        // Scalar tail for input sizes that aren't a multiple of LANES.
        for j in remainder_start..input_size {
            sum += input[j] * row[j];
        }
        out[i] = sum + bias[i];
    }
}

/// Output rows computed per pass of the NEON kernel.
#[cfg(target_arch = "aarch64")]
const ROWS: usize = 4;

/// NEON version of `dense_forward` (same contract): blocks of `ROWS`
/// output rows go through `dot4`, leftover rows through `dot1`.
#[cfg(target_arch = "aarch64")]
fn dense_forward(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    input_size: usize,
    output_size: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), input_size);
    debug_assert_eq!(bias.len(), output_size);
    debug_assert_eq!(weights.len(), input_size * output_size);
    debug_assert_eq!(out.len(), output_size);

    let blocked = output_size / ROWS * ROWS;
    for i in (0..blocked).step_by(ROWS) {
        let sums = dot4(input, &weights[i * input_size..(i + ROWS) * input_size]);
        for r in 0..ROWS {
            out[i + r] = sums[r] + bias[i + r];
        }
    }
    for i in blocked..output_size {
        out[i] = dot1(input, &weights[i * input_size..(i + 1) * input_size]) + bias[i];
    }
}

/// Dot products of `x` with each of the `ROWS` consecutive rows packed in
/// `w` (row-major, `ROWS x x.len()`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot4(x: &[f32], w: &[f32]) -> [f32; ROWS] {
    use core::arch::aarch64::*;

    let n = x.len();
    // A real assert rather than a debug one: it's what keeps the raw
    // pointer reads below in bounds.
    assert_eq!(w.len(), ROWS * n);

    let mut sums = [0.0f32; ROWS];
    let mut k = 0;
    // SAFETY: every load reads 4 floats at offset `k` (+4) of a row with
    // `k + 8 <= n` (or `k + 4 <= n`), and each row `r` spans
    // `w[r * n..(r + 1) * n]`, so all reads stay inside `x` and `w`.
    unsafe {
        let xp = x.as_ptr();
        let (w0, w1, w2, w3) = (w.as_ptr(), w.as_ptr().add(n), w.as_ptr().add(2 * n), w.as_ptr().add(3 * n));

        let zero = vdupq_n_f32(0.0);
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        let (mut a4, mut a5, mut a6, mut a7) = (zero, zero, zero, zero);

        while k + 8 <= n {
            let x0 = vld1q_f32(xp.add(k));
            let x1 = vld1q_f32(xp.add(k + 4));
            a0 = vfmaq_f32(a0, vld1q_f32(w0.add(k)), x0);
            a1 = vfmaq_f32(a1, vld1q_f32(w0.add(k + 4)), x1);
            a2 = vfmaq_f32(a2, vld1q_f32(w1.add(k)), x0);
            a3 = vfmaq_f32(a3, vld1q_f32(w1.add(k + 4)), x1);
            a4 = vfmaq_f32(a4, vld1q_f32(w2.add(k)), x0);
            a5 = vfmaq_f32(a5, vld1q_f32(w2.add(k + 4)), x1);
            a6 = vfmaq_f32(a6, vld1q_f32(w3.add(k)), x0);
            a7 = vfmaq_f32(a7, vld1q_f32(w3.add(k + 4)), x1);
            k += 8;
        }
        if k + 4 <= n {
            let x0 = vld1q_f32(xp.add(k));
            a0 = vfmaq_f32(a0, vld1q_f32(w0.add(k)), x0);
            a2 = vfmaq_f32(a2, vld1q_f32(w1.add(k)), x0);
            a4 = vfmaq_f32(a4, vld1q_f32(w2.add(k)), x0);
            a6 = vfmaq_f32(a6, vld1q_f32(w3.add(k)), x0);
            k += 4;
        }

        // Pairwise adds fold the four rows' accumulators into one vector
        // holding [row0, row1, row2, row3].
        let r01 = vpaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        let r23 = vpaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7));
        vst1q_f32(sums.as_mut_ptr(), vpaddq_f32(r01, r23));
    }

    // Scalar tail for input sizes that aren't a multiple of 4.
    for j in k..n {
        for (r, s) in sums.iter_mut().enumerate() {
            *s += x[j] * w[r * n + j];
        }
    }
    sums
}

/// Dot product of `x` and `w`, with 4 independent accumulators.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot1(x: &[f32], w: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let n = x.len();
    assert_eq!(w.len(), n);

    let mut sum;
    let mut k = 0;
    // SAFETY: as in `dot4`, every 4-float load at offset `k` has
    // `k + 4 <= n`.
    unsafe {
        let (xp, wp) = (x.as_ptr(), w.as_ptr());
        let zero = vdupq_n_f32(0.0);
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        while k + 16 <= n {
            a0 = vfmaq_f32(a0, vld1q_f32(wp.add(k)), vld1q_f32(xp.add(k)));
            a1 = vfmaq_f32(a1, vld1q_f32(wp.add(k + 4)), vld1q_f32(xp.add(k + 4)));
            a2 = vfmaq_f32(a2, vld1q_f32(wp.add(k + 8)), vld1q_f32(xp.add(k + 8)));
            a3 = vfmaq_f32(a3, vld1q_f32(wp.add(k + 12)), vld1q_f32(xp.add(k + 12)));
            k += 16;
        }
        while k + 4 <= n {
            a0 = vfmaq_f32(a0, vld1q_f32(wp.add(k)), vld1q_f32(xp.add(k)));
            k += 4;
        }
        sum = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    }

    for j in k..n {
        sum += x[j] * w[j];
    }
    sum
}

impl LinearLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in LinearLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let input_slice = input
            .data
            .as_slice()
            .expect("LinearLayer input must be a contiguous 1-D tensor");

        let mut output = vec![0.0; self.output_size];
        dense_forward(
            input_slice,
            &self.weights,
            &self.bias,
            self.input_size,
            self.output_size,
            &mut output,
        );

        Tensor::new(output, self.output_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

    #[test]
    fn matches_expected_output_small() {
        let input = vec![1.0, 2.0];
        let weights = vec![0.5, 0.5];
        let bias = vec![0.1];
        let mut out = vec![0.0; 1];
        dense_forward(&input, &weights, &bias, 2, 1, &mut out);
        assert_eq!(out, vec![1.6]);
    }

    #[test]
    fn matches_expected_output_wider_than_one_simd_register() {
        // input_size = 10 exercises both the SIMD chunk (8 lanes) and the
        // scalar tail (2 leftover elements).
        let input: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let weights = vec![1.0; 10];
        let bias = vec![0.0];
        let mut out = vec![0.0; 1];
        dense_forward(&input, &weights, &bias, 10, 1, &mut out);
        let expected: f32 = input.iter().sum();
        assert_eq!(out, vec![expected]);
    }

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn matches_naive_reference_across_row_blocks_and_tails() {
        // Covers every `output_size % 4` leftover and every input tail
        // (multiples of 8, an extra 4-chunk, scalar leftovers, < 4 wide,
        // the 16-wide single-row loop), plus a LeNet300100-sized layer.
        let shapes = (1..=21)
            .flat_map(|i| (1..=9).map(move |o| (i, o)))
            .chain([(784, 300), (300, 100), (100, 10)]);
        for (seed, (in_n, out_n)) in shapes.enumerate() {
            let input = pseudo(in_n, seed as u32);
            let weights = pseudo(in_n * out_n, seed as u32 + 1000);
            let bias = pseudo(out_n, seed as u32 + 2000);

            let mut out = vec![0.0; out_n];
            dense_forward(&input, &weights, &bias, in_n, out_n, &mut out);

            for o in 0..out_n {
                let row = &weights[o * in_n..(o + 1) * in_n];
                let expected: f32 = row.iter().zip(&input).map(|(w, x)| w * x).sum::<f32>() + bias[o];
                assert!(
                    (out[o] - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                    "({in_n} -> {out_n}) row {o}: got {}, expected {expected}",
                    out[o]
                );
            }
        }
    }

    #[test]
    fn test_dense_layer_forward() {
        let layer = LinearLayer {
            input_size: 2,
            output_size: 1,
            weights: vec![0.5, 0.5],
            bias: vec![0.1],
        };
        let input = Tensor::new(vec![1.0, 2.0], TensorShape::Flat(2));
        let output = layer.forward(&input);
        // (1.0 * 0.5) + (2.0 * 0.5) + 0.1 = 1.6
        assert_eq!(output.to_vec(), vec![1.6]);
        assert_eq!(output.shape(), TensorShape::Flat(1));
    }

    #[test]
    fn test_dense_layer_forward_multi_output() {
        let layer = LinearLayer {
            input_size: 2,
            output_size: 2,
            // row 0: [1.0, 0.0], row 1: [0.0, 1.0] -> identity-ish
            weights: vec![1.0, 0.0, 0.0, 1.0],
            bias: vec![0.0, 100.0],
        };
        let input = Tensor::new(vec![3.0, 4.0], TensorShape::Flat(2));
        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![3.0, 104.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch in LinearLayer")]
    fn test_dense_layer_wrong_input_shape() {
        let layer = LinearLayer {
            input_size: 2,
            output_size: 1,
            weights: vec![0.5, 0.5],
            bias: vec![0.1],
        };
        let input = Tensor::new(vec![1.0], TensorShape::Flat(1));
        let _ = layer.forward(&input);
    }
}
