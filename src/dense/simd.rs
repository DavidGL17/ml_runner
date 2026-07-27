//! SIMD-accelerated implementation of the dense-layer forward pass.
//!
//! Uses the `wide` crate, which picks portable SIMD instructions (SSE/AVX on
//! x86_64, NEON on aarch64, etc.) at compile time based on the build's
//! target features - no runtime CPU detection involved. Only compiled in
//! when the `simd` Cargo feature is enabled.

use wide::f32x8;

const LANES: usize = 8;

/// Computes `out = weights * input + bias` for a fully connected layer,
/// processing 8 elements of each dot product at a time.
pub fn dense_forward(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scalar_output_small() {
        let input = vec![1.0, 2.0];
        let weights = vec![0.5, 0.5];
        let bias = vec![0.1];
        let mut out = vec![0.0; 1];
        dense_forward(&input, &weights, &bias, 2, 1, &mut out);
        assert_eq!(out, vec![1.6]);
    }

    #[test]
    fn matches_scalar_output_wider_than_one_simd_register() {
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
}
