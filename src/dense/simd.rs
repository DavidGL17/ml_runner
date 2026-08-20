//! SIMD-accelerated forward-pass implementation for `DenseLayer`, via the
//! `wide` crate, which picks portable SIMD instructions (SSE/AVX on
//! x86_64, NEON on aarch64, etc.) at compile time based on the build's
//! target features - no runtime CPU detection involved. Only compiled in
//! when the `simd` Cargo feature is enabled.
//!
//! Unlike the default backend (`dense/scalar.rs`), this doesn't route
//! through `ndarray`'s `.dot()`/BLAS at all, so nothing needs to be linked
//! at build time - this is the option for embedded/microcontroller
//! targets that support SIMD instructions but have no BLAS port available.

use super::DenseLayer;
use crate::tensor::Tensor;
use wide::f32x8;

const LANES: usize = 8;

/// Computes `out = weights * input + bias` for a fully connected layer,
/// processing 8 elements of each dot product at a time.
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

impl DenseLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in DenseLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let input_slice = input
            .data
            .as_slice()
            .expect("DenseLayer input must be a contiguous 1-D tensor");

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

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer {
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
        let layer = DenseLayer {
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
    #[should_panic(expected = "Shape mismatch in DenseLayer")]
    fn test_dense_layer_wrong_input_shape() {
        let layer = DenseLayer {
            input_size: 2,
            output_size: 1,
            weights: vec![0.5, 0.5],
            bias: vec![0.1],
        };
        let input = Tensor::new(vec![1.0], TensorShape::Flat(1));
        let _ = layer.forward(&input);
    }
}
