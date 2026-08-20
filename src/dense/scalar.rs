//! Default forward-pass implementation for `DenseLayer`, via `ndarray`'s
//! `.dot()`. Weights/bias stay as flat `Vec<f32>` fields on `DenseLayer`
//! (so the JSON model format is unaffected) but are borrowed here as
//! `ndarray` views - no copying - for the matrix-vector multiply.
//!
//! `.dot()` resolves to a BLAS `sgemv` call when this crate is built with
//! the `blas` feature, or to `ndarray`'s own portable
//! `matrixmultiply`-backed implementation otherwise. Compiled in whenever
//! the `simd` feature is *not* enabled - see `dense/simd.rs` for the
//! alternative, BLAS-free backend.

use super::DenseLayer;
use crate::tensor::Tensor;
use ndarray::{Array1, ArrayView1, ArrayView2, Ix1};

impl DenseLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in DenseLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let weights = ArrayView2::from_shape((self.output_size, self.input_size), &self.weights)
            .expect("DenseLayer weights length doesn't match input_size * output_size");
        let bias = ArrayView1::from(&self.bias);
        let input_vec: ArrayView1<f32> = input
            .data
            .view()
            .into_dimensionality::<Ix1>()
            .expect("DenseLayer input is not 1-D");

        // weights (output_size x input_size) * input (input_size) + bias
        let output: Array1<f32> = weights.dot(&input_vec) + &bias;

        Tensor::from_array(output.into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

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
