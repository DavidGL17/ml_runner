//! The `DenseLayer` type and its forward pass, kept alongside the
//! pluggable `dense_forward` backends (see `mod.rs`) that it delegates to.

use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DenseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl DenseLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::Flat(self.input_size)
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        TensorShape::Flat(self.output_size)
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape,
            self.input_shape(),
            "Shape mismatch in DenseLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape
        );

        let mut output = vec![0.0; self.output_size];

        // Which implementation this call resolves to (scalar, SIMD, ...) is decided at compile time by Cargo features - see src/dense/mod.rs.
        super::dense_forward(
            &input.data,
            &self.weights,
            &self.bias,
            self.input_size,
            self.output_size,
            &mut output,
        );

        Tensor::new(output.to_vec(), self.output_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer {
            input_size: 2,
            output_size: 1,
            weights: vec![0.5, 0.5],
            bias: vec![0.1],
        };
        let input = Tensor::flat(vec![1.0, 2.0]);
        let output = layer.forward(&input);
        // (1.0 * 0.5) + (2.0 * 0.5) + 0.1 = 1.6
        assert_eq!(output.data, vec![1.6]);
        assert_eq!(output.shape, TensorShape::Flat(1));
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
        let input = Tensor::flat(vec![1.0]);
        let _ = layer.forward(&input);
    }
}
