//! The `DenseLayer` type and its forward pass, kept alongside the
//! pluggable `dense_forward` backends (see `mod.rs`) that it delegates to.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DenseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl DenseLayer {
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.input_size,
            "Input size mismatch in DenseLayer"
        );

        let mut output = vec![0.0; self.output_size];

        // Which implementation this call resolves to (scalar, SIMD, ...) is
        // decided at compile time by Cargo features - see src/dense/mod.rs.
        super::dense_forward(
            input,
            &self.weights,
            &self.bias,
            self.input_size,
            self.output_size,
            &mut output,
        );

        output
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
        let input = vec![1.0, 2.0];
        let output = layer.forward(&input);
        // (1.0 * 0.5) + (2.0 * 0.5) + 0.1 = 1.6
        assert_eq!(output, vec![1.6]);
    }

    #[test]
    #[should_panic(expected = "Input size mismatch in DenseLayer")]
    fn test_dense_layer_wrong_input_size() {
        let layer = DenseLayer {
            input_size: 2,
            output_size: 1,
            weights: vec![0.5, 0.5],
            bias: vec![0.1],
        };
        let input = vec![1.0];
        let _ = layer.forward(&input);
    }
}
