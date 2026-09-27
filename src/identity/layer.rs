use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

/// A no-op layer: output equals input, both in shape and data. Useful as
/// the Rust-side counterpart of ONNX's `Identity` op, which shows up for
/// things like a graph output that's just an aliased/renamed intermediate
/// tensor, or a pass-through left behind by some exporters' optimization
/// passes.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IdentityLayer {
    pub shape: TensorShape,
}

impl IdentityLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        self.shape.clone()
    }

    /// The shape this layer produces, given a matching input shape.
    /// Identical to the input shape - this layer never changes shape.
    pub fn output_shape(&self) -> TensorShape {
        self.shape.clone()
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in IdentityLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        input.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_output_shape_matches_input_shape() {
        let layer = IdentityLayer {
            shape: TensorShape::D3 {
                dim1: 3,
                dim2: 4,
                dim3: 5,
            },
        };

        assert_eq!(layer.input_shape(), layer.output_shape());
        assert_eq!(
            layer.output_shape(),
            TensorShape::D3 {
                dim1: 3,
                dim2: 4,
                dim3: 5,
            }
        );
    }

    #[test]
    fn test_forward_preserves_data_and_shape() {
        let layer = IdentityLayer {
            shape: TensorShape::D2 { dim1: 2, dim2: 2 },
        };

        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], layer.input_shape());
        let output = layer.forward(&input);

        assert_eq!(output.shape(), TensorShape::D2 { dim1: 2, dim2: 2 });
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_forward_on_flat_input() {
        let layer = IdentityLayer {
            shape: TensorShape::Flat(3),
        };

        let input = Tensor::new(vec![7.0, 8.0, 9.0], layer.input_shape());
        let output = layer.forward(&input);

        assert_eq!(output.shape(), TensorShape::Flat(3));
        assert_eq!(output.to_vec(), vec![7.0, 8.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = IdentityLayer {
            shape: TensorShape::D3 {
                dim1: 3,
                dim2: 4,
                dim3: 5,
            },
        };

        let wrong_input = Tensor::new(vec![0.0; 10], TensorShape::Flat(10));
        layer.forward(&wrong_input);
    }
}