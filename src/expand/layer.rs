use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

/// Broadcasts a tensor up to a larger shape - the Rust-side counterpart of
/// ONNX's `Expand`. Follows numpy/ONNX broadcasting rules: shapes are
/// aligned from the trailing dimension, and every input dim must either
/// equal the target dim or be 1. The target shape is a constant in the
/// exported graph, so it's stored here as a declared `output_shape` (the
/// same way `Unsqueeze`/`Shape`/`Gather` declare theirs) rather than being
/// read from a second input tensor at runtime.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExpandLayer {
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
}

impl ExpandLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        self.output_shape.clone()
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in ExpandLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let view = input
            .data
            .broadcast(self.output_shape.to_ixdyn())
            .unwrap_or_else(|| {
                panic!(
                    "ExpandLayer: input shape {:?} can't be broadcast to output shape {:?}",
                    self.input_shape, self.output_shape
                )
            });

        Tensor::from_array(view.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_shapes_are_declared_independently() {
        let layer = ExpandLayer {
            input_shape: TensorShape::Flat(3),
            output_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
        };
        assert_eq!(layer.input_shape(), TensorShape::Flat(3));
        assert_eq!(layer.output_shape(), TensorShape::D2 { dim1: 2, dim2: 3 });
    }

    /// Same rank, size-1 dim stretched: (1, 3) -> (2, 3).
    #[test]
    fn test_forward_expands_size_one_dim() {
        let layer = ExpandLayer {
            input_shape: TensorShape::D2 { dim1: 1, dim2: 3 },
            output_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 2, dim2: 3 });
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    /// Trailing-aligned rank growth: (3,) -> (2, 3).
    #[test]
    fn test_forward_adds_leading_dim() {
        let layer = ExpandLayer {
            input_shape: TensorShape::Flat(3),
            output_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    /// Per-channel bias-style expand: (C, 1, 1) -> (C, H, W).
    #[test]
    fn test_forward_expands_channel_to_spatial() {
        let layer = ExpandLayer {
            input_shape: TensorShape::D3 { dim1: 2, dim2: 1, dim3: 1 },
            output_shape: TensorShape::D3 { dim1: 2, dim2: 2, dim3: 2 },
        };
        let input = Tensor::new(vec![5.0, 7.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(
            output.to_vec(),
            vec![5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0]
        );
    }

    #[test]
    fn test_forward_identity_when_shapes_match() {
        let layer = ExpandLayer {
            input_shape: TensorShape::Flat(2),
            output_shape: TensorShape::Flat(2),
        };
        let input = Tensor::new(vec![4.0, 5.0], layer.input_shape());
        assert_eq!(layer.forward(&input).to_vec(), vec![4.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = ExpandLayer {
            input_shape: TensorShape::Flat(3),
            output_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
        };
        let wrong = Tensor::new(vec![0.0; 2], TensorShape::Flat(2));
        layer.forward(&wrong);
    }

    #[test]
    #[should_panic(expected = "can't be broadcast")]
    fn test_forward_rejects_unbroadcastable_shapes() {
        let layer = ExpandLayer {
            input_shape: TensorShape::Flat(3),
            output_shape: TensorShape::D2 { dim1: 2, dim2: 4 },
        };
        let input = Tensor::new(vec![0.0; 3], layer.input_shape());
        layer.forward(&input);
    }
}