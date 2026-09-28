use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

/// Removes size-1 axes from a tensor's declared shape without touching its
/// data - the Rust-side counterpart of ONNX's `Squeeze`, and the inverse of
/// `UnsqueezeLayer`. Like `Unsqueeze` (and `Flatten`) this is a
/// total-size-preserving reshape, so `output_shape` is stored explicitly
/// rather than derived: which axes were squeezed is the exporter's concern,
/// and the runtime only needs the two declared shapes to line up with its
/// neighbours.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SqueezeLayer {
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
}

impl SqueezeLayer {
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
            "Shape mismatch in SqueezeLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        assert_eq!(
            self.input_shape.total_size(),
            self.output_shape.total_size(),
            "SqueezeLayer: input shape {:?} and output shape {:?} don't hold the same number of elements",
            self.input_shape,
            self.output_shape
        );

        let reshaped = input
            .data
            .clone()
            .into_shape_with_order(self.output_shape().dims())
            .expect("SqueezeLayer: total element count changed during reshape");

        Tensor::from_array(reshaped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squeeze_shapes_are_declared_independently() {
        let layer = SqueezeLayer {
            input_shape: TensorShape::D2 { dim1: 1, dim2: 4 },
            output_shape: TensorShape::Flat(4),
        };
        assert_eq!(layer.input_shape(), TensorShape::D2 { dim1: 1, dim2: 4 });
        assert_eq!(layer.output_shape(), TensorShape::Flat(4));
    }

    #[test]
    fn test_forward_removes_leading_axis() {
        let layer = SqueezeLayer {
            input_shape: TensorShape::D2 { dim1: 1, dim2: 3 },
            output_shape: TensorShape::Flat(3),
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::Flat(3));
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_forward_removes_trailing_axis() {
        let layer = SqueezeLayer {
            input_shape: TensorShape::D2 { dim1: 3, dim2: 1 },
            output_shape: TensorShape::Flat(3),
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::Flat(3));
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    /// Global-average-pool style: (C, 1, 1) -> (C).
    #[test]
    fn test_forward_removes_multiple_axes() {
        let layer = SqueezeLayer {
            input_shape: TensorShape::D3 { dim1: 2, dim2: 1, dim3: 1 },
            output_shape: TensorShape::Flat(2),
        };
        let input = Tensor::new(vec![5.0, 7.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::Flat(2));
        assert_eq!(output.to_vec(), vec![5.0, 7.0]);
    }

    /// Squeezing an axis out of the middle of a rank-3 tensor: (2, 1, 3) -> (2, 3).
    #[test]
    fn test_forward_removes_middle_axis() {
        let layer = SqueezeLayer {
            input_shape: TensorShape::D3 { dim1: 2, dim2: 1, dim3: 3 },
            output_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 2, dim2: 3 });
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_forward_on_single_element_tensor() {
        let layer = SqueezeLayer {
            input_shape: TensorShape::D2 { dim1: 1, dim2: 1 },
            output_shape: TensorShape::Flat(1),
        };
        let input = Tensor::new(vec![7.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::Flat(1));
        assert_eq!(output.to_vec(), vec![7.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = SqueezeLayer {
            input_shape: TensorShape::D2 { dim1: 1, dim2: 4 },
            output_shape: TensorShape::Flat(4),
        };
        let wrong_input = Tensor::new(vec![0.0; 3], TensorShape::Flat(3));
        layer.forward(&wrong_input);
    }

    #[test]
    #[should_panic(expected = "don't hold the same number of elements")]
    fn test_forward_rejects_size_changing_shapes() {
        let layer = SqueezeLayer {
            input_shape: TensorShape::D2 { dim1: 1, dim2: 4 },
            output_shape: TensorShape::Flat(3),
        };
        let input = Tensor::new(vec![0.0; 4], layer.input_shape());
        layer.forward(&input);
    }
}