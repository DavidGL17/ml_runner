use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

/// A general row-major reshape - the Rust-side counterpart of ONNX's
/// `Reshape` when it isn't the `(batch, -1)` flatten pattern handled by
/// `FlattenLayer`. Like `SqueezeLayer`/`UnsqueezeLayer` it only changes the
/// declared shape, never the data order, so both shapes are stored
/// explicitly and just have to hold the same number of elements.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReshapeLayer {
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
}

impl ReshapeLayer {
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
            "Shape mismatch in ReshapeLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        assert_eq!(
            self.input_shape.total_size(),
            self.output_shape.total_size(),
            "ReshapeLayer: input shape {:?} and output shape {:?} don't hold the same number of elements",
            self.input_shape,
            self.output_shape
        );

        let reshaped = input
            .data
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order(self.output_shape.to_ixdyn())
            .expect("ReshapeLayer: total element count changed during reshape");

        Tensor::from_array(reshaped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_keeps_row_major_order() {
        let layer = ReshapeLayer {
            input_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
            output_shape: TensorShape::D3 { dim1: 3, dim2: 2, dim3: 1 },
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::D3 { dim1: 3, dim2: 2, dim3: 1 });
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_forward_to_flat() {
        let layer = ReshapeLayer {
            input_shape: TensorShape::D2 { dim1: 1, dim2: 4 },
            output_shape: TensorShape::Flat(4),
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::Flat(4));
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = ReshapeLayer {
            input_shape: TensorShape::D2 { dim1: 1, dim2: 4 },
            output_shape: TensorShape::Flat(4),
        };
        layer.forward(&Tensor::new(vec![0.0; 3], TensorShape::Flat(3)));
    }

    #[test]
    #[should_panic(expected = "don't hold the same number of elements")]
    fn test_forward_rejects_size_changing_shapes() {
        let layer = ReshapeLayer {
            input_shape: TensorShape::D2 { dim1: 1, dim2: 4 },
            output_shape: TensorShape::Flat(3),
        };
        layer.forward(&Tensor::new(vec![0.0; 4], layer.input_shape()));
    }
}
