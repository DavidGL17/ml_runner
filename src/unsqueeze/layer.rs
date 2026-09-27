use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

/// Inserts a size-1 axis into a tensor's declared shape without touching
/// its data - the Rust-side counterpart of ONNX's `Unsqueeze`. Since the
/// data itself never changes (only how its element count is split across
/// dimensions), this is really the same operation `FlattenLayer` performs
/// - a total-size-preserving reshape - just going the other direction
/// (rank grows instead of collapsing to 1), which is why `output_shape`
/// is stored explicitly here rather than derived, the same way
/// `ShapeLayer`/`GatherLayer` declare both shapes independently instead
/// of computing one from the other.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UnsqueezeLayer {
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
}

impl UnsqueezeLayer {
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
            "Shape mismatch in UnsqueezeLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        assert_eq!(
            self.input_shape.total_size(),
            self.output_shape.total_size(),
            "UnsqueezeLayer: input shape {:?} and output shape {:?} don't hold the same number of elements",
            self.input_shape,
            self.output_shape
        );

        let reshaped = input
            .data
            .clone()
            .into_shape_with_order(self.output_shape().dims())
            .expect("UnsqueezeLayer: total element count changed during reshape");

        Tensor::from_array(reshaped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsqueeze_shapes_are_declared_independently() {
        let layer = UnsqueezeLayer {
            input_shape: TensorShape::Flat(4),
            output_shape: TensorShape::D2 { dim1: 1, dim2: 4 },
        };
        assert_eq!(layer.input_shape(), TensorShape::Flat(4));
        assert_eq!(layer.output_shape(), TensorShape::D2 { dim1: 1, dim2: 4 });
    }

    #[test]
    fn test_forward_inserts_leading_axis() {
        let layer = UnsqueezeLayer {
            input_shape: TensorShape::Flat(3),
            output_shape: TensorShape::D2 { dim1: 1, dim2: 3 },
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 1, dim2: 3 });
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_forward_inserts_trailing_axis() {
        let layer = UnsqueezeLayer {
            input_shape: TensorShape::Flat(3),
            output_shape: TensorShape::D2 { dim1: 3, dim2: 1 },
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 3, dim2: 1 });
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    /// A single scalar-ish value (Flat(1), e.g. the output of a Gather
    /// with a scalar index - see the Python exporter's gather.py) turning
    /// into a 1x1 tensor: the degenerate case that shows up most often in
    /// dynamic-shape-computation chains.
    #[test]
    fn test_forward_on_single_element_tensor() {
        let layer = UnsqueezeLayer {
            input_shape: TensorShape::Flat(1),
            output_shape: TensorShape::D2 { dim1: 1, dim2: 1 },
        };
        let input = Tensor::new(vec![7.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 1, dim2: 1 });
        assert_eq!(output.to_vec(), vec![7.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = UnsqueezeLayer {
            input_shape: TensorShape::Flat(4),
            output_shape: TensorShape::D2 { dim1: 1, dim2: 4 },
        };
        let wrong_input = Tensor::new(vec![0.0; 3], TensorShape::Flat(3));
        layer.forward(&wrong_input);
    }
}