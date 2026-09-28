use crate::tensor::{Tensor, TensorShape};
use ndarray::{concatenate, Axis};
use serde::{Deserialize, Serialize};

/// Joins two or more tensors along `axis` into one - the Rust-side
/// counterpart of ONNX's `Concat`. Unlike `AddLayer` (the other
/// multi-input layer, where every operand shares one declared shape),
/// `Concat`'s operands generally differ along `axis` - e.g. joining a
/// `D3{32,8,8}` and a `D3{16,8,8}` into a `D3{48,8,8}` - so there's no
/// single "the" input shape to validate every input against. This
/// declares one shape per operand instead.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConcatLayer {
    pub axis: usize,
    pub input_shapes: Vec<TensorShape>,
    pub output_shape: TensorShape,
}

impl ConcatLayer {
    /// The declared shape of each operand, in the same order as the
    /// node's real tensor inputs. `Model::validate_shapes` checks each
    /// input position against its own entry here, instead of going
    /// through the generic single-shape `Layer::input_shape()` path that
    /// every other variant (including `Add`) uses.
    pub fn input_shapes(&self) -> Vec<TensorShape> {
        self.input_shapes.clone()
    }

    /// A single representative shape, kept only so `Layer::input_shape()`
    /// (used generically across every variant, e.g. by tests that just
    /// need "a" valid input shape to build a `Tensor` with) has something
    /// type-correct to return. Real shape validation always goes through
    /// `input_shapes()` instead.
    pub fn input_shape(&self) -> TensorShape {
        self.input_shapes
            .first()
            .cloned()
            .unwrap_or_else(|| self.output_shape.clone())
    }

    pub fn output_shape(&self) -> TensorShape {
        self.output_shape.clone()
    }

    pub fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        assert_eq!(
            inputs.len(),
            self.input_shapes.len(),
            "ConcatLayer: expected {} input tensor(s), got {}",
            self.input_shapes.len(),
            inputs.len()
        );

        for (i, (tensor, expected)) in inputs.iter().zip(self.input_shapes.iter()).enumerate() {
            assert_eq!(
                tensor.shape(),
                *expected,
                "Shape mismatch in ConcatLayer input {}: expected {:?}, got {:?}",
                i,
                expected,
                tensor.shape()
            );
        }

        let views: Vec<_> = inputs.iter().map(|t| t.data.view()).collect();
        let result = concatenate(Axis(self.axis), &views)
            .expect("ConcatLayer: inputs aren't concatenable along the declared axis");

        let result_shape = TensorShape::from_dims(result.shape());
        assert_eq!(
            result_shape, self.output_shape,
            "ConcatLayer: concatenated result shape {:?} doesn't match declared output shape {:?}",
            result_shape, self.output_shape
        );

        Tensor::from_array(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_output_shape_is_declared_independently() {
        let layer = ConcatLayer {
            axis: 0,
            input_shapes: vec![
                TensorShape::D3 {
                    dim1: 2,
                    dim2: 3,
                    dim3: 3,
                },
                TensorShape::D3 {
                    dim1: 1,
                    dim2: 3,
                    dim3: 3,
                },
            ],
            output_shape: TensorShape::D3 {
                dim1: 3,
                dim2: 3,
                dim3: 3,
            },
        };
        assert_eq!(
            layer.output_shape(),
            TensorShape::D3 {
                dim1: 3,
                dim2: 3,
                dim3: 3
            }
        );
    }

    #[test]
    fn test_forward_concatenates_along_leading_axis() {
        let layer = ConcatLayer {
            axis: 0,
            input_shapes: vec![
                TensorShape::D2 { dim1: 1, dim2: 2 },
                TensorShape::D2 { dim1: 2, dim2: 2 },
            ],
            output_shape: TensorShape::D2 { dim1: 3, dim2: 2 },
        };
        let a = Tensor::new(vec![1.0, 2.0], TensorShape::D2 { dim1: 1, dim2: 2 });
        let b = Tensor::new(
            vec![3.0, 4.0, 5.0, 6.0],
            TensorShape::D2 { dim1: 2, dim2: 2 },
        );
        let output = layer.forward(&[&a, &b]);
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 3, dim2: 2 });
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_forward_concatenates_along_trailing_axis() {
        let layer = ConcatLayer {
            axis: 1,
            input_shapes: vec![
                TensorShape::D2 { dim1: 2, dim2: 1 },
                TensorShape::D2 { dim1: 2, dim2: 2 },
            ],
            output_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
        };
        #[rustfmt::skip]
        let a = Tensor::new(vec![1.0, 4.0], TensorShape::D2 { dim1: 2, dim2: 1 });
        #[rustfmt::skip]
        let b = Tensor::new(vec![2.0, 3.0, 5.0, 6.0], TensorShape::D2 { dim1: 2, dim2: 2 });
        let output = layer.forward(&[&a, &b]);
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 2, dim2: 3 });
        // row 0: [1,2,3], row 1: [4,5,6]
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_forward_more_than_two_inputs() {
        let layer = ConcatLayer {
            axis: 0,
            input_shapes: vec![
                TensorShape::Flat(1),
                TensorShape::Flat(1),
                TensorShape::Flat(1),
            ],
            output_shape: TensorShape::Flat(3),
        };
        let a = Tensor::new(vec![1.0], TensorShape::Flat(1));
        let b = Tensor::new(vec![2.0], TensorShape::Flat(1));
        let c = Tensor::new(vec![3.0], TensorShape::Flat(1));
        let output = layer.forward(&[&a, &b, &c]);
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "expected 2 input tensor(s)")]
    fn test_forward_rejects_wrong_input_count() {
        let layer = ConcatLayer {
            axis: 0,
            input_shapes: vec![TensorShape::Flat(1), TensorShape::Flat(1)],
            output_shape: TensorShape::Flat(2),
        };
        let a = Tensor::new(vec![1.0], TensorShape::Flat(1));
        layer.forward(&[&a]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch in ConcatLayer input")]
    fn test_forward_rejects_mismatched_input_shape() {
        let layer = ConcatLayer {
            axis: 0,
            input_shapes: vec![TensorShape::Flat(2), TensorShape::Flat(2)],
            output_shape: TensorShape::Flat(4),
        };
        let a = Tensor::new(vec![1.0, 2.0], TensorShape::Flat(2));
        let wrong_b = Tensor::new(vec![1.0, 2.0, 3.0], TensorShape::Flat(3));
        layer.forward(&[&a, &wrong_b]);
    }
}