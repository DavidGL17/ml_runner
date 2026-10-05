//! The `AddLayer` type itself: fields, declared shapes and input validation.
//! The forward pass lives in a sibling module - `scalar.rs` by default, or
//! `simd.rs` with the `simd` Cargo feature - see `add/mod.rs`.

use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AddLayer {
    pub shape: TensorShape,
    pub constants: Vec<Vec<f32>>, // each entry is either shape.total_size() long, or a single scalar to broadcast
}

impl AddLayer {
    pub fn input_shape(&self) -> TensorShape {
        self.shape.clone()
    }

    pub fn output_shape(&self) -> TensorShape {
        self.shape.clone()
    }

    /// Checks that `inputs` is non-empty, that every tensor matches this
    /// layer's shape, and that every constant is either full-size or a
    /// scalar. Shared by both forward backends so they panic identically.
    pub(super) fn validate(&self, inputs: &[&Tensor]) {
        assert!(
            !inputs.is_empty(),
            "AddLayer requires at least one tensor input"
        );

        for input in inputs {
            assert_eq!(
                input.shape(),
                self.input_shape(),
                "Shape mismatch in AddLayer: expected {:?}, got {:?}",
                self.input_shape(),
                input.shape()
            );
        }

        let total_size = self.shape.total_size();
        for constant in &self.constants {
            if constant.len() != total_size && constant.len() != 1 {
                panic!(
                    "AddLayer constant length {} doesn't match tensor size {} and isn't a scalar",
                    constant.len(),
                    total_size
                );
            }
        }
    }
}

// `forward` is provided by whichever backend is compiled in, so these tests
// exercise the active backend: run them with and without `--features simd`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_layer_shapes_are_identity() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![vec![1.0, 2.0, 3.0]],
        };
        assert_eq!(layer.input_shape(), TensorShape::Flat(3));
        assert_eq!(layer.output_shape(), TensorShape::Flat(3));
    }

    #[test]
    fn test_forward_single_elementwise_constant() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![vec![10.0, 20.0, 30.0]],
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], TensorShape::Flat(3));
        let output = layer.forward(&[&input]);
        assert_eq!(output.to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_forward_scalar_broadcast_constant() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![vec![5.0]],
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], TensorShape::Flat(3));
        let output = layer.forward(&[&input]);
        assert_eq!(output.to_vec(), vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_forward_multiple_constants_are_all_applied() {
        let layer = AddLayer {
            shape: TensorShape::Flat(2),
            constants: vec![vec![1.0, 1.0], vec![100.0]],
        };
        let input = Tensor::new(vec![0.0, 0.0], TensorShape::Flat(2));
        let output = layer.forward(&[&input]);
        // input + [1,1] + broadcast(100) = [101, 101]
        assert_eq!(output.to_vec(), vec![101.0, 101.0]);
    }

    #[test]
    fn test_forward_no_constants_is_identity() {
        let layer = AddLayer {
            shape: TensorShape::Flat(2),
            constants: vec![],
        };
        let input = Tensor::new(vec![4.0, 5.0], TensorShape::Flat(2));
        let output = layer.forward(&[&input]);
        assert_eq!(output.to_vec(), vec![4.0, 5.0]);
    }

    #[test]
    fn test_forward_works_on_multi_dim_shape() {
        let layer = AddLayer {
            shape: TensorShape::D2 { dim1: 2, dim2: 2 },
            constants: vec![vec![1.0, 2.0, 3.0, 4.0]],
        };
        let input = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], layer.shape.clone());
        let output = layer.forward(&[&input]);
        assert_eq!(output.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// A residual/skip connection: two genuine computed tensors, no
    /// constants at all - e.g. `x + shortcut(x)` in a ResNet block.
    #[test]
    fn test_forward_two_tensor_inputs_residual_add() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![],
        };
        let main_branch = Tensor::new(vec![1.0, 2.0, 3.0], TensorShape::Flat(3));
        let shortcut = Tensor::new(vec![10.0, 20.0, 30.0], TensorShape::Flat(3));
        let output = layer.forward(&[&main_branch, &shortcut]);
        assert_eq!(output.to_vec(), vec![11.0, 22.0, 33.0]);
    }

    /// Two real tensor inputs *and* a constant - both code paths in the
    /// same forward call, to make sure they compose rather than being
    /// mutually exclusive.
    #[test]
    fn test_forward_two_tensor_inputs_plus_constant() {
        let layer = AddLayer {
            shape: TensorShape::Flat(2),
            constants: vec![vec![100.0]],
        };
        let a = Tensor::new(vec![1.0, 2.0], TensorShape::Flat(2));
        let b = Tensor::new(vec![10.0, 20.0], TensorShape::Flat(2));
        let output = layer.forward(&[&a, &b]);
        // (a + b) + broadcast(100) = [111, 122]
        assert_eq!(output.to_vec(), vec![111.0, 122.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![vec![1.0, 2.0, 3.0]],
        };
        let wrong_input = Tensor::new(vec![0.0, 0.0], TensorShape::Flat(2));
        layer.forward(&[&wrong_input]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_mismatched_second_tensor_shape() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![],
        };
        let a = Tensor::new(vec![1.0, 2.0, 3.0], TensorShape::Flat(3));
        let wrong_b = Tensor::new(vec![1.0, 2.0], TensorShape::Flat(2));
        layer.forward(&[&a, &wrong_b]);
    }

    #[test]
    #[should_panic(expected = "isn't a scalar")]
    fn test_forward_rejects_bad_constant_length() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![vec![1.0, 2.0]], // neither 3 nor 1
        };
        let input = Tensor::new(vec![0.0, 0.0, 0.0], TensorShape::Flat(3));
        layer.forward(&[&input]);
    }

    #[test]
    #[should_panic(expected = "at least one tensor input")]
    fn test_forward_rejects_empty_inputs() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![],
        };
        layer.forward(&[]);
    }
}