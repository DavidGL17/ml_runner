//! Activation functions for neural network layers.
//!
//! This module provides various activation functions that can be applied to
//! layer outputs to introduce non-linearity into the model.

use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

/// Available activation functions.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum ActivationType {
    #[serde(rename = "relu")]
    ReLU,
    #[serde(rename = "sigmoid")]
    Sigmoid,
    #[serde(rename = "tanh")]
    Tanh,
    #[serde(rename = "softmax")]
    Softmax,
    #[serde(rename = "linear")]
    Linear,
}

impl ActivationType {
    /// Apply the activation function to a single value.
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationType::ReLU => x.max(0.0),
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationType::Tanh => x.tanh(),
            ActivationType::Linear => x,
            ActivationType::Softmax => {
                panic!("Softmax is not an element-wise operation; call apply_slice instead")
            }
        }
    }

    /// Apply the activation function to a slice of values.
    pub fn apply_slice(&self, x: &mut [f32]) {
        match self {
            ActivationType::Softmax => Self::softmax_in_place(x),
            _ => {
                for val in x.iter_mut() {
                    *val = self.apply(*val);
                }
            }
        }
    }

    /// Numerically stable softmax, applied in place.
    fn softmax_in_place(x: &mut [f32]) {
        if x.is_empty() {
            return;
        }

        // Subtract max for numerical stability (prevents overflow in exp()).
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for val in x.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }
        for val in x.iter_mut() {
            *val /= sum;
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ActivationLayer {
    pub activation_type: ActivationType,
    pub shape: TensorShape,
}

impl ActivationLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        self.shape.clone()
    }

    /// Activations are element-wise (or, for softmax, shape-preserving),
    /// so output shape always matches input shape.
    pub fn output_shape(&self) -> TensorShape {
        self.input_shape()
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape,
            self.input_shape(),
            "Shape mismatch in ActivationLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape
        );

        let mut data = input.data.clone();
        self.activation_type.apply_slice(&mut data);
        Tensor::new(data, self.output_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_activation() {
        let activation = ActivationType::ReLU;
        assert_eq!(activation.apply(-1.0), 0.0);
        assert_eq!(activation.apply(0.0), 0.0);
        assert_eq!(activation.apply(1.0), 1.0);
    }

    #[test]
    fn test_sigmoid_activation() {
        let activation = ActivationType::Sigmoid;
        assert_eq!(activation.apply(0.0), 0.5);
        assert!(activation.apply(10.0) > 0.999);
        assert!(activation.apply(-10.0) < 0.001);
    }

    #[test]
    fn test_tanh_activation() {
        let activation = ActivationType::Tanh;
        assert_eq!(activation.apply(0.0), 0.0);
        assert_eq!(activation.apply(10.0), 1.0);
        assert_eq!(activation.apply(-10.0), -1.0);
    }

    #[test]
    fn test_linear_activation() {
        let activation = ActivationType::Linear;
        assert_eq!(activation.apply(0.0), 0.0);
        assert_eq!(activation.apply(1.0), 1.0);
        assert_eq!(activation.apply(-1.0), -1.0);
    }

    #[test]
    fn test_apply_slice() {
        let activation = ActivationType::ReLU;
        let mut values = vec![-1.0, 0.0, 1.0, -2.0];
        activation.apply_slice(&mut values);
        assert_eq!(values, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_softmax_activation() {
        let activation = ActivationType::Softmax;
        let mut values = vec![1.0, 2.0, 3.0];
        activation.apply_slice(&mut values);

        // Should sum to 1.0
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should be monotonic: larger input -> larger output
        assert!(values[0] < values[1]);
        assert!(values[1] < values[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let activation = ActivationType::Softmax;
        let mut values = vec![1000.0, 1000.0, 1000.0];
        activation.apply_slice(&mut values);
        // Equal inputs -> equal outputs, no NaN/inf
        assert!((values[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic]
    fn test_softmax_apply_single_panics() {
        ActivationType::Softmax.apply(1.0);
    }

    #[test]
    fn test_activation_layer_forward() {
        let layer = ActivationLayer {
            activation_type: ActivationType::ReLU,
            shape: TensorShape::Flat(3),
        };
        let input = Tensor::new(vec![-1.0, 0.0, 1.0], TensorShape::Flat(3));
        let output = layer.forward(&input);
        // ReLU: [-1.0, 0.0, 1.0] -> [0.0, 0.0, 1.0]
        assert_eq!(output.data, vec![0.0, 0.0, 1.0]);
    }

    /// Same as above, but sitting between conv layers on D3 data - this is
    /// the case that motivated storing a full TensorShape instead of assuming
    /// Flat.
    #[test]
    fn test_activation_layer_forward_d3() {
        let layer = ActivationLayer {
            activation_type: ActivationType::ReLU,
            shape: TensorShape::D3 {
                dim1: 1,
                dim2: 2,
                dim3: 2,
            },
        };
        let input = Tensor::new(
            vec![-1.0, 0.0, 1.0, 2.0],
            TensorShape::D3 {
                dim1: 1,
                dim2: 2,
                dim3: 2,
            },
        );
        let output = layer.forward(&input);

        assert_eq!(
            output.shape,
            TensorShape::D3 {
                dim1: 1,
                dim2: 2,
                dim3: 2,
            }
        );
        assert_eq!(output.data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch in ActivationLayer")]
    fn test_activation_layer_wrong_shape() {
        let layer = ActivationLayer {
            activation_type: ActivationType::ReLU,
            shape: TensorShape::Flat(3),
        };
        let input = Tensor::new(vec![-1.0, 0.0], TensorShape::Flat(2));
        let _ = layer.forward(&input);
    }
}
