use std::vec;

use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RNNLayer {
    
}

impl RNNLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::Flat(1)
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        TensorShape::Flat(1)
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in RNNLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        Tensor::new(vec![1.], self.output_shape())
    }
}