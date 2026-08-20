//! The `DenseLayer` type itself: fields and declared shapes only. The
//! forward pass lives in a sibling module - `scalar.rs` by default, or
//! `simd.rs` with the `simd` Cargo feature - see `dense/mod.rs`.

use crate::tensor::TensorShape;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DenseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl DenseLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::Flat(self.input_size)
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        TensorShape::Flat(self.output_size)
    }
}