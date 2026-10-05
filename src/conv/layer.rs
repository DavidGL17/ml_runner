//! The `Conv2DLayer` type itself: fields and declared shapes only. The
//! forward pass lives in a sibling module - `scalar.rs` by default, or
//! `simd.rs` with the `simd` Cargo feature - see `conv/mod.rs`.

use crate::tensor::TensorShape;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conv2DLayer {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub height: usize,
    pub width: usize,
    pub weights: Vec<f32>, // (C_out, C_in, k, k)
    pub bias: Vec<f32>,    // (C_out)
}

impl Conv2DLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::D3 {
            dim1: self.input_channels,
            dim2: self.height,
            dim3: self.width,
        }
    }

    /// Spatial output size `(out_h, out_w)`.
    pub(super) fn output_hw(&self) -> (usize, usize) {
        (
            (self.height + 2 * self.padding - self.kernel_size) / self.stride + 1,
            (self.width + 2 * self.padding - self.kernel_size) / self.stride + 1,
        )
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        let (out_h, out_w) = self.output_hw();
        TensorShape::D3 {
            dim1: self.output_channels,
            dim2: out_h,
            dim3: out_w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_layer_output_shape() {
        let layer = Conv2DLayer {
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_channels: 3,
            output_channels: 3,
            height: 224,
            width: 224,
            weights: vec![0.0; 3 * 3 * 3 * 3],
            bias: vec![0.0; 3],
        };

        assert_eq!(
            layer.output_shape(),
            TensorShape::D3 {
                dim1: 3,
                dim2: 224,
                dim3: 224,
            }
        );
    }
}