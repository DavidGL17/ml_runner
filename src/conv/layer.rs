use crate::tensor::{Tensor, TensorShape};
use ndarray::Array4;
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
}

impl Conv2DLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::CHW {
            channels: self.input_channels,
            height: self.height,
            width: self.width,
        }
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        TensorShape::CHW {
            channels: self.output_channels,
            height: (self.height + 2 * self.padding - self.kernel_size) / self.stride + 1,
            width: (self.width + 2 * self.padding - self.kernel_size) / self.stride + 1,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape,
            self.input_shape(),
            "Shape mismatch in DenseLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape
        );

        Tensor::new(input.data.clone(), self.output_shape())
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
        };

        assert_eq!(
            layer.output_shape(),
            TensorShape::CHW {
                channels: 3,
                height: 224,
                width: 224,
            }
        );
    }
}
