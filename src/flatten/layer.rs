use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Flatten {
    pub shape: TensorShape,
}

impl Flatten {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        self.shape.clone()
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        TensorShape::Flat(self.shape.total_size())
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape,
            self.input_shape(),
            "Shape mismatch in DenseLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape
        );

        let flatten_size = self.output_shape().total_size();

        let mut output = vec![0.0; flatten_size];

        for i in 0..flatten_size {
            output[i] = input.data[i];
        }

        Tensor::new(output, self.output_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_output_shape_from_chw() {
        let layer = Flatten {
            shape: TensorShape::CHW {
                channels: 3,
                height: 4,
                width: 5,
            },
        };

        assert_eq!(layer.output_shape(), TensorShape::Flat(60));
    }

    #[test]
    fn test_flatten_output_shape_from_flat() {
        let layer = Flatten {
            shape: TensorShape::Flat(10),
        };

        assert_eq!(layer.output_shape(), TensorShape::Flat(10));
    }

    /// Since `Tensor::data` is already a flat `Vec<f32>` regardless of shape,
    /// flattening a CHW tensor should preserve element order exactly -
    /// only the shape metadata changes, not the underlying data.
    #[test]
    fn test_forward_preserves_data_order() {
        let layer = Flatten {
            shape: TensorShape::CHW {
                channels: 3,
                height: 2,
                width: 2,
            },
        };

        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                // channel 0
                1.0, 2.0,
                3.0, 4.0,
                // channel 1
                5.0, 6.0,
                7.0, 8.0,
                // channel 2
                9.0, 10.0,
                11.0, 12.0,
            ],
            layer.input_shape(),
        );

        let output = layer.forward(&input);

        assert_eq!(output.shape, TensorShape::Flat(12));
        assert_eq!(
            output.data,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );
    }

    /// Flattening an already-flat tensor is a no-op on the data.
    #[test]
    fn test_forward_on_already_flat_input() {
        let layer = Flatten {
            shape: TensorShape::Flat(4),
        };

        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], layer.input_shape());
        let output = layer.forward(&input);

        assert_eq!(output.shape, TensorShape::Flat(4));
        assert_eq!(output.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = Flatten {
            shape: TensorShape::CHW {
                channels: 3,
                height: 4,
                width: 5,
            },
        };

        let wrong_input = Tensor::new(vec![0.0; 10], TensorShape::Flat(10));
        layer.forward(&wrong_input);
    }
}
