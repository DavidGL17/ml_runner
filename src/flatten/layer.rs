use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FlattenLayer {
    pub shape: TensorShape,
}

impl FlattenLayer {
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
            input.shape(),
            self.input_shape(),
            "Shape mismatch in FlattenLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let flat = input
            .data
            .clone()
            .into_shape_with_order(self.output_shape().dims())
            .expect("FlattenLayer: total element count changed during reshape");

        Tensor::from_array(flat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_output_shape_from_d3() {
        let layer = FlattenLayer {
            shape: TensorShape::D3 {
                dim1: 3,
                dim2: 4,
                dim3: 5,
            },
        };

        assert_eq!(layer.output_shape(), TensorShape::Flat(60));
    }

    #[test]
    fn test_flatten_output_shape_from_flat() {
        let layer = FlattenLayer {
            shape: TensorShape::Flat(10),
        };

        assert_eq!(layer.output_shape(), TensorShape::Flat(10));
    }

    /// Since `Tensor::data` is already a flat `Vec<f32>` regardless of shape,
    /// flattening a D3 tensor should preserve element order exactly -
    /// only the shape metadata changes, not the underlying data.
    #[test]
    fn test_forward_preserves_data_order() {
        let layer = FlattenLayer {
            shape: TensorShape::D3 {
                dim1: 3,
                dim2: 2,
                dim3: 2,
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

        assert_eq!(output.shape(), TensorShape::Flat(12));
        assert_eq!(
            output.to_vec(),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );
    }

    /// Flattening an already-flat tensor is a no-op on the data.
    #[test]
    fn test_forward_on_already_flat_input() {
        let layer = FlattenLayer {
            shape: TensorShape::Flat(4),
        };

        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], layer.input_shape());
        let output = layer.forward(&input);

        assert_eq!(output.shape(), TensorShape::Flat(4));
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = FlattenLayer {
            shape: TensorShape::D3 {
                dim1: 3,
                dim2: 4,
                dim3: 5,
            },
        };

        let wrong_input = Tensor::new(vec![0.0; 10], TensorShape::Flat(10));
        layer.forward(&wrong_input);
    }
}
