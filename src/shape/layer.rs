use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShapeLayer {
    pub input_shape: TensorShape,
    pub output_shape: TensorShape, // always Flat(rank of input_shape)
}

impl ShapeLayer {
    pub fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    pub fn output_shape(&self) -> TensorShape {
        self.output_shape.clone()
    }

    /// Returns a 1-D tensor whose values are the input's own dimensions,
    /// e.g. an input of shape D3{2,3,4} produces [2.0, 3.0, 4.0]. The
    /// input's *values* are irrelevant here - only its shape is read.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in ShapeLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let dims: Vec<f32> = input.data.shape().iter().map(|&d| d as f32).collect();

        let out_shape = self.output_shape();
        assert_eq!(
            dims.len(),
            out_shape.total_size(),
            "ShapeLayer: input rank {} doesn't match declared output shape {:?} (expects {} values)",
            dims.len(),
            out_shape,
            out_shape.total_size()
        );

        Tensor::new(dims, out_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_layer_shapes() {
        let layer = ShapeLayer {
            input_shape: TensorShape::D3 { dim1: 2, dim2: 3, dim3: 4 },
            output_shape: TensorShape::Flat(3),
        };
        assert_eq!(layer.input_shape(), TensorShape::D3 { dim1: 2, dim2: 3, dim3: 4 });
        assert_eq!(layer.output_shape(), TensorShape::Flat(3));
    }

    #[test]
    fn test_forward_reports_d3_dims() {
        let layer = ShapeLayer {
            input_shape: TensorShape::D3 { dim1: 2, dim2: 3, dim3: 4 },
            output_shape: TensorShape::Flat(3),
        };
        let input = Tensor::new(vec![0.0; 24], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::Flat(3));
        assert_eq!(output.to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_forward_reports_flat_dims() {
        let layer = ShapeLayer {
            input_shape: TensorShape::Flat(5),
            output_shape: TensorShape::Flat(1),
        };
        let input = Tensor::new(vec![9.0, 9.0, 9.0, 9.0, 9.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![5.0]);
    }

    #[test]
    fn test_forward_ignores_input_values() {
        // Shape only reports dimensions, never the data itself.
        let layer = ShapeLayer {
            input_shape: TensorShape::D2 { dim1: 2, dim2: 2 },
            output_shape: TensorShape::Flat(2),
        };
        let input = Tensor::new(vec![-1.0, 0.0, 1000.0, -1000.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![2.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = ShapeLayer {
            input_shape: TensorShape::Flat(3),
            output_shape: TensorShape::Flat(1),
        };
        let wrong_input = Tensor::new(vec![0.0, 0.0], TensorShape::Flat(2));
        layer.forward(&wrong_input);
    }
}