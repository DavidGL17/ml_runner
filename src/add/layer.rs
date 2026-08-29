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

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in AddLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let total_size = self.shape.total_size();
        let mut data = input.data.clone();

        for constant in &self.constants {
            if constant.len() == total_size {
                for (d, c) in data.iter_mut().zip(constant.iter()) {
                    *d += c;
                }
            } else if constant.len() == 1 {
                let c = constant[0];
                data.mapv_inplace(|v| v + c);
            } else {
                panic!(
                    "AddLayer constant length {} doesn't match tensor size {} and isn't a scalar",
                    constant.len(),
                    total_size
                );
            }
        }

        Tensor::from_array(data)
    }
}

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
        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_forward_scalar_broadcast_constant() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![vec![5.0]],
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], TensorShape::Flat(3));
        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_forward_multiple_constants_are_all_applied() {
        let layer = AddLayer {
            shape: TensorShape::Flat(2),
            constants: vec![vec![1.0, 1.0], vec![100.0]],
        };
        let input = Tensor::new(vec![0.0, 0.0], TensorShape::Flat(2));
        let output = layer.forward(&input);
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
        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![4.0, 5.0]);
    }

    #[test]
    fn test_forward_works_on_multi_dim_shape() {
        let layer = AddLayer {
            shape: TensorShape::D2 { dim1: 2, dim2: 2 },
            constants: vec![vec![1.0, 2.0, 3.0, 4.0]],
        };
        let input = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], layer.shape.clone());
        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![vec![1.0, 2.0, 3.0]],
        };
        let wrong_input = Tensor::new(vec![0.0, 0.0], TensorShape::Flat(2));
        layer.forward(&wrong_input);
    }

    #[test]
    #[should_panic(expected = "isn't a scalar")]
    fn test_forward_rejects_bad_constant_length() {
        let layer = AddLayer {
            shape: TensorShape::Flat(3),
            constants: vec![vec![1.0, 2.0]], // neither 3 nor 1
        };
        let input = Tensor::new(vec![0.0, 0.0, 0.0], TensorShape::Flat(3));
        layer.forward(&input);
    }
}