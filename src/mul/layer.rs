use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

/// Elementwise multiplication - the Rust-side counterpart of ONNX's `Mul`,
/// mirroring `AddLayer`: every tensor input (all sharing `shape`) and every
/// baked-in constant are multiplied together. A constant is either
/// `shape.total_size()` long (already broadcast by the exporter) or a
/// single scalar. The most common use is a frozen BatchNorm's `x * scale`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MulLayer {
    pub shape: TensorShape,
    pub constants: Vec<Vec<f32>>,
}

impl MulLayer {
    pub fn input_shape(&self) -> TensorShape {
        self.shape.clone()
    }

    pub fn output_shape(&self) -> TensorShape {
        self.shape.clone()
    }

    pub fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        assert!(
            !inputs.is_empty(),
            "MulLayer requires at least one tensor input"
        );
        for input in inputs {
            assert_eq!(
                input.shape(),
                self.input_shape(),
                "Shape mismatch in MulLayer: expected {:?}, got {:?}",
                self.input_shape(),
                input.shape()
            );
        }

        let mut data = inputs[0].data.as_standard_layout().into_owned();

        for extra in &inputs[1..] {
            data.zip_mut_with(&extra.data, |d, e| *d *= e);
        }

        let total_size = self.shape.total_size();
        for constant in &self.constants {
            if constant.len() == total_size {
                for (d, c) in data.iter_mut().zip(constant.iter()) {
                    *d *= c;
                }
            } else if constant.len() == 1 {
                let c = constant[0];
                data.mapv_inplace(|v| v * c);
            } else {
                panic!(
                    "MulLayer constant length {} doesn't match tensor size {} and isn't a scalar",
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
    fn test_forward_elementwise_constant() {
        let layer = MulLayer {
            shape: TensorShape::Flat(3),
            constants: vec![vec![2.0, 3.0, 4.0]],
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], layer.input_shape());
        assert_eq!(layer.forward(&[&input]).to_vec(), vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_forward_scalar_constant() {
        let layer = MulLayer {
            shape: TensorShape::D2 { dim1: 1, dim2: 2 },
            constants: vec![vec![0.5]],
        };
        let input = Tensor::new(vec![4.0, 6.0], layer.input_shape());
        assert_eq!(layer.forward(&[&input]).to_vec(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_forward_two_tensor_inputs() {
        let layer = MulLayer {
            shape: TensorShape::Flat(2),
            constants: vec![],
        };
        let a = Tensor::new(vec![2.0, 3.0], layer.input_shape());
        let b = Tensor::new(vec![5.0, 7.0], layer.input_shape());
        assert_eq!(layer.forward(&[&a, &b]).to_vec(), vec![10.0, 21.0]);
    }

    #[test]
    #[should_panic(expected = "isn't a scalar")]
    fn test_forward_rejects_mismatched_constant() {
        let layer = MulLayer {
            shape: TensorShape::Flat(3),
            constants: vec![vec![1.0, 2.0]],
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], layer.input_shape());
        layer.forward(&[&input]);
    }
}
