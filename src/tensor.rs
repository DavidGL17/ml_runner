use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorShape {
    /// A plain 1-D vector of `n` values - used by `DenseLayer` and `ActivationLayer`.
    Flat(usize),
    /// Channels x height x width - used by convolutional layers.
    CHW {
        channels: usize,
        height: usize,
        width: usize,
    },
}

impl TensorShape {
    /// Total number of scalar elements this shape describes.
    pub fn numel(&self) -> usize {
        match self {
            TensorShape::Flat(n) => *n,
            TensorShape::CHW {
                channels,
                height,
                width,
            } => channels * height * width,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: TensorShape,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: TensorShape) -> Self {
        debug_assert_eq!(data.len(), shape.numel(), "Tensor data/shape size mismatch");
        Tensor { data, shape }
    }

    /// Convenience constructor for the common flat case.
    pub fn flat(data: Vec<f32>) -> Self {
        let n = data.len();
        Tensor::new(data, TensorShape::Flat(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_numel_matches_len() {
        let shape = TensorShape::Flat(4);
        assert_eq!(shape.numel(), 4);
    }

    #[test]
    fn chw_numel_multiplies_dims() {
        let shape = TensorShape::CHW {
            channels: 3,
            height: 4,
            width: 5,
        };
        assert_eq!(shape.numel(), 60);
    }

    #[test]
    fn flat_constructor_infers_shape() {
        let t = Tensor::flat(vec![1.0, 2.0, 3.0]);
        assert_eq!(t.shape, TensorShape::Flat(3));
    }
}
