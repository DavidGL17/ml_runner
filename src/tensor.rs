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
    pub fn total_size(&self) -> usize {
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
        debug_assert_eq!(
            data.len(),
            shape.total_size(),
            "Tensor data/shape size mismatch"
        );
        Tensor { data, shape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_total_size_matches_len() {
        let shape = TensorShape::Flat(4);
        assert_eq!(shape.total_size(), 4);
    }

    #[test]
    fn chw_total_size_multiplies_dims() {
        let shape = TensorShape::CHW {
            channels: 3,
            height: 4,
            width: 5,
        };
        assert_eq!(shape.total_size(), 60);
    }
}
