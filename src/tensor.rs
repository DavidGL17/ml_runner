use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorShape {
    Flat(usize),
    D3 {
        dim1: usize,
        dim2: usize,
        dim3: usize,
    },
}

impl TensorShape {
    /// Total number of scalar elements this shape describes.
    pub fn total_size(&self) -> usize {
        match self {
            TensorShape::Flat(n) => *n,
            TensorShape::D3 {
                dim1,
                dim2,
                dim3,
            } => dim1 * dim2 * dim3,
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
    fn d3_total_size_multiplies_dims() {
        let shape = TensorShape::D3 {
            dim1: 3,
            dim2: 4,
            dim3: 5,
        };
        assert_eq!(shape.total_size(), 60);
    }
}
