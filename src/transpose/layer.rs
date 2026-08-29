use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TransposeLayer {
    pub input_shape: TensorShape,
    pub perm: Vec<usize>, // e.g. [1, 0] to swap the two axes of a D2 tensor
}

impl TransposeLayer {
    pub fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    /// Permutes the input's dims according to `perm`.
    pub fn output_shape(&self) -> TensorShape {
        let in_dims = self.input_shape.dims();
        let out_dims: Vec<usize> = self.perm.iter().map(|&axis| in_dims[axis]).collect();
        TensorShape::from_dims(&out_dims)
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in TransposeLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        assert_eq!(
            self.perm.len(),
            input.data.ndim(),
            "TransposeLayer: perm length {} doesn't match input rank {}",
            self.perm.len(),
            input.data.ndim()
        );

        let permuted = input.data.clone().permuted_axes(self.perm.as_slice());
        // `permuted_axes` only changes the strides/view, not the underlying
        // memory layout, so force a standard-layout copy before handing the
        // data back out (same reasoning as GatherLayer's reshape step).
        let standardized = permuted.as_standard_layout().into_owned();

        Tensor::from_array(standardized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_layer_output_shape_d2() {
        let layer = TransposeLayer {
            input_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
            perm: vec![1, 0],
        };
        assert_eq!(layer.output_shape(), TensorShape::D2 { dim1: 3, dim2: 2 });
    }

    #[test]
    fn test_transpose_layer_output_shape_d3() {
        let layer = TransposeLayer {
            input_shape: TensorShape::D3 { dim1: 2, dim2: 3, dim3: 4 },
            perm: vec![2, 0, 1],
        };
        assert_eq!(layer.output_shape(), TensorShape::D3 { dim1: 4, dim2: 2, dim3: 3 });
    }

    #[test]
    fn test_forward_swap_d2_axes() {
        let layer = TransposeLayer {
            input_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
            perm: vec![1, 0],
        };
        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ],
            layer.input_shape(),
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 3, dim2: 2 });
        assert_eq!(output.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_forward_identity_perm_is_noop() {
        let layer = TransposeLayer {
            input_shape: TensorShape::D2 { dim1: 2, dim2: 2 },
            perm: vec![0, 1],
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_forward_d3_permutation() {
        let layer = TransposeLayer {
            input_shape: TensorShape::D3 { dim1: 2, dim2: 1, dim3: 2 },
            perm: vec![2, 0, 1],
        };
        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                1.0, 2.0,
                3.0, 4.0,
            ],
            layer.input_shape(),
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::D3 { dim1: 2, dim2: 2, dim3: 1 });
        // input[c,0,w] -> output[w,c,0]
        assert_eq!(output.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = TransposeLayer {
            input_shape: TensorShape::D2 { dim1: 2, dim2: 2 },
            perm: vec![1, 0],
        };
        let wrong_input = Tensor::new(vec![0.0, 0.0, 0.0], TensorShape::Flat(3));
        layer.forward(&wrong_input);
    }

    #[test]
    #[should_panic(expected = "perm length")]
    fn test_forward_rejects_mismatched_perm_length() {
        let layer = TransposeLayer {
            input_shape: TensorShape::D2 { dim1: 2, dim2: 2 },
            perm: vec![2, 1, 0],
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], layer.input_shape());
        layer.forward(&input);
    }
}