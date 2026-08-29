use crate::tensor::{Tensor, TensorShape};
use ndarray::Axis;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GatherLayer {
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
    pub axis: usize,
    pub indices: Option<Vec<f32>>, // None means indices are dynamic - unsupported at runtime
}

impl GatherLayer {
    pub fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    pub fn output_shape(&self) -> TensorShape {
        self.output_shape.clone()
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in GatherLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let indices = self.indices.as_ref().unwrap_or_else(|| {
            panic!(
                "GatherLayer has no constant indices: dynamic (runtime-computed) indices \
             aren't supported since forward() only takes a single input tensor"
            )
        });

        let idx: Vec<usize> = indices
            .iter()
            .map(|&x| {
                assert!(
                    x >= 0.0,
                    "GatherLayer: negative indices aren't supported, got {}",
                    x
                );
                x as usize
            })
            .collect();

        assert!(
            self.axis < input.data.ndim(),
            "GatherLayer: axis {} out of bounds for {}-D input",
            self.axis,
            input.data.ndim()
        );

        let gathered = input.data.select(Axis(self.axis), &idx);
        // `select` doesn't guarantee standard (row-major) layout except along
        // axis 0, so force it before reshaping or `into_shape_with_order` can
        // fail with IncompatibleLayout even though the element count matches.
        let gathered = gathered.as_standard_layout().into_owned();

        let out_shape = self.output_shape();
        assert_eq!(
            gathered.len(),
            out_shape.total_size(),
            "GatherLayer: gathered data has {} elements, but declared output shape {:?} expects {}",
            gathered.len(),
            out_shape,
            out_shape.total_size()
        );

        let reshaped = gathered
            .into_shape_with_order(out_shape.to_ixdyn())
            .expect("GatherLayer: gathered data doesn't match declared output shape");

        Tensor::from_array(reshaped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_layer_shapes() {
        let layer = GatherLayer {
            input_shape: TensorShape::D2 { dim1: 3, dim2: 4 },
            output_shape: TensorShape::Flat(4),
            axis: 0,
            indices: Some(vec![1.0]),
        };
        assert_eq!(layer.input_shape(), TensorShape::D2 { dim1: 3, dim2: 4 });
        assert_eq!(layer.output_shape(), TensorShape::Flat(4));
    }

    #[test]
    fn test_forward_gather_single_row_axis0() {
        // Simulates pulling out one row (e.g. Y_h from an RNN's stacked output).
        let layer = GatherLayer {
            input_shape: TensorShape::D2 { dim1: 3, dim2: 2 },
            output_shape: TensorShape::Flat(2),
            axis: 0,
            indices: Some(vec![1.0]),
        };
        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
            ],
            layer.input_shape(),
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::Flat(2));
        assert_eq!(output.to_vec(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_forward_gather_multiple_indices_axis0() {
        let layer = GatherLayer {
            input_shape: TensorShape::D2 { dim1: 3, dim2: 2 },
            output_shape: TensorShape::D2 { dim1: 2, dim2: 2 },
            axis: 0,
            indices: Some(vec![2.0, 0.0]),
        };
        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
            ],
            layer.input_shape(),
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 2, dim2: 2 });
        // row 2, then row 0
        assert_eq!(output.to_vec(), vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn test_forward_gather_along_axis1() {
        let layer = GatherLayer {
            input_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
            output_shape: TensorShape::D2 { dim1: 2, dim2: 2 },
            axis: 1,
            indices: Some(vec![0.0, 2.0]),
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
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 2, dim2: 2 });
        // columns 0 and 2 of each row
        assert_eq!(output.to_vec(), vec![1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = GatherLayer {
            input_shape: TensorShape::Flat(3),
            output_shape: TensorShape::Flat(1),
            axis: 0,
            indices: Some(vec![0.0]),
        };
        let wrong_input = Tensor::new(vec![0.0, 0.0], TensorShape::Flat(2));
        layer.forward(&wrong_input);
    }

    #[test]
    #[should_panic(expected = "dynamic")]
    fn test_forward_rejects_dynamic_indices() {
        let layer = GatherLayer {
            input_shape: TensorShape::Flat(3),
            output_shape: TensorShape::Flat(1),
            axis: 0,
            indices: None,
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], TensorShape::Flat(3));
        layer.forward(&input);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_forward_rejects_axis_out_of_bounds() {
        let layer = GatherLayer {
            input_shape: TensorShape::Flat(3),
            output_shape: TensorShape::Flat(1),
            axis: 5,
            indices: Some(vec![0.0]),
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], TensorShape::Flat(3));
        layer.forward(&input);
    }
}
