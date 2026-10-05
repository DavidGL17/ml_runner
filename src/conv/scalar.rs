//! Default forward-pass implementation for `Conv2DLayer`: a straightforward
//! direct convolution over `ndarray` views. Compiled in whenever the `simd`
//! feature is *not* enabled - see `conv/simd.rs` for the alternative.

use super::Conv2DLayer;
use crate::tensor::Tensor;
use ndarray::{Array3, ArrayView1, ArrayView3, ArrayView4, Ix3};

impl Conv2DLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in Conv2DLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let (out_h, out_w) = self.output_hw();

        let k = self.kernel_size;
        let weights = ArrayView4::from_shape(
            (self.output_channels, self.input_channels, k, k),
            &self.weights,
        )
        .expect(
            "Conv2DLayer weights length doesn't match output_channels * input_channels * kernel_size^2",
        );
        let bias = ArrayView1::from(&self.bias);
        let input_view: ArrayView3<f32> = input
            .data
            .view()
            .into_dimensionality::<Ix3>()
            .expect("Conv2DLayer input is not 3-D");

        let mut output = Array3::<f32>::zeros((self.output_channels, out_h, out_w));

        let k_i = k as isize;
        let pad = self.padding as isize;
        let stride = self.stride as isize;
        let in_h = self.height as isize;
        let in_w = self.width as isize;

        for oc in 0..self.output_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias[oc];

                    for ic in 0..self.input_channels {
                        for kh in 0..k_i {
                            let ih = oh as isize * stride + kh - pad;
                            if ih < 0 || ih >= in_h {
                                continue;
                            }
                            for kw in 0..k_i {
                                let iw = ow as isize * stride + kw - pad;
                                if iw < 0 || iw >= in_w {
                                    continue;
                                }

                                acc += input_view[[ic, ih as usize, iw as usize]]
                                    * weights[[oc, ic, kh as usize, kw as usize]];
                            }
                        }
                    }

                    output[[oc, oh, ow]] = acc;
                }
            }
        }

        Tensor::from_array(output.into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

    #[test]
    fn test_forward_no_padding_sum_kernel() {
        let layer = Conv2DLayer {
            kernel_size: 2,
            stride: 1,
            padding: 0,
            input_channels: 1,
            output_channels: 1,
            height: 3,
            width: 3,
            weights: vec![1.0; 4],
            bias: vec![0.0],
        };

        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
            layer.input_shape(),
        );

        let output = layer.forward(&input);

        assert_eq!(
            output.shape(),
            TensorShape::D3 { dim1: 1, dim2: 2, dim3: 2 }
        );
        assert_eq!(output.to_vec(), vec![12.0, 16.0, 24.0, 28.0]);
    }

    #[test]
    fn test_forward_bias_is_added() {
        let layer = Conv2DLayer {
            kernel_size: 2,
            stride: 1,
            padding: 0,
            input_channels: 1,
            output_channels: 1,
            height: 3,
            width: 3,
            weights: vec![1.0; 4],
            bias: vec![10.0],
        };

        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
            layer.input_shape(),
        );

        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![22.0, 26.0, 34.0, 38.0]);
    }

    #[test]
    fn test_forward_with_padding() {
        let layer = Conv2DLayer {
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_channels: 1,
            output_channels: 1,
            height: 3,
            width: 3,
            weights: vec![1.0; 9],
            bias: vec![0.0],
        };

        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
            layer.input_shape(),
        );

        let output = layer.forward(&input);

        #[rustfmt::skip]
        let expected = vec![
            12.0, 21.0, 16.0,
            27.0, 45.0, 33.0,
            24.0, 39.0, 28.0,
        ];
        assert_eq!(output.to_vec(), expected);
    }

    #[test]
    fn test_forward_with_stride() {
        let layer = Conv2DLayer {
            kernel_size: 2,
            stride: 2,
            padding: 0,
            input_channels: 1,
            output_channels: 1,
            height: 4,
            width: 4,
            weights: vec![1.0; 4],
            bias: vec![0.0],
        };

        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                1.0,  2.0,  3.0,  4.0,
                5.0,  6.0,  7.0,  8.0,
                9.0,  10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            layer.input_shape(),
        );

        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![14.0, 22.0, 46.0, 54.0]);
    }

    #[test]
    fn test_forward_multi_input_channel() {
        let layer = Conv2DLayer {
            kernel_size: 1,
            stride: 1,
            padding: 0,
            input_channels: 2,
            output_channels: 1,
            height: 2,
            width: 2,
            weights: vec![2.0, 3.0],
            bias: vec![1.0],
        };

        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0,
            ],
            layer.input_shape(),
        );

        let output = layer.forward(&input);
        assert_eq!(output.to_vec(), vec![18.0, 23.0, 28.0, 33.0]);
    }

    #[test]
    fn test_forward_multi_output_channel() {
        let layer = Conv2DLayer {
            kernel_size: 1,
            stride: 1,
            padding: 0,
            input_channels: 1,
            output_channels: 2,
            height: 2,
            width: 2,
            weights: vec![5.0, 10.0],
            bias: vec![0.0, 100.0],
        };

        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], layer.input_shape());
        let output = layer.forward(&input);

        assert_eq!(
            output.to_vec(),
            vec![5.0, 10.0, 15.0, 20.0, 110.0, 120.0, 130.0, 140.0]
        );
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = Conv2DLayer {
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_channels: 3,
            output_channels: 3,
            height: 224,
            width: 224,
            weights: vec![0.0; 3 * 3 * 3 * 3],
            bias: vec![0.0; 3],
        };
        let wrong_input = Tensor::new(vec![0.0; 10], TensorShape::Flat(10));
        layer.forward(&wrong_input);
    }
}