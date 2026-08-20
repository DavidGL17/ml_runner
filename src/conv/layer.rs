use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conv2DLayer {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub height: usize,
    pub width: usize,
    pub weights: Vec<f32>, // (C_out, C_in, k, k)
    pub bias: Vec<f32>,    // (C_out)
}

impl Conv2DLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::D3 {
            dim1: self.input_channels,
            dim2: self.height,
            dim3: self.width,
        }
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        TensorShape::D3 {
            dim1: self.output_channels,
            dim2: (self.height + 2 * self.padding - self.kernel_size) / self.stride + 1,
            dim3: (self.width + 2 * self.padding - self.kernel_size) / self.stride + 1,
        }
    }

    /// Index into `weights` for (out_channel, in_channel, kh, kw).
    #[inline]
    fn weight_idx(&self, oc: usize, ic: usize, kh: usize, kw: usize) -> usize {
        let k = self.kernel_size;
        ((oc * self.input_channels + ic) * k + kh) * k + kw
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in Conv2DLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let out_shape = self.output_shape();
        let (out_h, out_w) = match out_shape {
            TensorShape::D3 { dim2: height, dim3: width, .. } => (height, width),
            _ => unreachable!(),
        };

        let mut out_data = vec![0.0f32; self.output_channels * out_h * out_w];

        let k = self.kernel_size as isize;
        let pad = self.padding as isize;
        let stride = self.stride as isize;
        let in_h = self.height as isize;
        let in_w = self.width as isize;

        for oc in 0..self.output_channels {
            let out_idx_c = oc * (out_h * out_w);
            for oh in 0..out_h {
                let out_idx_h = out_idx_c + oh * out_w;
                for ow in 0..out_w {
                    let mut acc = self.bias[oc];

                    for ic in 0..self.input_channels {
                        for kh in 0..k {
                            let ih = oh as isize * stride + kh - pad;
                            if ih < 0 || ih >= in_h {
                                continue;
                            }
                            for kw in 0..k {
                                let iw = ow as isize * stride + kw - pad;
                                if iw < 0 || iw >= in_w {
                                    continue;
                                }

                                let in_idx = ic * (self.height * self.width)
                                    + ih as usize * self.width
                                    + iw as usize;
                                let w_idx = self.weight_idx(oc, ic, kh as usize, kw as usize);

                                acc += input.data.as_slice().unwrap()[in_idx] * self.weights[w_idx];
                            }
                        }
                    }

                    let out_idx = out_idx_h + ow;
                    out_data[out_idx] = acc;
                }
            }
        }

        Tensor::new(out_data, out_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_layer_output_shape() {
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

        assert_eq!(
            layer.output_shape(),
            TensorShape::D3 {
                dim1: 3,
                dim2: 224,
                dim3: 224,
            }
        );
    }

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
            weights: vec![1.0; 4], // single 2x2 all-ones kernel
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
            TensorShape::D3 {
                dim1: 1,
                dim2: 2,
                dim3: 2,
            }
        );
        // window sums: [1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9]
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

        assert_eq!(
            output.shape(),
            TensorShape::D3 {
                dim1: 1,
                dim2: 3,
                dim3: 3,
            }
        );
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

        assert_eq!(
            output.shape(),
            TensorShape::D3 {
                dim1: 1,
                dim2: 2,
                dim3: 2,
            }
        );
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
            weights: vec![2.0, 3.0], // weight for input channel 0 = 2.0, channel 1 = 3.0
            bias: vec![1.0],
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
            ],
            layer.input_shape(),
        );

        let output = layer.forward(&input);

        // out[h][w] = bias + in0[h][w]*2 + in1[h][w]*3
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
            weights: vec![5.0, 10.0], // oc0 weight = 5.0, oc1 weight = 10.0
            bias: vec![0.0, 100.0],   // oc0 bias = 0, oc1 bias = 100
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

        assert_eq!(
            output.shape(),
            TensorShape::D3 {
                dim1: 2,
                dim2: 2,
                dim3: 2,
            }
        );
        // channel 0: input * 5
        // channel 1: input * 10 + 100
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
