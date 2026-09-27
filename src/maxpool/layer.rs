use crate::tensor::{Tensor, TensorShape};
use ndarray::{Array3, ArrayView3, Ix3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MaxPool2DLayer {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl MaxPool2DLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::D3 {
            dim1: self.channels,
            dim2: self.height,
            dim3: self.width,
        }
    }

    /// The shape this layer produces, given a matching input shape.
    /// Channel count is unchanged - pooling only downsamples spatially.
    pub fn output_shape(&self) -> TensorShape {
        TensorShape::D3 {
            dim1: self.channels,
            dim2: (self.height + 2 * self.padding - self.kernel_size) / self.stride + 1,
            dim3: (self.width + 2 * self.padding - self.kernel_size) / self.stride + 1,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in MaxPool2DLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let out_shape = self.output_shape();
        let (out_h, out_w) = match out_shape {
            TensorShape::D3 {
                dim2: height,
                dim3: width,
                ..
            } => (height, width),
            _ => unreachable!(),
        };

        let input_view: ArrayView3<f32> = input
            .data
            .view()
            .into_dimensionality::<Ix3>()
            .expect("MaxPool2DLayer input is not 3-D");

        let mut output = Array3::<f32>::zeros((self.channels, out_h, out_w));

        let k_i = self.kernel_size as isize;
        let pad = self.padding as isize;
        let stride = self.stride as isize;
        let in_h = self.height as isize;
        let in_w = self.width as isize;

        for c in 0..self.channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    // Unlike Conv2DLayer, out-of-bounds (padded) positions
                    // are simply skipped rather than treated as zero - a
                    // zero-padded max-pool would silently corrupt the max
                    // whenever every real value in the window is negative.
                    let mut max_val = f32::NEG_INFINITY;

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

                            let v = input_view[[c, ih as usize, iw as usize]];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                    }

                    assert!(
                        max_val.is_finite(),
                        "MaxPool2DLayer: window at (channel {}, {}, {}) contains no in-bounds elements",
                        c,
                        oh,
                        ow
                    );

                    output[[c, oh, ow]] = max_val;
                }
            }
        }

        Tensor::from_array(output.into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxpool2d_layer_output_shape() {
        let layer = MaxPool2DLayer {
            kernel_size: 2,
            stride: 2,
            padding: 0,
            channels: 3,
            height: 224,
            width: 224,
        };

        assert_eq!(
            layer.output_shape(),
            TensorShape::D3 {
                dim1: 3,
                dim2: 112,
                dim3: 112,
            }
        );
    }

    #[test]
    fn test_forward_basic_2x2_no_padding() {
        let layer = MaxPool2DLayer {
            kernel_size: 2,
            stride: 2,
            padding: 0,
            channels: 1,
            height: 4,
            width: 4,
        };

        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                1.0,  3.0,  2.0,  4.0,
                5.0,  6.0,  1.0,  0.0,
                9.0,  2.0,  8.0,  7.0,
                1.0,  1.0,  3.0,  4.0,
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
        // top-left window max(1,3,5,6)=6, top-right max(2,4,1,0)=4
        // bottom-left max(9,2,1,1)=9, bottom-right max(8,7,3,4)=8
        assert_eq!(output.to_vec(), vec![6.0, 4.0, 9.0, 8.0]);
    }

    #[test]
    fn test_forward_overlapping_windows_stride_less_than_kernel() {
        let layer = MaxPool2DLayer {
            kernel_size: 2,
            stride: 1,
            padding: 0,
            channels: 1,
            height: 3,
            width: 3,
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
        // window maxes: max(1,2,4,5)=5, max(2,3,5,6)=6, max(4,5,7,8)=8, max(5,6,8,9)=9
        assert_eq!(output.to_vec(), vec![5.0, 6.0, 8.0, 9.0]);
    }

    #[test]
    fn test_forward_with_padding_ignores_padded_cells() {
        // A single 3x3 window over a 2x2 input, padded by 1 on each side.
        // If padding were (incorrectly) treated as zero, a window of all
        // negative real values would report 0.0 as the max instead of the
        // true (negative) maximum.
        let layer = MaxPool2DLayer {
            kernel_size: 3,
            stride: 1,
            padding: 1,
            channels: 1,
            height: 2,
            width: 2,
        };

        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                -5.0, -2.0,
                -8.0, -1.0,
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
        // Every output window covers the whole 2x2 input (plus ignored
        // padding), so every position reports the same true max: -1.0.
        assert_eq!(output.to_vec(), vec![-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_forward_multi_channel_independent() {
        let layer = MaxPool2DLayer {
            kernel_size: 2,
            stride: 2,
            padding: 0,
            channels: 2,
            height: 2,
            width: 2,
        };

        #[rustfmt::skip]
        let input = Tensor::new(
            vec![
                // channel 0
                1.0, 2.0,
                3.0, 4.0,
                // channel 1
                40.0, 30.0,
                20.0, 10.0,
            ],
            layer.input_shape(),
        );

        let output = layer.forward(&input);

        assert_eq!(output.to_vec(), vec![4.0, 40.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_forward_rejects_wrong_input_shape() {
        let layer = MaxPool2DLayer {
            kernel_size: 2,
            stride: 2,
            padding: 0,
            channels: 3,
            height: 224,
            width: 224,
        };
        let wrong_input = Tensor::new(vec![0.0; 10], TensorShape::Flat(10));
        layer.forward(&wrong_input);
    }
}