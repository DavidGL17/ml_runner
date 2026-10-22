//! Forward-pass implementation for `Conv2DLayer`: im2col followed by a
//! single matrix multiply. Used with every Cargo feature - see
//! `conv/mod.rs`.
//!
//! The input is unrolled into a `(C_in * k * k) x (out_h * out_w)` column
//! matrix (zeros where the kernel overlaps the padding), so the whole
//! convolution becomes `weights (C_out x C_in*k*k) . cols + bias`. That
//! product goes through `ndarray`'s `.dot()`: a BLAS `sgemm` with the
//! `blas` feature, or `matrixmultiply`'s blocked GEMM otherwise - the same
//! split as `linear/scalar.rs`. The column matrix costs `k * k` times the
//! input's memory; a 1x1, stride-1, unpadded conv skips it, since its
//! input already is that matrix.

use super::Conv2DLayer;
use crate::tensor::Tensor;
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3, Axis, CowArray, Ix2, Ix3};

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
        let patch = self.input_channels * k * k;

        let weights = ArrayView2::from_shape((self.output_channels, patch), &self.weights).expect(
            "Conv2DLayer weights length doesn't match output_channels * input_channels * kernel_size^2",
        );
        let bias = ArrayView1::from(&self.bias);
        let input_view: ArrayView3<f32> = input
            .data
            .view()
            .into_dimensionality::<Ix3>()
            .expect("Conv2DLayer input is not 3-D");
        let input_std = input_view.as_standard_layout();

        let cols: CowArray<f32, Ix2> = if k == 1 && self.stride == 1 && self.padding == 0 {
            input_std
                .view()
                .into_shape_with_order((self.input_channels, out_h * out_w))
                .expect("standard-layout input reshapes to (C_in, H * W)")
                .into()
        } else {
            self.im2col(input_std.as_slice().expect("standard layout is contiguous"), out_h, out_w)
                .into()
        };

        // (C_out x C_in*k*k) . (C_in*k*k x out_h*out_w), bias broadcast per row.
        let output = weights.dot(&cols) + &bias.insert_axis(Axis(1));

        let output = output
            .into_shape_with_order((self.output_channels, out_h, out_w))
            .expect("conv GEMM output reshapes to (C_out, out_h, out_w)");
        Tensor::from_array(output.into_dyn())
    }

    /// Unrolls a `(C_in, H, W)` input (row-major) into the column matrix:
    /// row `(ic * k + kh) * k + kw`, column `oh * out_w + ow` holds the
    /// input pixel that kernel tap multiplies for that output pixel.
    ///
    /// Every element is written exactly once, in order: for each tap the
    /// range of output columns that lands inside the input is worked out
    /// up front (`tap_range`), so only the borders are zero-filled and the
    /// in-bounds stretch is a straight copy when `stride == 1`.
    fn im2col(&self, input: &[f32], out_h: usize, out_w: usize) -> Array2<f32> {
        let k = self.kernel_size;
        let (stride, pad) = (self.stride, self.padding);
        let rows = self.input_channels * k * k;
        let n = out_h * out_w;

        let mut cols = Vec::with_capacity(rows * n);
        for ic in 0..self.input_channels {
            let plane = &input[ic * self.height * self.width..(ic + 1) * self.height * self.width];
            for kh in 0..k {
                let (oh_lo, oh_hi) = tap_range(kh, pad, stride, self.height, out_h);
                for kw in 0..k {
                    let (ow_lo, ow_hi) = tap_range(kw, pad, stride, self.width, out_w);

                    // Output rows whose input row is padding: all zeros.
                    cols.resize(cols.len() + oh_lo * out_w, 0.0);
                    for oh in oh_lo..oh_hi {
                        let ih = oh * stride + kh - pad;
                        let src_row = &plane[ih * self.width..(ih + 1) * self.width];
                        let iw_lo = ow_lo * stride + kw - pad;

                        cols.resize(cols.len() + ow_lo, 0.0);
                        if stride == 1 {
                            cols.extend_from_slice(&src_row[iw_lo..iw_lo + (ow_hi - ow_lo)]);
                        } else {
                            cols.extend(src_row[iw_lo..].iter().step_by(stride).take(ow_hi - ow_lo));
                        }
                        cols.resize(cols.len() + (out_w - ow_hi), 0.0);
                    }
                    cols.resize(cols.len() + (out_h - oh_hi) * out_w, 0.0);
                }
            }
        }
        Array2::from_shape_vec((rows, n), cols).expect("im2col fills exactly rows * n elements")
    }
}

/// For kernel tap `t` along one axis, the range `lo..hi` of output
/// positions `o` whose input index `o * stride + t - pad` falls inside
/// `0..in_len`. Empty (`lo == hi`) when the tap only ever reads padding.
fn tap_range(t: usize, pad: usize, stride: usize, in_len: usize, out_len: usize) -> (usize, usize) {
    // o * stride + t >= pad  <=>  o >= ceil((pad - t) / stride)
    let lo = pad.saturating_sub(t).div_ceil(stride).min(out_len);
    // o * stride + t < in_len + pad  <=>  o < ceil((in_len + pad - t) / stride)
    let hi = (in_len + pad).saturating_sub(t).div_ceil(stride).min(out_len);
    (lo, hi.max(lo))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

    /// Naive direct convolution (the previous implementation), as an
    /// independent reference for the im2col + GEMM path.
    fn reference(layer: &Conv2DLayer, input: &[f32]) -> Vec<f32> {
        let (out_h, out_w) = layer.output_hw();
        let k = layer.kernel_size as isize;
        let (pad, stride) = (layer.padding as isize, layer.stride as isize);
        let (h, w) = (layer.height as isize, layer.width as isize);
        let mut out = vec![0.0; layer.output_channels * out_h * out_w];
        for oc in 0..layer.output_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = layer.bias[oc];
                    for ic in 0..layer.input_channels {
                        for kh in 0..k {
                            for kw in 0..k {
                                let ih = oh as isize * stride + kh - pad;
                                let iw = ow as isize * stride + kw - pad;
                                if ih < 0 || ih >= h || iw < 0 || iw >= w {
                                    continue;
                                }
                                let x = input[(ic * layer.height + ih as usize) * layer.width + iw as usize];
                                let wi = ((oc * layer.input_channels + ic) * layer.kernel_size + kh as usize)
                                    * layer.kernel_size
                                    + kw as usize;
                                acc += x * layer.weights[wi];
                            }
                        }
                    }
                    out[(oc * out_h + oh) * out_w + ow] = acc;
                }
            }
        }
        out
    }

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_forward_matches_direct_reference() {
        // (k, stride, padding, in_c, out_c, h, w): includes the 1x1 fast
        // path, a 1x1 conv that can't take it (stride 2), non-square
        // inputs, and padding wider than the stride.
        let cases = [
            (1, 1, 0, 4, 3, 5, 6),
            (1, 2, 0, 3, 2, 6, 5),
            (3, 1, 1, 3, 4, 7, 9),
            (3, 2, 1, 2, 5, 8, 8),
            (5, 1, 2, 2, 3, 6, 4),
            (2, 2, 0, 3, 3, 9, 7),
            (3, 3, 2, 1, 2, 10, 11),
            (1, 1, 1, 2, 3, 4, 5),
            (2, 3, 1, 2, 2, 9, 10),
        ];
        for (seed, &(k, stride, padding, in_c, out_c, h, w)) in cases.iter().enumerate() {
            let layer = Conv2DLayer {
                kernel_size: k,
                stride,
                padding,
                input_channels: in_c,
                output_channels: out_c,
                height: h,
                width: w,
                weights: pseudo(out_c * in_c * k * k, seed as u32),
                bias: pseudo(out_c, seed as u32 + 100),
            };
            let input_data = pseudo(in_c * h * w, seed as u32 + 200);
            let got = layer.forward(&Tensor::new(input_data.clone(), layer.input_shape())).to_vec();
            let expected = reference(&layer, &input_data);
            assert_eq!(got.len(), expected.len());
            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert!((g - e).abs() <= 1e-4, "case {seed}, index {i}: got {g}, expected {e}");
            }
        }
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