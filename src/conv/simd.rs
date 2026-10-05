//! SIMD-accelerated forward-pass implementation for `Conv2DLayer`, via the
//! `wide` crate (portable SIMD chosen at compile time from the build's
//! target features, no runtime CPU detection). Only compiled in when the
//! `simd` Cargo feature is enabled. Like `dense/simd.rs`, it has no BLAS or
//! other native dependency.
//!
//! The input is zero-padded once up front, so the hot loops need no bounds
//! checks. Two kernels are provided and `run_conv` picks one by output width:
//!
//! - `conv_forward` (`out_w >= 8`): vectorise across *output width*. For
//!   every output row we compute 8 adjacent output pixels at once: for each
//!   tap `(ic, kh, kw)` we splat the scalar weight and multiply it with the 8
//!   input pixels that feed those outputs (a contiguous load when
//!   `stride == 1`, a strided gather otherwise). Leftover columns
//!   (`out_w % 8`) use a scalar tail.
//! - `conv_forward_dot` (`out_w < 8`): vectorise along the *reduction* axis.
//!   Each output pixel's receptive field is gathered into a patch of length
//!   `C_in * k * k` and dotted with each filter, as in `dense/simd.rs`. This
//!   keeps deep layers with small feature maps (7x7, 4x4, ...) on the SIMD
//!   path instead of the scalar tail.

use super::Conv2DLayer;
use crate::tensor::Tensor;
use std::borrow::Cow;
use wide::f32x8;

const LANES: usize = 8;

/// Returns the input zero-padded to `(C, H + 2p, W + 2p)`; borrows when `p == 0`.
fn pad_input(input: &[f32], c: usize, h: usize, w: usize, pad: usize) -> Cow<'_, [f32]> {
    if pad == 0 {
        return Cow::Borrowed(input);
    }
    let (ph, pw) = (h + 2 * pad, w + 2 * pad);
    let mut out = vec![0.0f32; c * ph * pw];
    for ic in 0..c {
        for ih in 0..h {
            let src = &input[(ic * h + ih) * w..(ic * h + ih + 1) * w];
            let dst_start = (ic * ph + ih + pad) * pw + pad;
            out[dst_start..dst_start + w].copy_from_slice(src);
        }
    }
    Cow::Owned(out)
}

/// Direct convolution on a pre-padded input. Layouts: `padded` is
/// `(C_in, ph, pw)`, `weights` is `(C_out, C_in, k, k)`, `out` is
/// `(C_out, out_h, out_w)`, all row-major.
#[allow(clippy::too_many_arguments)]
fn conv_forward(
    padded: &[f32],
    ph: usize,
    pw: usize,
    weights: &[f32],
    bias: &[f32],
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(padded.len(), in_c * ph * pw);
    debug_assert_eq!(weights.len(), out_c * in_c * k * k);
    debug_assert_eq!(bias.len(), out_c);
    debug_assert_eq!(out.len(), out_c * out_h * out_w);

    let blocks = out_w / LANES;
    let tail_start = blocks * LANES;

    for oc in 0..out_c {
        for oh in 0..out_h {
            let row_out = &mut out[(oc * out_h + oh) * out_w..(oc * out_h + oh + 1) * out_w];

            // SIMD body: 8 output pixels per iteration.
            for blk in 0..blocks {
                let ow0 = blk * LANES;
                let mut acc = f32x8::splat(bias[oc]);

                for ic in 0..in_c {
                    for kh in 0..k {
                        let row_base = (ic * ph + oh * stride + kh) * pw + ow0 * stride;
                        let w_base = ((oc * in_c + ic) * k + kh) * k;
                        for kw in 0..k {
                            let wv = f32x8::splat(weights[w_base + kw]);
                            let base = row_base + kw;
                            let x = if stride == 1 {
                                f32x8::from(
                                    <[f32; LANES]>::try_from(&padded[base..base + LANES]).unwrap(),
                                )
                            } else {
                                let mut lane = [0.0f32; LANES];
                                for (l, v) in lane.iter_mut().enumerate() {
                                    *v = padded[base + l * stride];
                                }
                                f32x8::from(lane)
                            };
                            acc += x * wv;
                        }
                    }
                }

                row_out[ow0..ow0 + LANES].copy_from_slice(&acc.to_array());
            }

            // Scalar tail for output widths that aren't a multiple of LANES.
            for ow in tail_start..out_w {
                let mut sum = bias[oc];
                for ic in 0..in_c {
                    for kh in 0..k {
                        let row_base = (ic * ph + oh * stride + kh) * pw + ow * stride;
                        let w_base = ((oc * in_c + ic) * k + kh) * k;
                        for kw in 0..k {
                            sum += padded[row_base + kw] * weights[w_base + kw];
                        }
                    }
                }
                row_out[ow] = sum;
            }
        }
    }
}

/// 8-lane dot product with a scalar tail (same scheme as `dense/simd.rs`).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let chunks = a.len() / LANES;
    let mut acc = f32x8::ZERO;
    for c in 0..chunks {
        let base = c * LANES;
        let x = f32x8::from(<[f32; LANES]>::try_from(&a[base..base + LANES]).unwrap());
        let y = f32x8::from(<[f32; LANES]>::try_from(&b[base..base + LANES]).unwrap());
        acc += x * y;
    }
    let mut sum = acc.reduce_add();
    for j in chunks * LANES..a.len() {
        sum += a[j] * b[j];
    }
    sum
}

/// Same contract as `conv_forward`, but vectorises along the reduction axis
/// instead of the output width, so it stays efficient when `out_w < LANES`
/// (deep layers with small feature maps).
///
/// For each output pixel we gather its receptive field once into a patch of
/// length `K = C_in * k * k` laid out `(ic, kh, kw)` - the same order as one
/// output channel's filter in `weights` - then take a dot product against
/// every output channel's filter. The gather copies `k` contiguous floats
/// per `(ic, kh)` row, and is amortised over all `out_c` filters.
#[allow(clippy::too_many_arguments)]
fn conv_forward_dot(
    padded: &[f32],
    ph: usize,
    pw: usize,
    weights: &[f32],
    bias: &[f32],
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(padded.len(), in_c * ph * pw);
    debug_assert_eq!(weights.len(), out_c * in_c * k * k);
    debug_assert_eq!(bias.len(), out_c);
    debug_assert_eq!(out.len(), out_c * out_h * out_w);

    let patch_len = in_c * k * k;
    let mut patch = vec![0.0f32; patch_len];

    for oh in 0..out_h {
        for ow in 0..out_w {
            // Gather the receptive field for this output pixel.
            let mut p = 0;
            for ic in 0..in_c {
                for kh in 0..k {
                    let src = (ic * ph + oh * stride + kh) * pw + ow * stride;
                    patch[p..p + k].copy_from_slice(&padded[src..src + k]);
                    p += k;
                }
            }

            for oc in 0..out_c {
                let filter = &weights[oc * patch_len..(oc + 1) * patch_len];
                out[(oc * out_h + oh) * out_w + ow] = dot(&patch, filter) + bias[oc];
            }
        }
    }
}

/// Picks the kernel for a given output width. The width-vectorised kernel
/// needs at least one full register of output pixels to do any SIMD work;
/// below that, vectorising along the reduction axis wins.
#[allow(clippy::too_many_arguments)]
fn run_conv(
    padded: &[f32],
    ph: usize,
    pw: usize,
    weights: &[f32],
    bias: &[f32],
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
    out: &mut [f32],
) {
    if out_w >= LANES {
        conv_forward(padded, ph, pw, weights, bias, in_c, out_c, k, stride, out_h, out_w, out);
    } else {
        conv_forward_dot(padded, ph, pw, weights, bias, in_c, out_c, k, stride, out_h, out_w, out);
    }
}

impl Conv2DLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in Conv2DLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );
        assert_eq!(
            self.weights.len(),
            self.output_channels * self.input_channels * self.kernel_size * self.kernel_size,
            "Conv2DLayer weights length doesn't match output_channels * input_channels * kernel_size^2"
        );

        let input_slice = input
            .data
            .as_slice()
            .expect("Conv2DLayer input must be a contiguous tensor");

        let (out_h, out_w) = self.output_hw();
        let padded = pad_input(
            input_slice,
            self.input_channels,
            self.height,
            self.width,
            self.padding,
        );
        let ph = self.height + 2 * self.padding;
        let pw = self.width + 2 * self.padding;

        let mut output = vec![0.0f32; self.output_channels * out_h * out_w];
        run_conv(
            &padded,
            ph,
            pw,
            &self.weights,
            &self.bias,
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            self.stride,
            out_h,
            out_w,
            &mut output,
        );

        Tensor::new(output, self.output_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

    /// Naive reference convolution, independent of the SIMD code path.
    fn reference(layer: &Conv2DLayer, input: &[f32]) -> Vec<f32> {
        let (out_h, out_w) = layer.output_hw();
        let k = layer.kernel_size;
        let mut out = vec![0.0; layer.output_channels * out_h * out_w];
        for oc in 0..layer.output_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = layer.bias[oc];
                    for ic in 0..layer.input_channels {
                        for kh in 0..k {
                            for kw in 0..k {
                                let ih = (oh * layer.stride + kh) as isize - layer.padding as isize;
                                let iw = (ow * layer.stride + kw) as isize - layer.padding as isize;
                                if ih < 0
                                    || iw < 0
                                    || ih >= layer.height as isize
                                    || iw >= layer.width as isize
                                {
                                    continue;
                                }
                                let x = input[(ic * layer.height + ih as usize) * layer.width
                                    + iw as usize];
                                let w = layer.weights
                                    [((oc * layer.input_channels + ic) * k + kh) * k + kw];
                                acc += x * w;
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
        // Small deterministic values so f32 results stay close to reference.
        (0..n)
            .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 20) % 17) as f32 / 8.0 - 1.0)
            .collect()
    }

    fn check(k: usize, stride: usize, padding: usize, in_c: usize, out_c: usize, h: usize, w: usize) {
        let layer = Conv2DLayer {
            kernel_size: k,
            stride,
            padding,
            input_channels: in_c,
            output_channels: out_c,
            height: h,
            width: w,
            weights: pseudo(out_c * in_c * k * k, 1),
            bias: pseudo(out_c, 2),
        };
        let data = pseudo(in_c * h * w, 3);
        let expected = reference(&layer, &data);
        let got = layer.forward(&Tensor::new(data, layer.input_shape())).to_vec();
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-3, "mismatch at {i}: {a} vs {b}");
        }
    }

    /// Runs one specific kernel (bypassing the width-based dispatch) and
    /// compares it with the reference, so both paths are tested on every shape.
    fn check_both_kernels(
        k: usize,
        stride: usize,
        padding: usize,
        in_c: usize,
        out_c: usize,
        h: usize,
        w: usize,
    ) {
        let layer = Conv2DLayer {
            kernel_size: k,
            stride,
            padding,
            input_channels: in_c,
            output_channels: out_c,
            height: h,
            width: w,
            weights: pseudo(out_c * in_c * k * k, 1),
            bias: pseudo(out_c, 2),
        };
        let data = pseudo(in_c * h * w, 3);
        let expected = reference(&layer, &data);
        let (out_h, out_w) = layer.output_hw();
        let padded = pad_input(&data, in_c, h, w, padding);
        let (ph, pw) = (h + 2 * padding, w + 2 * padding);

        type Kernel = fn(
            &[f32], usize, usize, &[f32], &[f32], usize, usize, usize, usize, usize, usize,
            &mut [f32],
        );
        for (name, kernel) in [
            ("width", conv_forward as Kernel),
            ("dot", conv_forward_dot as Kernel),
        ] {
            let mut got = vec![0.0; out_c * out_h * out_w];
            kernel(
                &padded, ph, pw, &layer.weights, &layer.bias, in_c, out_c, k, stride, out_h,
                out_w, &mut got,
            );
            for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
                assert!((a - b).abs() < 1e-3, "{name} kernel mismatch at {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn both_kernels_match_reference() {
        check_both_kernels(3, 1, 1, 3, 4, 12, 19); // wide, with tail
        check_both_kernels(3, 1, 0, 2, 2, 5, 5); // narrower than a register
        check_both_kernels(3, 2, 1, 3, 2, 17, 35); // strided
        check_both_kernels(1, 1, 0, 8, 4, 6, 16); // pointwise
        check_both_kernels(3, 1, 1, 16, 8, 7, 7); // deep layer: C_in*k*k = 144
        check_both_kernels(5, 1, 2, 1, 3, 9, 9); // K = 25, not a multiple of 8
    }

    #[test]
    fn dispatch_uses_dot_path_for_narrow_output() {
        // out_w = 7 < LANES: goes through the dot kernel via `forward`.
        check(3, 1, 1, 16, 8, 7, 7);
    }

    #[test]
    fn matches_reference_wide_output_with_tail() {
        // out_w = 19 -> two SIMD blocks + 3-wide scalar tail
        check(3, 1, 1, 3, 4, 12, 19);
    }

    #[test]
    fn matches_reference_narrower_than_one_register() {
        check(3, 1, 0, 2, 2, 5, 5);
    }

    #[test]
    fn matches_reference_with_stride() {
        check(3, 2, 1, 3, 2, 17, 35);
    }

    #[test]
    fn matches_reference_pointwise() {
        check(1, 1, 0, 8, 4, 6, 16);
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
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            layer.input_shape(),
        );
        #[rustfmt::skip]
        let expected = vec![
            12.0, 21.0, 16.0,
            27.0, 45.0, 33.0,
            24.0, 39.0, 28.0,
        ];
        assert_eq!(layer.forward(&input).to_vec(), expected);
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