//! SIMD-accelerated forward-pass implementation for `MaxPool2DLayer`, via
//! the `wide` crate (portable SIMD chosen at compile time from the build's
//! target features, no runtime CPU detection). Only compiled in when the
//! `simd` Cargo feature is enabled. Input validation is shared and lives in
//! `MaxPool2DLayer::validate` (`maxpool/layer.rs`).
//!
//! Strategy (same as the width kernel in `conv/simd.rs`): vectorise across
//! *output width*. If padding is needed the input is copied once into a
//! buffer padded with `-inf` - not zero - so padded cells can never win a
//! max, matching the scalar backend's "skip out-of-bounds cells" rule
//! without any bounds checks in the hot loop. For each output row, 8
//! adjacent output pixels are computed at once: for each window tap
//! `(kh, kw)` we load the 8 input pixels that feed them (a contiguous load
//! when `stride == 1`, a strided gather otherwise) and take a lane-wise max.
//! Leftover columns (`out_w % 8`) use a scalar tail.
//!
//! Like the scalar backend, a window that contains no real input element
//! panics; here that is detected afterwards as a non-finite output.

use super::MaxPool2DLayer;
use crate::tensor::Tensor;
use std::borrow::Cow;
use wide::f32x8;

const LANES: usize = 8;

/// Returns the input padded to `(C, H + 2p, W + 2p)` with `-inf`; borrows
/// when `p == 0`.
fn pad_input(input: &[f32], c: usize, h: usize, w: usize, pad: usize) -> Cow<'_, [f32]> {
    if pad == 0 {
        return Cow::Borrowed(input);
    }
    let (ph, pw) = (h + 2 * pad, w + 2 * pad);
    let mut out = vec![f32::NEG_INFINITY; c * ph * pw];
    for ic in 0..c {
        for ih in 0..h {
            let src = &input[(ic * h + ih) * w..(ic * h + ih + 1) * w];
            let dst_start = (ic * ph + ih + pad) * pw + pad;
            out[dst_start..dst_start + w].copy_from_slice(src);
        }
    }
    Cow::Owned(out)
}

/// Max-pool over a pre-padded input. Layouts: `padded` is `(C, ph, pw)` and
/// `out` is `(C, out_h, out_w)`, both row-major.
#[allow(clippy::too_many_arguments)]
fn maxpool_forward(
    padded: &[f32],
    ph: usize,
    pw: usize,
    channels: usize,
    k: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(padded.len(), channels * ph * pw);
    debug_assert_eq!(out.len(), channels * out_h * out_w);

    let blocks = out_w / LANES;
    let tail_start = blocks * LANES;

    for c in 0..channels {
        for oh in 0..out_h {
            let row_out = &mut out[(c * out_h + oh) * out_w..(c * out_h + oh + 1) * out_w];

            // SIMD body: 8 output pixels per iteration.
            for blk in 0..blocks {
                let ow0 = blk * LANES;
                let mut acc = f32x8::splat(f32::NEG_INFINITY);

                for kh in 0..k {
                    let row_base = (c * ph + oh * stride + kh) * pw + ow0 * stride;
                    for kw in 0..k {
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
                        acc = acc.max(x);
                    }
                }

                row_out[ow0..ow0 + LANES].copy_from_slice(&acc.to_array());
            }

            // Scalar tail for output widths that aren't a multiple of LANES.
            for ow in tail_start..out_w {
                let mut m = f32::NEG_INFINITY;
                for kh in 0..k {
                    let row_base = (c * ph + oh * stride + kh) * pw + ow * stride;
                    for kw in 0..k {
                        let v = padded[row_base + kw];
                        if v > m {
                            m = v;
                        }
                    }
                }
                row_out[ow] = m;
            }
        }
    }
}

impl MaxPool2DLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.validate(input);

        let input_slice = input
            .data
            .as_slice()
            .expect("MaxPool2DLayer input must be a contiguous tensor");

        let (out_h, out_w) = self.output_hw();
        let padded = pad_input(input_slice, self.channels, self.height, self.width, self.padding);
        let ph = self.height + 2 * self.padding;
        let pw = self.width + 2 * self.padding;

        let mut output = vec![0.0f32; self.channels * out_h * out_w];
        maxpool_forward(
            &padded,
            ph,
            pw,
            self.channels,
            self.kernel_size,
            self.stride,
            out_h,
            out_w,
            &mut output,
        );

        // A window with no real element leaves -inf (all padding) in its
        // output; report it the way the scalar backend does.
        if let Some(idx) = output.iter().position(|v| !v.is_finite()) {
            let plane = out_h * out_w;
            let (c, rem) = (idx / plane, idx % plane);
            panic!(
                "MaxPool2DLayer: window at (channel {}, {}, {}) contains no in-bounds elements",
                c,
                rem / out_w,
                rem % out_w
            );
        }

        Tensor::new(output, self.output_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive reference, independent of the SIMD code path: skips
    /// out-of-bounds cells exactly like the scalar backend.
    fn reference(layer: &MaxPool2DLayer, input: &[f32]) -> Vec<f32> {
        let (out_h, out_w) = layer.output_hw();
        let mut out = vec![0.0; layer.channels * out_h * out_w];
        for c in 0..layer.channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut m = f32::NEG_INFINITY;
                    for kh in 0..layer.kernel_size {
                        for kw in 0..layer.kernel_size {
                            let ih = (oh * layer.stride + kh) as isize - layer.padding as isize;
                            let iw = (ow * layer.stride + kw) as isize - layer.padding as isize;
                            if ih < 0
                                || iw < 0
                                || ih >= layer.height as isize
                                || iw >= layer.width as isize
                            {
                                continue;
                            }
                            let v = input[(c * layer.height + ih as usize) * layer.width + iw as usize];
                            if v > m {
                                m = v;
                            }
                        }
                    }
                    out[(c * out_h + oh) * out_w + ow] = m;
                }
            }
        }
        out
    }

    /// Integer-valued data in [-15, 15], so negatives and ties are common
    /// and max comparisons are exact.
    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) >> 20;
                (v % 31) as f32 - 15.0
            })
            .collect()
    }

    fn check(k: usize, stride: usize, padding: usize, channels: usize, h: usize, w: usize) {
        let layer = MaxPool2DLayer {
            kernel_size: k,
            stride,
            padding,
            channels,
            height: h,
            width: w,
        };
        let data = pseudo(channels * h * w, 3);
        let expected = reference(&layer, &data);
        let got = layer.forward(&Tensor::new(data, layer.input_shape())).to_vec();
        assert_eq!(got, expected);
    }

    #[test]
    fn matches_reference_wide_output_with_tail() {
        // out_w = 19 -> two SIMD blocks + 3-wide scalar tail
        check(3, 1, 1, 3, 12, 19);
    }

    #[test]
    fn matches_reference_common_2x2_stride_2() {
        // out_w = 20 -> strided gather path, 2 blocks + 4 tail
        check(2, 2, 0, 4, 16, 40);
    }

    #[test]
    fn matches_reference_3x3_stride_2_padded() {
        // The typical ResNet-style stem pool.
        check(3, 2, 1, 2, 17, 35);
    }

    #[test]
    fn matches_reference_narrower_than_one_register() {
        // out_w = 3 -> everything goes through the scalar tail
        check(2, 2, 0, 2, 6, 6);
    }

    #[test]
    fn matches_reference_overlapping_windows() {
        check(3, 1, 0, 1, 10, 24);
    }

    #[test]
    fn padding_never_wins_over_negative_values() {
        // All real values negative: a zero-padded implementation would
        // report 0.0 at the borders.
        let layer = MaxPool2DLayer {
            kernel_size: 3,
            stride: 1,
            padding: 1,
            channels: 1,
            height: 3,
            width: 9, // out_w = 9 -> one SIMD block + tail
        };
        let data: Vec<f32> = (0..27).map(|i| -1.0 - i as f32).collect();
        let got = layer
            .forward(&Tensor::new(data.clone(), layer.input_shape()))
            .to_vec();
        assert_eq!(got, reference(&layer, &data));
        assert!(got.iter().all(|&v| v < 0.0));
    }
}