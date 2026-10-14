//! Default forward-pass implementation for `MaxPool2DLayer`: a direct loop
//! over `ndarray` views. Compiled in whenever the `simd` feature is *not*
//! enabled - see `maxpool/simd.rs` for the alternative. Input validation is
//! shared and lives in `MaxPool2DLayer::validate` (`maxpool/layer.rs`).

use super::MaxPool2DLayer;
use crate::tensor::Tensor;
use ndarray::{Array3, ArrayView3, Ix3};

impl MaxPool2DLayer {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.validate(input);

        let (out_h, out_w) = self.output_hw();

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