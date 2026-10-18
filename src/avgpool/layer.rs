use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

/// 2D average pooling over a (C, H, W) tensor - the Rust-side counterpart
/// of ONNX's `AveragePool`. Same geometry fields as `MaxPool2DLayer`
/// (square kernel, one stride, symmetric zero padding, floor-mode output
/// size). `count_include_pad` picks the divisor for windows overlapping the
/// padding: the full `kernel_size^2` when true (PyTorch's default), or only
/// the in-bounds element count when false (ONNX's default).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AvgPool2DLayer {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    #[serde(default)]
    pub count_include_pad: bool,
}

impl AvgPool2DLayer {
    /// The shape this layer expects to receive.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::D3 {
            dim1: self.channels,
            dim2: self.height,
            dim3: self.width,
        }
    }

    /// Spatial output size `(out_h, out_w)`.
    fn output_hw(&self) -> (usize, usize) {
        (
            (self.height + 2 * self.padding - self.kernel_size) / self.stride + 1,
            (self.width + 2 * self.padding - self.kernel_size) / self.stride + 1,
        )
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        let (out_h, out_w) = self.output_hw();
        TensorShape::D3 {
            dim1: self.channels,
            dim2: out_h,
            dim3: out_w,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape(),
            self.input_shape(),
            "Shape mismatch in AvgPool2DLayer: expected {:?}, got {:?}",
            self.input_shape(),
            input.shape()
        );

        let (out_h, out_w) = self.output_hw();
        let data = input.data.as_standard_layout();
        let x = data.as_slice().expect("standard layout is contiguous");
        let (h, w, k, p) = (self.height, self.width, self.kernel_size, self.padding);
        let mut out = Vec::with_capacity(self.channels * out_h * out_w);

        for c in 0..self.channels {
            let plane = &x[c * h * w..(c + 1) * h * w];
            for oh in 0..out_h {
                for ow in 0..out_w {
                    // Window bounds in padded coordinates, clipped to the real input.
                    let h0 = oh * self.stride;
                    let w0 = ow * self.stride;
                    let (h_start, h_end) = (h0.max(p) - p, (h0 + k).min(h + p) - p);
                    let (w_start, w_end) = (w0.max(p) - p, (w0 + k).min(w + p) - p);

                    let mut sum = 0.0f32;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            sum += plane[ih * w + iw];
                        }
                    }
                    let count = if self.count_include_pad {
                        k * k
                    } else {
                        (h_end - h_start) * (w_end - w_start)
                    };
                    out.push(sum / count as f32);
                }
            }
        }

        Tensor::new(out, self.output_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(kernel_size: usize, stride: usize, padding: usize, count_include_pad: bool) -> AvgPool2DLayer {
        AvgPool2DLayer {
            kernel_size,
            stride,
            padding,
            channels: 1,
            height: 2,
            width: 2,
            count_include_pad,
        }
    }

    #[test]
    fn test_forward_simple() {
        let layer = layer(2, 2, 0, false);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 6.0], layer.input_shape());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), TensorShape::D3 { dim1: 1, dim2: 1, dim3: 1 });
        assert_eq!(output.to_vec(), vec![3.0]);
    }

    #[test]
    fn test_forward_kernel_one_is_identity() {
        let layer = layer(1, 1, 0, false);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], layer.input_shape());
        assert_eq!(layer.forward(&input).to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_forward_padding_divisor() {
        // Top-left window of a padded 2x2 input only covers input[0][0].
        let excluded = layer(2, 1, 1, false);
        let included = layer(2, 1, 1, true);
        let input = Tensor::new(vec![4.0, 0.0, 0.0, 0.0], excluded.input_shape());
        assert_eq!(excluded.forward(&input).to_vec()[0], 4.0);
        assert_eq!(included.forward(&input).to_vec()[0], 1.0);
    }
}
