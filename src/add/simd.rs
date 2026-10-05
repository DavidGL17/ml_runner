//! SIMD-accelerated forward-pass implementation for `AddLayer`, via the
//! `wide` crate (portable SIMD chosen at compile time from the build's
//! target features, no runtime CPU detection). Only compiled in when the
//! `simd` Cargo feature is enabled. Input validation is shared and lives in
//! `AddLayer::validate` (`add/layer.rs`).
//!
//! Addition is elementwise, so the kernels just walk the flat buffer 8
//! floats at a time with a scalar tail for lengths that aren't a multiple
//! of the lane count. Each tensor input and each constant is one pass over
//! the buffer.

use super::AddLayer;
use crate::tensor::Tensor;
use wide::f32x8;

const LANES: usize = 8;

/// `dst[i] += src[i]` for all `i`.
fn add_assign(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());

    let chunks = dst.len() / LANES;
    for c in 0..chunks {
        let base = c * LANES;
        let a = f32x8::from(<[f32; LANES]>::try_from(&dst[base..base + LANES]).unwrap());
        let b = f32x8::from(<[f32; LANES]>::try_from(&src[base..base + LANES]).unwrap());
        dst[base..base + LANES].copy_from_slice(&(a + b).to_array());
    }
    for j in chunks * LANES..dst.len() {
        dst[j] += src[j];
    }
}

/// `dst[i] += value` for all `i`.
fn add_scalar(dst: &mut [f32], value: f32) {
    let splat = f32x8::splat(value);
    let chunks = dst.len() / LANES;
    for c in 0..chunks {
        let base = c * LANES;
        let a = f32x8::from(<[f32; LANES]>::try_from(&dst[base..base + LANES]).unwrap());
        dst[base..base + LANES].copy_from_slice(&(a + splat).to_array());
    }
    for v in &mut dst[chunks * LANES..] {
        *v += value;
    }
}

fn contiguous(t: &Tensor) -> &[f32] {
    t.data
        .as_slice()
        .expect("AddLayer input must be a contiguous tensor")
}

impl AddLayer {
    pub fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        self.validate(inputs);

        let mut data = contiguous(inputs[0]).to_vec();

        for extra in &inputs[1..] {
            add_assign(&mut data, contiguous(extra));
        }

        let total_size = self.shape.total_size();
        for constant in &self.constants {
            if constant.len() == total_size {
                add_assign(&mut data, constant);
            } else {
                // `validate` guarantees the only other case is a scalar.
                add_scalar(&mut data, constant[0]);
            }
        }

        Tensor::new(data, self.output_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

    #[test]
    fn add_assign_covers_simd_chunks_and_tail() {
        // 19 = two 8-lane chunks + 3 scalar leftovers
        let mut dst: Vec<f32> = (0..19).map(|x| x as f32).collect();
        let src: Vec<f32> = (0..19).map(|x| 100.0 + x as f32).collect();
        let expected: Vec<f32> = dst.iter().zip(&src).map(|(a, b)| a + b).collect();
        add_assign(&mut dst, &src);
        assert_eq!(dst, expected);
    }

    #[test]
    fn add_assign_shorter_than_one_register() {
        let mut dst = vec![1.0, 2.0, 3.0];
        add_assign(&mut dst, &[10.0, 20.0, 30.0]);
        assert_eq!(dst, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_scalar_covers_simd_chunks_and_tail() {
        let mut dst: Vec<f32> = (0..11).map(|x| x as f32).collect();
        let expected: Vec<f32> = dst.iter().map(|x| x + 0.5).collect();
        add_scalar(&mut dst, 0.5);
        assert_eq!(dst, expected);
    }

    #[test]
    fn forward_wider_than_one_register_with_everything_combined() {
        // 3x7 = 21 elements: two SIMD chunks + 5-element tail.
        let shape = TensorShape::D2 { dim1: 3, dim2: 7 };
        let n = 21;
        let full: Vec<f32> = (0..n).map(|x| x as f32 * 0.25).collect();
        let layer = AddLayer {
            shape: shape.clone(),
            constants: vec![full.clone(), vec![3.0]],
        };
        let a_data: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|x| 2.0 * x as f32).collect();
        let a = Tensor::new(a_data.clone(), shape.clone());
        let b = Tensor::new(b_data.clone(), shape.clone());

        let output = layer.forward(&[&a, &b]);

        let expected: Vec<f32> = (0..n)
            .map(|i| a_data[i] + b_data[i] + full[i] + 3.0)
            .collect();
        assert_eq!(output.to_vec(), expected);
        assert_eq!(output.shape(), shape);
    }
}