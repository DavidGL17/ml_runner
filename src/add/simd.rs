//! SIMD-accelerated forward-pass implementation for `AddLayer`, via the
//! `wide` crate (portable SIMD chosen at compile time from the build's
//! target features, no runtime CPU detection). Only compiled in when the
//! `simd` Cargo feature is enabled. Input validation is shared and lives in
//! `AddLayer::validate` (`add/layer.rs`).
//!
//! Addition is memory-bound, so the kernel is *fused*: it walks the buffer
//! once, and for each 8-float chunk loads that chunk from every operand
//! (the primary input, any extra tensor inputs, then the constants), sums
//! them in a register and stores the result once. That is one read per
//! operand plus a single write, instead of a full read-modify-write sweep
//! of the buffer per operand, and it needs no initial copy of the input.
//!
//! Operands are summed in a fixed order - primary input, extra tensor
//! inputs in order, then constants in order - the same order as the scalar
//! backend, so both produce bit-identical results.

use super::AddLayer;
use crate::tensor::Tensor;
use wide::f32x8;

const LANES: usize = 8;

/// One addend of the fused sum: either a full-size buffer or a scalar that
/// is broadcast across the whole tensor.
#[derive(Clone, Copy)]
enum Operand<'a> {
    Full(&'a [f32]),
    Scalar(f32),
}

#[inline(always)]
fn load(s: &[f32], base: usize) -> f32x8 {
    f32x8::from(<[f32; LANES]>::try_from(&s[base..base + LANES]).unwrap())
}

/// `out[i] = first[i] + operands[0][i] + operands[1][i] + ...`, accumulated
/// left to right, in a single pass over the buffers.
fn fused_add(first: &[f32], operands: &[Operand], out: &mut [f32]) {
    debug_assert_eq!(first.len(), out.len());
    debug_assert!(operands.iter().all(|op| match op {
        Operand::Full(s) => s.len() == first.len(),
        Operand::Scalar(_) => true,
    }));

    let chunks = first.len() / LANES;

    for c in 0..chunks {
        let base = c * LANES;
        let mut acc = load(first, base);
        for op in operands {
            acc += match op {
                Operand::Full(s) => load(s, base),
                Operand::Scalar(v) => f32x8::splat(*v),
            };
        }
        out[base..base + LANES].copy_from_slice(&acc.to_array());
    }

    // Scalar tail for lengths that aren't a multiple of LANES.
    for j in chunks * LANES..first.len() {
        let mut sum = first[j];
        for op in operands {
            sum += match op {
                Operand::Full(s) => s[j],
                Operand::Scalar(v) => *v,
            };
        }
        out[j] = sum;
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

        let total_size = self.shape.total_size();

        let mut operands: Vec<Operand> =
            Vec::with_capacity(inputs.len() - 1 + self.constants.len());
        operands.extend(inputs[1..].iter().map(|t| Operand::Full(contiguous(t))));
        operands.extend(self.constants.iter().map(|c| {
            if c.len() == total_size {
                Operand::Full(c)
            } else {
                // `validate` guarantees the only other case is a scalar.
                Operand::Scalar(c[0])
            }
        }));

        let mut out = vec![0.0f32; total_size];
        fused_add(contiguous(inputs[0]), &operands, &mut out);

        Tensor::new(out, self.output_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

    #[test]
    fn fused_add_covers_simd_chunks_and_tail() {
        // 19 = two 8-lane chunks + 3 scalar leftovers
        let first: Vec<f32> = (0..19).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..19).map(|x| 100.0 + x as f32).collect();
        let c: Vec<f32> = (0..19).map(|x| 0.5 * x as f32).collect();
        let expected: Vec<f32> = (0..19).map(|i| first[i] + b[i] + 7.0 + c[i]).collect();

        let mut out = vec![0.0; 19];
        fused_add(
            &first,
            &[Operand::Full(&b), Operand::Scalar(7.0), Operand::Full(&c)],
            &mut out,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn fused_add_shorter_than_one_register() {
        let mut out = vec![0.0; 3];
        fused_add(
            &[1.0, 2.0, 3.0],
            &[Operand::Full(&[10.0, 20.0, 30.0])],
            &mut out,
        );
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn fused_add_with_no_operands_copies_input() {
        let first: Vec<f32> = (0..11).map(|x| x as f32).collect();
        let mut out = vec![0.0; 11];
        fused_add(&first, &[], &mut out);
        assert_eq!(out, first);
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

    /// Operand order is part of the contract: summing out of order can
    /// change the result in the last bits, so this checks the fused kernel
    /// matches a strict left-to-right sum on values where order matters.
    #[test]
    fn forward_sums_in_scalar_backend_order() {
        let shape = TensorShape::Flat(9);
        let layer = AddLayer {
            shape: shape.clone(),
            constants: vec![vec![1.0e-8], vec![-1.0e8]],
        };
        let data = vec![1.0e8f32; 9];
        let extra = vec![1.0f32; 9];
        let a = Tensor::new(data.clone(), shape.clone());
        let b = Tensor::new(extra.clone(), shape.clone());

        let output = layer.forward(&[&a, &b]);

        let expected: Vec<f32> = (0..9)
            .map(|i| ((data[i] + extra[i]) + 1.0e-8) + -1.0e8)
            .collect();
        assert_eq!(output.to_vec(), expected);
    }
}