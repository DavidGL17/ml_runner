//! Plain, dependency-free implementation of the dense-layer forward pass.
//!
//! No architecture-specific instructions, no external crates - this is the
//! backend that should always work, including on the most constrained
//! embedded targets.

/// Computes `out = weights * input + bias` for a fully connected layer.
///
/// `weights` is a flattened `output_size x input_size` row-major matrix.
///
/// This module is always compiled in (even when another backend is
/// selected) so it stays available as a reference implementation and keeps
/// its own tests running regardless of which feature flags are set - hence
/// the `allow(dead_code)` when it isn't the active backend.
#[allow(dead_code)]
pub fn dense_forward(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    input_size: usize,
    output_size: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), input_size);
    debug_assert_eq!(bias.len(), output_size);
    debug_assert_eq!(weights.len(), input_size * output_size);
    debug_assert_eq!(out.len(), output_size);

    for i in 0..output_size {
        let row = &weights[i * input_size..(i + 1) * input_size];
        let mut sum = 0.0f32;
        for j in 0..input_size {
            sum += input[j] * row[j];
        }
        out[i] = sum + bias[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_hand_computed_output() {
        let input = vec![1.0, 2.0];
        let weights = vec![0.5, 0.5];
        let bias = vec![0.1];
        let mut out = vec![0.0; 1];
        dense_forward(&input, &weights, &bias, 2, 1, &mut out);
        assert_eq!(out, vec![1.6]);
    }
}
