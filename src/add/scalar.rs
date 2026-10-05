//! Default forward-pass implementation for `AddLayer`, using plain
//! `ndarray` iteration. Compiled in whenever the `simd` feature is *not*
//! enabled - see `add/simd.rs` for the alternative. Input validation is
//! shared and lives in `AddLayer::validate` (`add/layer.rs`).

use super::AddLayer;
use crate::tensor::Tensor;

impl AddLayer {
    pub fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        self.validate(inputs);

        let mut data = inputs[0].data.clone();

        for extra in &inputs[1..] {
            for (d, e) in data.iter_mut().zip(extra.data.iter()) {
                *d += e;
            }
        }

        let total_size = self.shape.total_size();
        for constant in &self.constants {
            if constant.len() == total_size {
                for (d, c) in data.iter_mut().zip(constant.iter()) {
                    *d += c;
                }
            } else {
                // `validate` guarantees the only other case is a scalar.
                let c = constant[0];
                data.mapv_inplace(|v| v + c);
            }
        }

        Tensor::from_array(data)
    }
}