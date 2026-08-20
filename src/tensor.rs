use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

/// Declarative shape used in model JSON (layer configs, model input/output).
/// This is intentionally separate from the runtime array shape carried by
/// `ndarray` - it's what lets `Model::validate_shapes` check a layer chain
/// without any data flowing through it, and what lets us tell a `Flat(4)`
/// apart from a `D2 { 2, 2 }` even though both hold 4 elements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorShape {
    Flat(usize),
    D2 { dim1: usize, dim2: usize },
    D3 { dim1: usize, dim2: usize, dim3: usize },
}

impl TensorShape {
    /// Total number of scalar elements this shape describes.
    pub fn total_size(&self) -> usize {
        match self {
            TensorShape::Flat(n) => *n,
            TensorShape::D2 { dim1, dim2 } => dim1 * dim2,
            TensorShape::D3 { dim1, dim2, dim3 } => dim1 * dim2 * dim3,
        }
    }

    /// The dimensions of this shape, in `ndarray` order (outermost first).
    pub fn dims(&self) -> Vec<usize> {
        match self {
            TensorShape::Flat(n) => vec![*n],
            TensorShape::D2 { dim1, dim2 } => vec![*dim1, *dim2],
            TensorShape::D3 { dim1, dim2, dim3 } => vec![*dim1, *dim2, *dim3],
        }
    }

    /// The `ndarray` dynamic-rank shape equivalent to this `TensorShape`.
    pub fn to_ixdyn(&self) -> IxDyn {
        IxDyn(&self.dims())
    }

    /// Reconstructs a `TensorShape` from a raw `ndarray` shape (e.g.
    /// `array.shape()`). Panics on ranks other than 1-3, since those are
    /// the only ranks this model format understands.
    pub fn from_dims(dims: &[usize]) -> TensorShape {
        match dims {
            [n] => TensorShape::Flat(*n),
            [dim1, dim2] => TensorShape::D2 { dim1: *dim1, dim2: *dim2 },
            [dim1, dim2, dim3] => TensorShape::D3 { dim1: *dim1, dim2: *dim2, dim3: *dim3 },
            _ => panic!("Unsupported tensor rank: {:?} (only 1-3 dims are supported)", dims),
        }
    }
}

/// A tensor backed by `ndarray`'s dynamic-rank array, which is what gives
/// layers access to `ndarray`'s math functions (`mapv`, `dot`, reshaping,
/// reductions, ...) and - when the `blas` feature is enabled - BLAS-backed
/// matrix multiplication.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: ArrayD<f32>,
}

impl Tensor {
    /// Builds a tensor from flat, row-major data and a declared shape.
    ///
    /// Panics if `data.len()` doesn't match `shape.total_size()`, mirroring
    /// the old `debug_assert_eq!` - except this check now also runs in
    /// release builds, since `ArrayD::from_shape_vec` validates it for us.
    pub fn new(data: Vec<f32>, shape: TensorShape) -> Self {
        let array = ArrayD::from_shape_vec(shape.to_ixdyn(), data)
            .expect("Tensor data/shape size mismatch");
        Tensor { data: array }
    }

    /// Wraps an existing `ndarray` array directly, skipping the
    /// flat-`Vec` round trip. Used by layers that compute their output as
    /// an `ndarray` array already (e.g. via `.dot()` or `.mapv()`).
    pub fn from_array(array: ArrayD<f32>) -> Self {
        Tensor { data: array }
    }

    /// The declarative `TensorShape` this tensor's data currently has.
    pub fn shape(&self) -> TensorShape {
        TensorShape::from_dims(self.data.shape())
    }

    /// Flat, row-major copy of the tensor's data. Mainly a convenience for tests and callers that just want a plain `Vec<f32>` 
    /// (e.g. to hand off to code outside this crate).
    #[allow(dead_code)]
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_total_size_matches_len() {
        let shape = TensorShape::Flat(4);
        assert_eq!(shape.total_size(), 4);
    }

    #[test]
    fn d2_total_size_multiplies_dims() {
        let shape = TensorShape::D2 { dim1: 3, dim2: 4 };
        assert_eq!(shape.total_size(), 12);
    }

    #[test]
    fn d3_total_size_multiplies_dims() {
        let shape = TensorShape::D3 { dim1: 3, dim2: 4, dim3: 5 };
        assert_eq!(shape.total_size(), 60);
    }

    #[test]
    fn tensor_new_builds_matching_ndarray_shape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], TensorShape::D2 { dim1: 2, dim2: 2 });
        assert_eq!(t.data.shape(), &[2, 2]);
        assert_eq!(t.shape(), TensorShape::D2 { dim1: 2, dim2: 2 });
    }

    #[test]
    #[should_panic(expected = "Tensor data/shape size mismatch")]
    fn tensor_new_panics_on_size_mismatch() {
        let _ = Tensor::new(vec![1.0, 2.0], TensorShape::Flat(3));
    }

    #[test]
    fn shape_round_trips_through_from_dims() {
        assert_eq!(TensorShape::from_dims(&[7]), TensorShape::Flat(7));
        assert_eq!(TensorShape::from_dims(&[2, 3]), TensorShape::D2 { dim1: 2, dim2: 3 });
        assert_eq!(
            TensorShape::from_dims(&[2, 3, 4]),
            TensorShape::D3 { dim1: 2, dim2: 3, dim3: 4 }
        );
    }
}
