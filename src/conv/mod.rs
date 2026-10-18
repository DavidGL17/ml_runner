//! The `Conv2DLayer` type (fields + declared shapes in `layer.rs`). Unlike
//! the other compute-heavy layers, conv has a single forward pass for every
//! Cargo feature, in `gemm.rs`: im2col + one matrix multiply through
//! `ndarray`'s `.dot()` (BLAS `sgemm` with the `blas` feature, the pure-Rust,
//! SIMD-using `matrixmultiply` crate otherwise - so the `simd` feature still
//! needs nothing linked). It replaced a direct `wide`-based convolution that
//! benchmarked 1.5-7x slower on both x86_64 and a Raspberry Pi 4; that
//! kernel's trade-off was using no extra memory, whereas im2col needs a
//! column buffer `k * k` times the size of the input.

mod gemm;
mod layer;

pub use layer::Conv2DLayer;
