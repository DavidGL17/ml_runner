//! The `RNNLayer`, `GRULayer` and `LSTMLayer` types (fields, declared
//! shapes and validation in `layer.rs`). Their forward passes live in
//! exactly one sibling module, chosen at compile time via this crate's
//! Cargo features:
//!
//!   - `scalar.rs` (default, i.e. whenever `simd` is *not* enabled): via
//!     `ndarray`'s `.dot()` - `ndarray`'s own portable `matrixmultiply`
//!     crate, or a linked system BLAS with the `blas` feature.
//!   - `simd.rs` (with the `simd` feature): a `wide`-based backend with
//!     fused 8-lane gate matrix-vector products, bypassing `.dot()`/BLAS
//!     entirely - no native library to link.

mod layer;

#[cfg(not(feature = "simd"))]
mod scalar;

#[cfg(feature = "simd")]
mod simd;

pub use layer::{GRULayer, LSTMLayer, RNNLayer};