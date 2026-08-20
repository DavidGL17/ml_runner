//! The `DenseLayer` type (fields + declared shapes in `layer.rs`). Its
//! forward pass lives in exactly one sibling module, chosen at compile
//! time via this crate's Cargo features (see `Cargo.toml`):
//!
//!   - `scalar.rs` (default, i.e. whenever `simd` is *not* enabled): via
//!     `ndarray`'s `.dot()` - `ndarray`'s own portable `matrixmultiply`
//!     crate, or a linked system BLAS with the `blas` feature.
//!   - `simd.rs` (with the `simd` feature): a dependency-light backend
//!     using the `wide` crate, bypassing `ndarray`'s `.dot()`/BLAS
//!     entirely - no library needs linking, which is what makes this the
//!     option for embedded/microcontroller targets that support SIMD but
//!     have no BLAS port available.
//!
//! Note that `ndarray` itself is always a dependency of this crate
//! regardless of which backend is picked here - it's what `Tensor` is
//! built on (see `src/tensor.rs`), not just something this module opts
//! into.

mod layer;

#[cfg(not(feature = "simd"))]
mod scalar;

#[cfg(feature = "simd")]
mod simd;

pub use layer::DenseLayer;
