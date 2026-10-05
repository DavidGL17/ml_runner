//! The `AddLayer` type (fields, declared shapes and input validation in
//! `layer.rs`). Its forward pass lives in exactly one sibling module,
//! chosen at compile time via this crate's Cargo features:
//!
//!   - `scalar.rs` (default, i.e. whenever `simd` is *not* enabled): plain
//!     `ndarray` iteration.
//!   - `simd.rs` (with the `simd` feature): a `wide`-based backend that adds
//!     8 floats at a time, with no native library to link.

mod layer;

#[cfg(not(feature = "simd"))]
mod scalar;

#[cfg(feature = "simd")]
mod simd;

pub use layer::AddLayer;