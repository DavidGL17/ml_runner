//! Pluggable compute backends for layer math, plus the `DenseLayer` type
//! that uses them.
//!
//! Which backend is compiled into the binary is decided entirely at build
//! time via Cargo features (see `Cargo.toml`). There is no runtime
//! detection and no function-pointer indirection: the `#[cfg(...)]` blocks
//! below make sure exactly one implementation of `dense_forward` exists in
//! the final binary, so this is friendly to `no_std` / embedded targets
//! where you know the target hardware ahead of time.
//!
//! To add a new backend (e.g. a CMSIS-NN or DSP-specific path for a given
//! microcontroller):
//!   1. Add a new module here (e.g. `pub mod cmsis;`) gated behind a new
//!      Cargo feature.
//!   2. Implement a `dense_forward` function with the same signature as
//!      the ones in `scalar.rs` / `simd.rs`.
//!   3. Add a `#[cfg(feature = "...")]` arm in the re-export below,
//!      ordered by priority (most specific/fastest first).

pub mod scalar;

#[cfg(feature = "simd")]
pub mod simd;

mod layer;

pub use layer::DenseLayer;

// --- Compile-time backend selection -----------------------------------
//
// Exactly one of these `pub use` lines is active depending on which
// features are enabled. If more backends are added later, list them here
// with the highest-priority one first, guarded so only one wins.

#[cfg(feature = "simd")]
pub use simd::dense_forward;

#[cfg(not(feature = "simd"))]
pub use scalar::dense_forward;
