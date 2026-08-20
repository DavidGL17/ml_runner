//! Only does anything when the `blas` feature is enabled. In that case,
//! probes for OpenBLAS via `pkg-config` and attaches the resulting linker
//! flags directly to this crate's `[[bin]]` target.
//!
//! Background: `openblas-src` (a dependency several hops down the graph,
//! pulled in by the `blas` feature - see Cargo.toml) already does this
//! same probe in its own build script, and normally that's sufficient on
//! its own via Cargo's standard `cargo:rustc-link-lib` propagation. But on
//! at least one observed setup (this package having both a `[lib]` target,
//! `src/lib.rs`, and a `[[bin]]` target, `src/main.rs`, sharing the same
//! package), that propagation reliably reaches the `[lib]` target's own
//! compile (its `rustc` invocation ends up with `-l openblas`) but *not*
//! the `[[bin]]` target's final link step - even when the exact same
//! directive is re-emitted from this crate's own build script via the
//! normal `cargo:rustc-link-lib`/`cargo:rustc-link-search` mechanism.
//! Confirmed via `cargo build --verbose`: `cblas_sgemv`/`cblas_sdot`
//! end up undefined at link time despite OpenBLAS being correctly
//! installed and correctly detected.
//!
//! `cargo:rustc-link-arg-bin=<BIN_NAME>=<FLAG>` sidesteps that propagation
//! entirely by attaching a raw linker argument directly to one named
//! binary target's link line, rather than relying on native-library
//! metadata flowing from a build script through the dependency graph to
//! wherever the final artifact gets linked.
//!
//! Build scripts don't see Cargo features via `#[cfg(feature = "...")]`
//! (only library/binary code does); they have to check the
//! `CARGO_FEATURE_<NAME>` environment variable instead.

fn main() {
    if std::env::var("CARGO_FEATURE_BLAS").is_err() {
        return;
    }

    let lib = pkg_config::probe_library("openblas")
        .expect("Could not locate OpenBLAS via pkg-config for the `blas` feature");

    // The binary target's name defaults to the package name ("ml_runner")
    // since Cargo.toml has no explicit [[bin]] section overriding it.
    for path in &lib.link_paths {
        println!("cargo:rustc-link-arg-bin=ml_runner=-L{}", path.display());
    }
    for name in &lib.libs {
        println!("cargo:rustc-link-arg-bin=ml_runner=-l{}", name);
    }
}