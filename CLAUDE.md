# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Two packages in one repo that share a JSON model format:

- **`ml_runner_exporter/`** (Python, Poetry, published to PyPI as `ml-runner-exporter`): converts ONNX models into the JSON format.
- **`src/`** (Rust crate `ml_runner`, published to crates.io): loads that JSON and runs inference. It's meant to be device agnostic and to run on embedded targets, with backends picked at compile time by Cargo feature.

Only `f32` models are supported. Python and Rust are versioned and released separately: tags `python_X.Y.Z` publish to PyPI (the tag must match `poetry version`), and tags `rust_X.Y.Z` publish to crates.io.

## Commands

Rust:
```sh
cargo build                      # default (scalar/ndarray) backend
cargo build -F simd              # SIMD backend via the `wide` crate
cargo build -F blas              # needs system OpenBLAS + pkg-config (apt: libopenblas-dev pkg-config)
cargo test -F <default|simd|blas>  # CI runs each feature separately; `cargo test --all-features` also works
cargo test --test linear_model_tests run_dense_simple_model   # a single integration test
cargo test --lib model::tests::test_validate_shapes_ok        # a single unit test
cargo run -r -F simd -- <model.json> [iterations] [report.json]  # benchmark binary (src/main.rs)
```

Python:
```sh
poetry install
./lint.sh                        # ruff check + mypy (same checks as CI)
poetry run python main.py --model-path simple_linear_model.onnx --output-path export.json
poetry run python export_fixtures.py   # regenerates tests/fixtures/*.json
poetry run python benchmark.py             # PyTorch vs Rust (every feature), optionally on SSH devices; config in benchmark_config.json
```

Ruff runs with `select = ["ALL"]`, and mypy is strict (`disallow_untyped_defs`, etc.) on `ml_runner_exporter`. `export_fixtures.py`, `main.py`, `python_fixtures/` and notebooks are excluded from linting.

## Architecture

### JSON model format (the contract between both sides)
A model is `{inputs: [IoSpec], outputs: [IoSpec], nodes: [Node]}`. Each node has `id`, `inputs`/`outputs` (tensor names), and the layer fields flattened in, tagged by `"type"` (`"dense"`, `"conv2d"`, `"lstm"`, ...). On the Rust side, `TensorShape` is `Flat`, `D2` or `D3` (ranks 1 to 3 only). The batch dimension is dropped.

### Rust runtime
- `src/layers.rs`: the `define_layers!` macro generates the serde-tagged `Layer` enum along with its `input_shape`/`output_shape` dispatch. `forward` is written by hand because `Add` and `Concat` are the only layers that take more than one input. Every other layer reads `inputs[0]`.
- `src/model.rs`: `Model::forward` runs the nodes in order over a `HashMap<String, Tensor>` environment, so the nodes must already be topologically sorted. `validate_shapes` checks the graph using only the declared shapes. `allows_multiple_inputs` must stay in sync with `Layer::forward`.
- `src/tensor.rs`: `Tensor` wraps `ndarray::ArrayD<f32>`.
- Each layer has its own module directory: `layer.rs` holds the struct and its declared shapes. Compute-heavy layers (linear, conv, maxpool, rnn, add) also have `scalar.rs` (`#[cfg(not(feature = "simd"))]`, which uses ndarray and goes through BLAS when `blas` is on) and `simd.rs` (`#[cfg(feature = "simd")]`, which skips ndarray's dot/BLAS). Exactly one of the two gets compiled. Simple tensor ops only have `layer.rs`.
- `build.rs` exists only for `blas`. It re-emits the OpenBLAS linker flags directly onto the `ml_runner` bin target, because Cargo's propagation of those flags doesn't reach the bin's link step.

### Python exporter
- `onnx_exporter.export_onnx`: loads the model, runs ONNX shape inference, then repeats several constant-folding passes (`_resolve_constant_*`) until nothing changes. Folded nodes are skipped. Each remaining node is then parsed into a `LayerParser`.
- `node_dispatch.py`: maps ONNX op types to `*LayerParser` classes (`layers/` for the main layers, `layers/onnx/` for ONNX tensor ops). Each parser's `to_dict()` has to match the matching Rust struct's serde fields.
- `NodeContext.rust_dims` tracks the shape each tensor actually has on the Rust side, which differs from the ONNX shape (for example once the batch dimension is dropped or after recurrent outputs).

### Adding a layer end-to-end
1. Add a Python parser in `ml_runner_exporter/layers/` and register it in `node_dispatch.py`.
2. Add a Rust module (`layer.rs`, plus `scalar.rs`/`simd.rs` if it needs them). Add it to `mod` in `lib.rs` and as a `define_layers!` entry and `forward` arm in `layers.rs`.
3. Add a fixture model in `python_fixtures/` and an entry in the `fixtures` list in `export_fixtures.py`, then regenerate the fixtures.
4. Add an integration test in `tests/*_model_tests.rs` using `FixtureModelInput::load_json(...)` and `assert_abs_diff_eq!` (epsilon `1e-4`).
5. Update the support tables in README.md.

### Tests
The integration tests in `tests/` load `tests/fixtures/*.json`. Each fixture holds `{model, test_input, test_output}`, where the inputs and outputs are flat row-major arrays and the reference output comes from PyTorch or onnxruntime. The test harness (`tests/test_utils.rs`) currently only supports fixtures with a single input and a single output.
