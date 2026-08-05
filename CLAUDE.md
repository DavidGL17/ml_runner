# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning model execution library that provides a way to run Python ML models in embedded platforms. The system consists of two main components:

1. **Python Library**: Parses ONNX models and exports them to a JSON format
2. **Rust Library**: Runs the exported models on any platform with optimized functions depending on the platform's toolset

## Architecture

The codebase is organized into two main components:

### Python Side
- `main.py`: Command-line interface for exporting ONNX models to JSON
- `c_exporter/`: Python package containing the ONNX exporter
  - `onnx_exporter.py`: Core logic to parse ONNX models and convert them to the internal JSON format
  - `layer.py`: Defines layer parsers (currently only supports Linear/Dense layers)
  - `model.py`: Defines the model structure for JSON serialization

### Rust Side
- `src/`: Main Rust library implementation
  - `main.rs`: Example executable demonstrating usage
  - `model.rs`: Model struct and forward pass implementation
  - `layers.rs`: Layer enum and implementations (currently only Dense layers)
  - `dense/`: Backend implementations for dense layer computation
    - `scalar.rs`: Portable scalar implementation (always compiled in)
    - `simd.rs`: SIMD-accelerated implementation using the `wide` crate (optional feature)
- `tests/`: Test fixtures and utilities
  - `linear_model_tests.rs`: Integration tests using fixture models
  - `test_utils.rs`: Test utilities for loading and validating fixtures

## Key Features

- **Cross-platform execution**: The Rust library can run on any platform with optimized backends
- **SIMD support**: Optional SIMD acceleration via the `wide` crate when the `simd` feature is enabled
- **ONNX model support**: Can export ONNX models to the internal JSON format
- **Layer support**: Currently supports only dense (linear) layers

## Commands for Development

### Building
- `cargo build` - Build the default version (scalar backend)
- `cargo build --features simd` - Build with SIMD optimizations

### Testing
- `cargo test --all-features` - Run all tests with all features enabled
- `cargo test` - Run tests with default features

### Running
- `cargo run` - Run the default example
- `cargo run --features simd` - Run with SIMD optimizations
- `cargo run path/to/export.json` - Run with a specific model file

### Exporting Models
- `python main.py --model-path simple_linear_model.onnx --output-path export.json` - Export an ONNX model to JSON format

## File Structure

- `main.py` - Python command-line interface
- `c_exporter/` - Python package for model export
- `src/` - Rust library source code
- `tests/fixtures/` - Test model data files used for validation
- `tests/` - Test code for the Rust library