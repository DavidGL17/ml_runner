# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C library for running machine learning models with a focus on being fast and device-agnostic. It provides a framework for defining and executing neural network models in C, with support for different layer types including dense (fully connected) and convolutional 1D layers.

## Code Structure

### Core Components

- `include/` - Header files defining the API
  - `model.h` - Defines the Model structure and main API functions
  - `layer.h` - Defines layer types (Dense, Conv1D)
  - `layers/dense.h` - Defines DenseLayer structure and forward function

- `src/` - Source files implementing the functionality
  - `main.c` - Main entry point that initializes and runs the model
  - `model.c` - Implements model initialization, forward pass, and cleanup
  - `layers/dense.c` - Implements the forward pass for dense layers

- `c_exporter/` - Python tools for exporting ONNX models to C code
  - `onnx_exporter.py` - Main exporter that converts ONNX models to C headers
  - `model.py` - Handles model-level export logic and template wrapping
  - `layer.py` - Handles layer-specific export logic (Linear and Conv1D parsers)
  - `const.py` - Contains the C header templates

## How to Build and Run

1. Export the ONNX model to C:
   The `c_exporter` converts an ONNX model into a C header file (`src/model_weights.h`) containing the model's weights and layer definitions.
   ```bash
   conda activate pytorch
   python main.py
   ```

2. Build the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run the compiled executable:
   ```bash
   ./ml_model
   ```

## Python Exporter (c_exporter) Details

The `c_exporter` library is responsible for the transformation of an ONNX graph into a static C representation.

### Export Workflow
1. **Load & Verify**: `onnx_exporter.py` loads the `.onnx` file and performs an `onnx.checker.check_model` to ensure validity.
2. **Layer Parsing**: The script iterates through the ONNX nodes and identifies compatible layers:
   - **`Gemm` nodes** are parsed using `LinearLayerParser` to create `DenseLayer` definitions.
3. **Assembly**: `model.py` collects all parsed layer strings and wraps them with model-level constants (input/output sizes) and the `Model` structure initialization.
4. **Output**: The final output is a single `.h` file (typically `src/model_weights.h`) that uses include guards and includes the necessary layer headers.

### Key Classes
- `LayerParser` (ABC): The base class for all layer-specific parsing logic.
- `LinearLayerParser`: Generates `static const float` arrays for weights and biases, followed by a `DenseLayer` struct definition.

## C library details

### Adding New Layer Types

1. Add new layer type to `layer.h` enum
2. Create corresponding header file in `include/layers/`
3. Implement the layer logic in `src/layers/`
4. Update `model.c` to handle the new layer type in `model_forward`
5. Implement the extraction in the `c_exporter` library, so it exports the layer to the correct C representation based on what was done in the header file