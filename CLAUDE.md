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
  - `model.py` - Handles model-level export logic
  - `layer.py` - Handles layer-specific export logic for different layer types

## How to Build and Run

1. Export the onnx model to C:
   ```
   conda activate pytorch
   python main.py
   ```

2. Build the project using CMake:
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run the compiled executable:
   ```
   ./ml_model
   ```

## How to Develop

### Adding New Layer Types

1. Add new layer type to `layer.h` enum
2. Create corresponding header file in `include/layers/`
3. Implement the layer logic in `src/layers/`
4. Update `model.c` to handle the new layer type in `model_forward`

### Exporting ONNX Models

1. Place your ONNX model in the root directory
2. Update `main.py` with the model path and output path
3. Run `python main.py` to generate the C header file
4. The generated file will be placed in `src/model_weights.h`

## Key Files and Functions

- `model_init()` - Initializes model memory
- `model_forward()` - Runs forward pass through all layers
- `forward_dense()` - Implements dense layer forward pass
- `export_onnx()` - Main function to convert ONNX to C code