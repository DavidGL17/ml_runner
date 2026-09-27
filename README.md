# ML_RUNNER

A small library that aims to provide a fast and device agnostic way of running python ML models in embedded platforms using rust


## Features

- A python library to parse ML models into an exported json format. Exports from : 
  - ONNX models
- A rust library to run the exported models on any platforms, with optimized functions depending on the platforms toolset

The library supports complex models with multiple inputs and outputs, and with multiple paths inside the model.

### Supported layers in python library

|   layer   | ONNX support |
| :-------: | :----------: |
|  Linear   |     yes      |
|  Conv2D   |     yes      |
|  Flatten  |     yes      |
|    RNN    |     yes      |
|    GRU    |     yes      |
|    Add    |     yes      |
|  Gather   |     yes      |
|   Shape   |     yes      |
| Transpose |     yes      |

#### Supported activation functions

| function | ONNX support |
| :------: | :----------: |
|   ReLU   |     yes      |
| Sigmoid  |     yes      |
|   Tanh   |     yes      |
| Softmax  |     yes      |
|  Linear  |     yes      |

### Supported layers in Rust library

|  layer  | simple forward | BLAS  | simd  | comment |
| :-----: | :------------: | :---: | :---: | :-----: |
| Linear  |      yes       |  yes  |  yes  |         |
| Conv2D  |      yes       |  yes  |  no   |         |
| Flatten |      yes       |  yes  |  no   |         |
|   RNN   |      yes       |  yes  |  no   |         |
|   GRU   |      yes       |  yes  |  no   |         |

#### Supported activation functions

| function | simple application | BLAS  | simd  |
| :------: | :----------------: | :---: | :---: |
|   ReLU   |        yes         |  yes  |  no   |
| Sigmoid  |        yes         |  yes  |  no   |
|   Tanh   |        yes         |  yes  |  no   |
| Softmax  |        yes         |  yes  |  no   |
|  Linear  |        yes         |  yes  |  no   |

## Usage

### Python library

To export a simple example onnx model, you can run the following : 

```python
poetry run python main.py --model-path simple_linear_model.onnx --output-path export.json
```

or from your own code

```python
from ml_runner_exporter.onnx_exporter import export_onnx

output_model = export_onnx(model_path)
```

You can then save the exported model as a json file

### Rust library

To run a model using the rust library, first export it to json using the python library as described in [the python library section](#python-library). 

You can then import it and run it using : 

```rust
let model = match Model::from_json(&model_json) {
    Ok(m) => m,
    Err(e) => {
        eprintln!("Error parsing model JSON: {}", e);
        return;
    }
};

// Shape the input to what the model requires
let input = vec![...];
println!("Input: {:?}", input);

// Run forward pass
match model.forward(&input) {
    Ok(output) => println!("Output: {:?}", output),
    Err(e) => eprintln!("Error during forward pass: {}", e),
}
```

See the [main.rs](src/main.rs) file for a concrete example.

To run the library, you can use one of the following targets : 

- `cargo run` to run the default target (no optimization)
- `cargo run --features simd` to run with SIMD optimizations (requires `wide` crate)
- `cargo run --features blas` to run with BLAS optimizations (requires specific software depending on your OS)
- To compile in release mode, use `cargo run -r` (add the feature you want after) 

#### Testing

To run the tests, you can use `cargo test --all-features` to run all tests

## Benchmarks 

Results obtained by running `poetry run python benchmark.py`

## Results for model : LeNet300100
|   Runner   |                Backend                | Runs  |    Min     |    Max     |    Mean    |   Median   |   StdDev   | Errors |
| :--------: | :-----------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  490  | 70.100 µs  | 375.200 µs | 95.331 µs  | 97.500 µs  | 26.817 µs  |   0    |
|    rust    |       ndarray (matrixmultiply)        |  490  | 48.000 µs  | 131.300 µs | 51.847 µs  | 48.300 µs  | 10.784 µs  |   0    |
|    rust    |              simd (wide)              |  490  | 36.800 µs  | 89.500 µs  | 39.707 µs  | 37.900 µs  |  6.905 µs  |   0    |
|    rust    |            ndarray (BLAS)             |  490  | 13.600 µs  | 899.501 µs | 18.540 µs  | 14.000 µs  | 43.353 µs  |   0    |
| python@pi4 | PyTorch 2.13.0+cu130 (CPU, 4 threads) |  490  |  2.622 ms  | 11.906 ms  |  2.921 ms  |  2.746 ms  | 859.167 µs |   0    |
|  rust@pi4  |       ndarray (matrixmultiply)        |  490  | 186.703 µs |  1.060 ms  | 201.565 µs | 191.934 µs | 45.391 µs  |   0    |
|  rust@pi4  |              simd (wide)              |  490  | 211.091 µs | 438.498 µs | 224.922 µs | 217.036 µs | 27.027 µs  |   0    |
|  rust@pi4  |            ndarray (BLAS)             |  490  | 138.055 µs | 366.184 µs | 158.050 µs | 154.120 µs | 24.055 µs  |   0    |


## Results for model : LeNet5
|   Runner   |                Backend                | Runs  |    Min     |    Max    |    Mean    |   Median   |   StdDev   | Errors |
| :--------: | :-----------------------------------: | :---: | :--------: | :-------: | :--------: | :--------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  490  | 625.891 µs | 1.695 ms  | 792.485 µs | 788.888 µs | 115.077 µs |   0    |
|    rust    |       ndarray (matrixmultiply)        |  490  |  1.065 ms  | 2.297 ms  |  1.151 ms  |  1.078 ms  | 256.801 µs |   0    |
|    rust    |              simd (wide)              |  490  |  1.041 ms  | 2.417 ms  |  1.124 ms  |  1.055 ms  | 231.703 µs |   0    |
|    rust    |            ndarray (BLAS)             |  490  |  1.871 ms  | 16.903 ms |  5.412 ms  |  4.983 ms  |  2.097 ms  |   0    |
| python@pi4 | PyTorch 2.13.0+cu130 (CPU, 4 threads) |  490  |  3.425 ms  | 26.702 ms |  4.175 ms  |  3.598 ms  |  2.371 ms  |   0    |
|  rust@pi4  |       ndarray (matrixmultiply)        |  490  |  3.871 ms  | 6.387 ms  |  3.994 ms  |  3.937 ms  | 239.704 µs |   0    |
|  rust@pi4  |              simd (wide)              |  490  |  3.874 ms  | 6.172 ms  |  3.967 ms  |  3.931 ms  | 139.602 µs |   0    |
|  rust@pi4  |            ndarray (BLAS)             |  490  |  3.854 ms  | 12.139 ms |  4.024 ms  |  4.004 ms  | 463.948 µs |   0    |

## Roadmap

These are the next features that I would like to implement, in no specific order

- [ ] Add activation functions support in simd
- [ ] Add support for more layer types
- [ ] Add support to other optimization backends (cuda, ...)
- [ ] Export both the python library and rust library to pip/c rates.io for easier usage
- [ ] Add support for non float models (right now only float is supported, we should allow export and run of models in int)
- [ ] Implement parser for pytorch and tensorflow models
  - I will focus mainly on tensorflow support at first, since pytorch has a good onnx exporter