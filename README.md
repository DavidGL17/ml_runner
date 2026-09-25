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

## Results for model : HugeLinearModel
|   Runner   |                Backend                | Runs  |    Min    |    Max     |   Mean    |  Median   |   StdDev   | Errors |
| :--------: | :-----------------------------------: | :---: | :-------: | :--------: | :-------: | :-------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  490  | 2.360 ms  |  3.405 ms  | 2.704 ms  | 2.663 ms  | 178.340 µs |   0    |
|    rust    |       ndarray (matrixmultiply)        |  490  | 4.653 ms  | 12.572 ms  | 6.186 ms  | 6.011 ms  |  1.046 ms  |   0    |
|    rust    |              simd (wide)              |  490  | 4.191 ms  | 17.308 ms  | 5.581 ms  | 5.076 ms  |  1.502 ms  |   0    |
|    rust    |            ndarray (BLAS)             |  490  | 2.407 ms  | 121.661 ms | 29.574 ms | 17.558 ms | 25.830 ms  |   0    |
| python@pi4 | PyTorch 2.13.0+cu130 (CPU, 4 threads) |  490  | 59.463 ms | 127.800 ms | 68.915 ms | 68.743 ms |  7.792 ms  |   0    |
|  rust@pi4  |       ndarray (matrixmultiply)        |  490  | 16.096 ms | 25.735 ms  | 16.743 ms | 16.449 ms |  1.033 ms  |   0    |
|  rust@pi4  |              simd (wide)              |  490  | 16.256 ms | 22.659 ms  | 16.582 ms | 16.422 ms | 595.504 µs |   0    |
|  rust@pi4  |            ndarray (BLAS)             |  490  | 17.367 ms | 45.690 ms  | 18.031 ms | 17.723 ms |  1.954 ms  |   0    |


## Results for model : LongLinearModel
|   Runner   |                Backend                | Runs  |    Min    |    Max     |   Mean    |  Median   |   StdDev   | Errors |
| :--------: | :-----------------------------------: | :---: | :-------: | :--------: | :-------: | :-------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  490  | 1.906 ms  |  9.610 ms  | 3.367 ms  | 2.852 ms  |  1.385 ms  |   0    |
|    rust    |       ndarray (matrixmultiply)        |  490  | 2.981 ms  | 10.521 ms  | 4.174 ms  | 3.949 ms  | 944.390 µs |   0    |
|    rust    |              simd (wide)              |  490  | 2.636 ms  |  7.183 ms  | 3.851 ms  | 3.623 ms  | 888.879 µs |   0    |
|    rust    |            ndarray (BLAS)             |  490  | 1.469 ms  | 293.764 ms | 30.265 ms | 9.656 ms  | 41.311 ms  |   0    |
| python@pi4 | PyTorch 2.13.0+cu130 (CPU, 4 threads) |  490  | 61.014 ms | 128.781 ms | 64.296 ms | 62.297 ms |  7.781 ms  |   0    |
|  rust@pi4  |       ndarray (matrixmultiply)        |  490  | 9.697 ms  | 16.732 ms  | 10.136 ms | 9.912 ms  | 754.711 µs |   0    |
|  rust@pi4  |              simd (wide)              |  490  | 9.695 ms  | 17.008 ms  | 9.893 ms  | 9.812 ms  | 434.878 µs |   0    |
|  rust@pi4  |            ndarray (BLAS)             |  490  | 10.736 ms | 27.790 ms  | 11.364 ms | 11.015 ms |  1.517 ms  |   0    |

## Roadmap

These are the next features that I would like to implement, in no specific order

- [ ] Add activation functions support in simd
- [ ] Add support for more layer types
- [ ] Add support to other optimization backends (cuda, ...)
- [ ] Export both the python library and rust library to pip/c rates.io for easier usage
- [ ] Add support for non float models (right now only float is supported, we should allow export and run of models in int)
- [ ] Implement parser for pytorch and tensorflow models
  - I will focus mainly on tensorflow support at first, since pytorch has a good onnx exporter