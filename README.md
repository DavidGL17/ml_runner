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
| Runner |                Backend                 | Runs  |    Min    |    Max     |    Mean    |  Median   |   StdDev   | Errors |
| :----: | :------------------------------------: | :---: | :-------: | :--------: | :--------: | :-------: | :--------: | :----: |
| python | PyTorch 2.13.0+cu130 (CPU, 12 threads) |  40   | 36.636 µs | 381.468 µs | 50.179 µs  | 37.975 µs | 55.316 µs  |   0    |
|  rust  |        ndarray (matrixmultiply)        |  40   | 37.259 µs | 46.816 µs  | 38.722 µs  | 38.312 µs |  1.782 µs  |   0    |
|  rust  |              simd (wide)               |  40   | 27.774 µs | 39.345 µs  | 28.731 µs  | 28.235 µs |  2.019 µs  |   0    |
|  rust  |             ndarray (BLAS)             |  40   | 21.612 µs |  3.616 ms  | 226.237 µs | 23.605 µs | 691.868 µs |   0    |


## Results for model : LeNet5
| Runner |                Backend                 | Runs  |    Min     |    Max    |   Mean   |  Median  |   StdDev   | Errors |
| :----: | :------------------------------------: | :---: | :--------: | :-------: | :------: | :------: | :--------: | :----: |
| python | PyTorch 2.13.0+cu130 (CPU, 12 threads) |  40   | 922.032 µs | 43.915 ms | 7.462 ms | 1.133 ms | 12.300 ms  |   0    |
|  rust  |        ndarray (matrixmultiply)        |  40   | 887.140 µs | 2.124 ms  | 1.384 ms | 1.064 ms | 510.285 µs |   0    |
|  rust  |              simd (wide)               |  40   |  1.023 ms  | 1.264 ms  | 1.072 ms | 1.072 ms | 41.746 µs  |   0    |
|  rust  |             ndarray (BLAS)             |  40   |  1.584 ms  | 5.323 ms  | 2.416 ms | 2.065 ms | 826.819 µs |   0    |


## Results for model : ResNet18Cifar
| Runner |                Backend                 | Runs  |     Min     |     Max     |    Mean     |   Median    |   StdDev   | Errors |
| :----: | :------------------------------------: | :---: | :---------: | :---------: | :---------: | :---------: | :--------: | :----: |
| python | PyTorch 2.13.0+cu130 (CPU, 12 threads) |  40   |  13.508 ms  | 159.277 ms  |  32.707 ms  |  23.098 ms  | 24.923 ms  |   0    |
|  rust  |        ndarray (matrixmultiply)        |  40   | 1102.530 ms | 1584.818 ms | 1257.013 ms | 1239.900 ms | 104.677 ms |   0    |
|  rust  |              simd (wide)               |  40   | 1221.842 ms | 1730.508 ms | 1398.595 ms | 1357.967 ms | 126.078 ms |   0    |
|  rust  |             ndarray (BLAS)             |  40   | 1172.925 ms | 1812.935 ms | 1313.896 ms | 1282.455 ms | 129.294 ms |   0    |


## Results for model : LSTMClassifier
| Runner |                Backend                 | Runs  |    Min     |    Max     |    Mean    |   Median   |  StdDev   | Errors |
| :----: | :------------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :-------: | :----: |
| python | PyTorch 2.13.0+cu130 (CPU, 12 threads) |  40   | 320.412 µs |  7.107 ms  | 869.421 µs | 354.719 µs | 1.462 ms  |   0    |
|  rust  |        ndarray (matrixmultiply)        |  40   | 716.150 µs | 797.537 µs | 725.945 µs | 722.394 µs | 14.291 µs |   0    |
|  rust  |              simd (wide)               |  40   | 694.504 µs | 862.991 µs | 729.058 µs | 716.798 µs | 33.735 µs |   0    |
|  rust  |             ndarray (BLAS)             |  40   |  1.352 ms  | 15.160 ms  |  2.578 ms  |  1.865 ms  | 2.296 ms  |   0    |


## Results for model : AlexNetLite
| Runner |                Backend                 | Runs  |    Min     |    Max     |    Mean    |   Median   |  StdDev   | Errors |
| :----: | :------------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :-------: | :----: |
| python | PyTorch 2.13.0+cu130 (CPU, 12 threads) |  40   |  1.378 ms  | 10.491 ms  |  2.677 ms  |  1.813 ms  | 2.019 ms  |   0    |
|  rust  |        ndarray (matrixmultiply)        |  40   | 69.351 ms  | 180.900 ms | 84.218 ms  | 79.618 ms  | 19.903 ms |   0    |
|  rust  |              simd (wide)               |  40   | 73.295 ms  | 88.671 ms  | 78.329 ms  | 77.850 ms  | 3.700 ms  |   0    |
|  rust  |             ndarray (BLAS)             |  40   | 172.171 ms | 228.794 ms | 189.522 ms | 185.047 ms | 15.158 ms |   0    |


## Roadmap

These are the next features that I would like to implement, in no specific order

- [ ] Add activation functions support in simd
- [ ] Add support for more layer types
- [ ] Add support to other optimization backends (cuda, ...)
- [ ] Export both the python library and rust library to pip/c rates.io for easier usage
- [ ] Add support for non float models (right now only float is supported, we should allow export and run of models in int)
- [ ] Implement parser for pytorch and tensorflow models
  - I will focus mainly on tensorflow support at first, since pytorch has a good onnx exporter