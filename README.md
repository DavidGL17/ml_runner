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
|  MaxPool  |     yes      |
|    RNN    |     yes      |
|    GRU    |     yes      |
|   LSTM    |     yes      |
|    Add    |     yes      |
|  Concat   |     yes      |
|  Expand   |     yes      |
|  Gather   |     yes      |
| Identity  |     yes      |
|   Shape   |     yes      |
|  Squeeze  |     yes      |
| Transpose |     yes      |
| Unsqueeze |     yes      |

#### Supported activation functions

| function | ONNX support |
| :------: | :----------: |
|   ReLU   |     yes      |
| Sigmoid  |     yes      |
|   Tanh   |     yes      |
| Softmax  |     yes      |
|  Linear  |     yes      |

### Supported layers in Rust library

|   layer   | simple forward | BLAS  | simd  | comment |
| :-------: | :------------: | :---: | :---: | :-----: |
|  Linear   |      yes       |  yes  |  yes  |         |
|  Conv2D   |      yes       |  yes  |  no   |         |
|  Flatten  |      yes       |  yes  |  no   |         |
|  MaxPool  |      yes       |
|    RNN    |      yes       |  yes  |  no   |         |
|    GRU    |      yes       |  yes  |  no   |         |
|   LSTM    |      yes       |  yes  |  no   |         |
|    Add    |      yes       |  yes  |  no   |         |
|  Concat   |      yes       |  yes  |  no   |         |
|  Expand   |      yes       |  yes  |  no   |         |
|  Gather   |      yes       |  yes  |  no   |         |
| Identity  |      yes       |  yes  |  no   |         |
|   Shape   |      yes       |  yes  |  no   |         |
|  Squeeze  |      yes       |  yes  |  no   |         |
| Transpose |      yes       |  yes  |  no   |         |
| Unsqueeze |      yes       |  yes  |  no   |         |

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
|  Runner  |                Backend                | Runs  |    Min     |    Max     |    Mean    |   Median   |   StdDev   | Errors |
| :------: | :-----------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
|  python  | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  90   | 67.759 µs  | 110.158 µs | 82.651 µs  | 88.407 µs  | 11.613 µs  |   0    |
|   rust   |       ndarray (matrixmultiply)        |  90   | 49.853 µs  | 91.682 µs  | 51.261 µs  | 50.054 µs  |  5.480 µs  |   0    |
|   rust   |              simd (wide)              |  90   | 38.920 µs  | 68.812 µs  | 39.945 µs  | 39.121 µs  |  4.371 µs  |   0    |
|   rust   |            ndarray (BLAS)             |  90   |  7.924 µs  | 540.967 µs | 15.081 µs  |  8.727 µs  | 55.788 µs  |   0    |
|  python  |  PyTorch 2.13.0+cpu (CPU, 4 threads)  |  90   |  2.645 ms  |  5.756 ms  |  2.817 ms  |  2.742 ms  | 333.104 µs |   0    |
| rust@pi4 |       ndarray (matrixmultiply)        |  90   | 192.962 µs | 302.813 µs | 201.306 µs | 196.897 µs | 17.363 µs  |   0    |
| rust@pi4 |              simd (wide)              |  90   | 200.906 µs | 376.499 µs | 211.835 µs | 205.276 µs | 24.002 µs  |   0    |
| rust@pi4 |            ndarray (BLAS)             |  90   | 135.333 µs | 314.350 µs | 162.534 µs | 151.268 µs | 32.056 µs  |   0    |


## Results for model : LeNet5
|  Runner  |                Backend                | Runs  |    Min     |    Max     |    Mean    |   Median   |   StdDev   | Errors |
| :------: | :-----------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
|  python  | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  90   | 594.266 µs | 931.171 µs | 743.029 µs | 748.968 µs | 81.739 µs  |   0    |
|   rust   |       ndarray (matrixmultiply)        |  90   |  1.039 ms  |  1.279 ms  |  1.069 ms  |  1.056 ms  | 38.236 µs  |   0    |
|   rust   |              simd (wide)              |  90   |  1.036 ms  |  1.469 ms  |  1.056 ms  |  1.046 ms  | 47.232 µs  |   0    |
|   rust   |            ndarray (BLAS)             |  90   |  1.054 ms  |  8.295 ms  |  2.035 ms  |  1.890 ms  | 936.781 µs |   0    |
|  python  |  PyTorch 2.13.0+cpu (CPU, 4 threads)  |  90   |  3.341 ms  |  8.627 ms  |  3.725 ms  |  3.519 ms  | 782.934 µs |   0    |
| rust@pi4 |       ndarray (matrixmultiply)        |  90   |  3.421 ms  |  5.743 ms  |  3.517 ms  |  3.447 ms  | 311.022 µs |   0    |
| rust@pi4 |              simd (wide)              |  90   |  3.427 ms  |  3.636 ms  |  3.465 ms  |  3.443 ms  | 47.677 µs  |   0    |
| rust@pi4 |            ndarray (BLAS)             |  90   |  3.405 ms  |  5.606 ms  |  3.511 ms  |  3.420 ms  | 286.149 µs |   0    |


## Results for model : ResNet18Cifar
|  Runner  |                Backend                | Runs  |     Min     |     Max     |    Mean     |   Median    |   StdDev   | Errors |
| :------: | :-----------------------------------: | :---: | :---------: | :---------: | :---------: | :---------: | :--------: | :----: |
|  python  | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  90   |  9.954 ms   |  15.571 ms  |  12.397 ms  |  12.165 ms  | 941.963 µs |   0    |
|   rust   |       ndarray (matrixmultiply)        |  90   | 1694.931 ms | 1786.422 ms | 1739.714 ms | 1742.047 ms | 18.678 ms  |   0    |
|   rust   |              simd (wide)              |  90   | 1697.153 ms | 1899.089 ms | 1744.098 ms | 1744.596 ms | 27.978 ms  |   0    |
|   rust   |            ndarray (BLAS)             |  90   | 1690.060 ms | 1803.844 ms | 1740.977 ms | 1742.620 ms | 17.426 ms  |   0    |
|  python  |  PyTorch 2.13.0+cpu (CPU, 4 threads)  |  90   | 120.818 ms  | 194.788 ms  | 128.200 ms  | 125.639 ms  |  9.788 ms  |   0    |
| rust@pi4 |       ndarray (matrixmultiply)        |  90   | 5888.346 ms | 6050.638 ms | 5897.419 ms | 5892.463 ms | 22.466 ms  |   0    |
| rust@pi4 |              simd (wide)              |  90   | 5895.653 ms | 6440.316 ms | 6129.556 ms | 6122.039 ms | 160.488 ms |   0    |
| rust@pi4 |            ndarray (BLAS)             |  90   | 5875.337 ms | 5962.003 ms | 5891.817 ms | 5887.828 ms | 16.615 ms  |   0    |


## Results for model : LSTMClassifier
|  Runner  |                Backend                | Runs  |    Min     |    Max    |    Mean    |   Median   |   StdDev   | Errors |
| :------: | :-----------------------------------: | :---: | :--------: | :-------: | :--------: | :--------: | :--------: | :----: |
|  python  | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  90   | 449.515 µs | 1.929 ms  | 529.551 µs | 478.896 µs | 170.351 µs |   0    |
|   rust   |       ndarray (matrixmultiply)        |  90   |  1.146 ms  | 2.565 ms  |  1.230 ms  |  1.163 ms  | 271.257 µs |   0    |
|   rust   |              simd (wide)              |  90   |  1.145 ms  | 1.207 ms  |  1.162 ms  |  1.156 ms  | 16.614 µs  |   0    |
|   rust   |            ndarray (BLAS)             |  90   | 779.154 µs | 4.447 ms  | 973.017 µs | 827.234 µs | 525.149 µs |   0    |
|  python  |  PyTorch 2.13.0+cpu (CPU, 4 threads)  |  90   |  5.914 ms  | 7.095 ms  |  6.155 ms  |  6.133 ms  | 160.884 µs |   0    |
| rust@pi4 |       ndarray (matrixmultiply)        |  90   |  3.859 ms  | 7.309 ms  |  4.032 ms  |  3.931 ms  | 504.502 µs |   0    |
| rust@pi4 |              simd (wide)              |  90   |  3.853 ms  | 6.334 ms  |  3.985 ms  |  3.920 ms  | 269.716 µs |   0    |
| rust@pi4 |            ndarray (BLAS)             |  90   |  3.963 ms  | 10.119 ms |  4.483 ms  |  4.307 ms  | 842.502 µs |   0    |


## Results for model : AlexNetLite
|  Runner  |                Backend                | Runs  |    Min     |    Max     |    Mean    |   Median   |   StdDev   | Errors |
| :------: | :-----------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
|  python  | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  90   | 970.091 µs |  1.925 ms  |  1.219 ms  |  1.199 ms  | 203.077 µs |   0    |
|   rust   |       ndarray (matrixmultiply)        |  90   | 105.143 ms | 113.390 ms | 108.974 ms | 108.892 ms |  1.796 ms  |   0    |
|   rust   |              simd (wide)              |  90   | 105.990 ms | 118.049 ms | 108.769 ms | 108.750 ms |  1.641 ms  |   0    |
|   rust   |            ndarray (BLAS)             |  90   | 151.404 ms | 175.188 ms | 155.618 ms | 153.650 ms |  4.505 ms  |   0    |
|  python  |  PyTorch 2.13.0+cpu (CPU, 4 threads)  |  90   | 12.765 ms  | 47.563 ms  | 14.104 ms  | 13.141 ms  |  4.457 ms  |   0    |
| rust@pi4 |       ndarray (matrixmultiply)        |  90   | 353.874 ms | 361.722 ms | 355.113 ms | 354.521 ms |  1.493 ms  |   0    |
| rust@pi4 |              simd (wide)              |  90   | 355.242 ms | 361.952 ms | 356.378 ms | 355.986 ms |  1.155 ms  |   0    |
| rust@pi4 |            ndarray (BLAS)             |  90   | 352.098 ms | 359.755 ms | 353.287 ms | 352.886 ms |  1.346 ms  |   0    |


## Roadmap

These are the next features that I would like to implement, in no specific order

- [ ] Add activation functions support in simd
- [ ] Add support for more layer types
- [ ] Add support to other optimization backends (cuda, ...)
- [ ] Export both the python library and rust library to pip/c rates.io for easier usage
- [ ] Add support for non float models (right now only float is supported, we should allow export and run of models in int)
- [ ] Implement parser for pytorch and tensorflow models
  - I will focus mainly on tensorflow support at first, since pytorch has a good onnx exporter