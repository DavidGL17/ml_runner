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
|  Linear   |      ✅       |
|  Conv2D   |      ✅       |
|  Flatten  |      ✅       |
|  MaxPool  |      ✅       |
|    RNN    |      ✅       |
|    GRU    |      ✅       |
|   LSTM    |      ✅       |
|    Add    |      ✅       |
|  Concat   |      ✅       |
|  Expand   |      ✅       |
|  Gather   |      ✅       |
| Identity  |      ✅       |
|   Shape   |      ✅       |
|  Squeeze  |      ✅       |
| Transpose |      ✅       |
| Unsqueeze |      ✅       |

#### Supported activation functions

| function | ONNX support |
| :------: | :----------: |
|   ReLU   |      ✅       |
| Sigmoid  |      ✅       |
|   Tanh   |      ✅       |
| Softmax  |      ✅       |
|  Linear  |      ✅       |

### Supported layers in Rust library

|   layer   | simple forward | BLAS  |               simd               | comment |
| :-------: | :------------: | :---: | :------------------------------: | :-----: |
|  Linear   |       ✅        |   ✅   |                ✅                 |         |
|  Conv2D   |       ✅        |   ✅   |                ✅                 |         |
|  Flatten  |       ✅        |   ✅   | <span style="color:red">✖</span> |         |
|  MaxPool  |       ✅        |   ✅   | <span style="color:red">✖</span> |         |
|    RNN    |       ✅        |   ✅   |                ✅                 |         |
|    GRU    |       ✅        |   ✅   |                ✅                 |         |
|   LSTM    |       ✅        |   ✅   |                ✅                 |         |
|    Add    |       ✅        |   ✅   |                ✅                 |         |
|  Concat   |       ✅        |   ✅   | <span style="color:red">✖</span> |         |
|  Expand   |       ✅        |   ✅   | <span style="color:red">✖</span> |         |
|  Gather   |       ✅        |   ✅   | <span style="color:red">✖</span> |         |
| Identity  |       ✅        |   ✅   | <span style="color:red">✖</span> |         |
|   Shape   |       ✅        |   ✅   | <span style="color:red">✖</span> |         |
|  Squeeze  |       ✅        |   ✅   | <span style="color:red">✖</span> |         |
| Transpose |       ✅        |   ✅   | <span style="color:red">✖</span> |         |
| Unsqueeze |       ✅        |   ✅   | <span style="color:red">✖</span> |         |

#### Supported activation functions

| function | simple application | BLAS  |               simd               |
| :------: | :----------------: | :---: | :------------------------------: |
|   ReLU   |         ✅          |   ✅   | <span style="color:red">✖</span> |
| Sigmoid  |         ✅          |   ✅   | <span style="color:red">✖</span> |
|   Tanh   |         ✅          |   ✅   | <span style="color:red">✖</span> |
| Softmax  |         ✅          |   ✅   | <span style="color:red">✖</span> |
|  Linear  |         ✅          |   ✅   | <span style="color:red">✖</span> |

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
|   Runner   |               Backend               | Runs  |    Min     |    Max     |    Mean    |   Median   |  StdDev   | Errors |
| :--------: | :---------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :-------: | :----: |
|   python   | PyTorch 2.13.0+cpu (CPU, 6 threads) |  90   | 91.896 µs  | 254.490 µs | 98.498 µs  | 93.746 µs  | 19.302 µs |   0    |
|    rust    |      ndarray (matrixmultiply)       |  90   | 49.497 µs  | 120.293 µs | 50.942 µs  | 49.697 µs  | 7.708 µs  |   0    |
|    rust    |             simd (wide)             |  90   | 37.899 µs  | 56.599 µs  | 38.973 µs  | 38.099 µs  | 3.058 µs  |   0    |
|    rust    |           ndarray (BLAS)            |  90   | 15.199 µs  | 120.298 µs | 17.292 µs  | 15.500 µs  | 11.811 µs |   0    |
| python@pi4 | PyTorch 2.13.0+cpu (CPU, 4 threads) |  90   |  2.676 ms  |  3.022 ms  |  2.792 ms  |  2.780 ms  | 68.336 µs |   0    |
|  rust@pi4  |      ndarray (matrixmultiply)       |  90   | 194.628 µs | 263.629 µs | 201.892 µs | 198.406 µs | 11.821 µs |   0    |
|  rust@pi4  |             simd (wide)             |  90   | 209.332 µs | 474.202 µs | 235.135 µs | 214.111 µs | 42.661 µs |   0    |
|  rust@pi4  |           ndarray (BLAS)            |  90   | 136.740 µs | 407.924 µs | 175.923 µs | 172.888 µs | 42.764 µs |   0    |


## Results for model : LeNet5
|   Runner   |               Backend               | Runs  |    Min     |    Max     |    Mean    |   Median   |   StdDev   | Errors |
| :--------: | :---------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cpu (CPU, 6 threads) |  90   | 599.899 µs |  1.163 ms  | 744.490 µs | 717.450 µs | 103.960 µs |   0    |
|    rust    |      ndarray (matrixmultiply)       |  90   |  1.050 ms  |  1.219 ms  |  1.064 ms  |  1.061 ms  | 21.132 µs  |   0    |
|    rust    |             simd (wide)             |  90   | 147.500 µs | 194.600 µs | 150.967 µs | 147.900 µs |  8.961 µs  |   0    |
|    rust    |           ndarray (BLAS)            |  90   |  1.048 ms  |  6.422 ms  |  2.029 ms  |  1.914 ms  | 622.109 µs |   0    |
| python@pi4 | PyTorch 2.13.0+cpu (CPU, 4 threads) |  90   |  3.372 ms  |  7.290 ms  |  3.607 ms  |  3.497 ms  | 523.462 µs |   0    |
|  rust@pi4  |      ndarray (matrixmultiply)       |  90   |  3.495 ms  |  3.781 ms  |  3.540 ms  |  3.512 ms  | 63.278 µs  |   0    |
|  rust@pi4  |             simd (wide)             |  90   | 682.682 µs | 873.718 µs | 688.967 µs | 683.451 µs | 21.530 µs  |   0    |
|  rust@pi4  |           ndarray (BLAS)            |  90   |  3.641 ms  |  8.432 ms  |  4.016 ms  |  3.772 ms  | 715.717 µs |   0    |


## Results for model : ResNet18Cifar
|   Runner   |               Backend               | Runs  |     Min     |     Max     |    Mean     |   Median    |   StdDev   | Errors |
| :--------: | :---------------------------------: | :---: | :---------: | :---------: | :---------: | :---------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cpu (CPU, 6 threads) |  90   |  11.216 ms  |  13.599 ms  |  12.305 ms  |  12.293 ms  | 624.581 µs |   0    |
|    rust    |      ndarray (matrixmultiply)       |  90   | 1718.186 ms | 2095.916 ms | 1774.921 ms | 1751.964 ms | 61.342 ms  |   0    |
|    rust    |             simd (wide)             |  90   | 116.151 ms  | 136.158 ms  | 120.161 ms  | 119.236 ms  |  3.375 ms  |   0    |
|    rust    |           ndarray (BLAS)            |  90   | 1716.017 ms | 2207.893 ms | 1766.499 ms | 1735.467 ms | 86.572 ms  |   0    |
| python@pi4 | PyTorch 2.13.0+cpu (CPU, 4 threads) |  90   | 115.224 ms  | 217.947 ms  | 130.071 ms  | 122.797 ms  | 18.866 ms  |   0    |
|  rust@pi4  |      ndarray (matrixmultiply)       |  90   | 5911.441 ms | 6358.975 ms | 5935.305 ms | 5916.225 ms | 67.651 ms  |   0    |
|  rust@pi4  |             simd (wide)             |  90   | 584.094 ms  | 639.562 ms  | 588.680 ms  | 587.118 ms  |  6.923 ms  |   0    |
|  rust@pi4  |           ndarray (BLAS)            |  90   | 5926.182 ms | 9536.005 ms | 5973.480 ms | 5932.462 ms | 377.677 ms |   0    |


## Results for model : LSTMClassifier
|   Runner   |               Backend               | Runs  |    Min     |    Max     |    Mean    |   Median   |   StdDev   | Errors |
| :--------: | :---------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cpu (CPU, 6 threads) |  90   | 452.400 µs |  2.612 ms  | 669.788 µs | 591.450 µs | 303.512 µs |   0    |
|    rust    |      ndarray (matrixmultiply)       |  90   |  1.107 ms  |  1.514 ms  |  1.153 ms  |  1.139 ms  | 57.725 µs  |   0    |
|    rust    |             simd (wide)             |  90   | 815.800 µs | 973.500 µs | 833.559 µs | 819.350 µs | 29.408 µs  |   0    |
|    rust    |           ndarray (BLAS)            |  90   | 812.200 µs |  1.772 ms  |  1.261 ms  |  1.343 ms  | 219.139 µs |   0    |
| python@pi4 | PyTorch 2.13.0+cpu (CPU, 4 threads) |  90   |  5.856 ms  |  8.289 ms  |  6.133 ms  |  6.038 ms  | 403.953 µs |   0    |
|  rust@pi4  |      ndarray (matrixmultiply)       |  90   |  3.776 ms  |  4.384 ms  |  3.880 ms  |  3.847 ms  | 97.515 µs  |   0    |
|  rust@pi4  |             simd (wide)             |  90   |  3.681 ms  |  5.109 ms  |  3.848 ms  |  3.785 ms  | 220.169 µs |   0    |
|  rust@pi4  |           ndarray (BLAS)            |  90   |  2.633 ms  |  3.080 ms  |  2.779 ms  |  2.756 ms  | 111.023 µs |   0    |


## Results for model : AlexNetLite
|   Runner   |               Backend               | Runs  |    Min     |    Max     |    Mean    |   Median   |   StdDev   | Errors |
| :--------: | :---------------------------------: | :---: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
|   python   | PyTorch 2.13.0+cpu (CPU, 6 threads) |  90   | 875.400 µs |  1.745 ms  |  1.184 ms  |  1.179 ms  | 176.547 µs |   0    |
|    rust    |      ndarray (matrixmultiply)       |  90   | 105.468 ms | 128.976 ms | 109.770 ms | 109.177 ms |  3.337 ms  |   0    |
|    rust    |             simd (wide)             |  90   | 15.001 ms  | 17.005 ms  | 15.523 ms  | 15.447 ms  | 306.171 µs |   0    |
|    rust    |           ndarray (BLAS)            |  90   | 149.402 ms | 186.177 ms | 154.248 ms | 152.428 ms |  5.196 ms  |   0    |
| python@pi4 | PyTorch 2.13.0+cpu (CPU, 4 threads) |  90   | 12.357 ms  | 36.246 ms  | 14.286 ms  | 12.797 ms  |  4.802 ms  |   0    |
|  rust@pi4  |      ndarray (matrixmultiply)       |  90   | 355.300 ms | 363.338 ms | 356.673 ms | 356.149 ms |  1.477 ms  |   0    |
|  rust@pi4  |             simd (wide)             |  90   | 69.085 ms  | 75.760 ms  | 70.166 ms  | 69.712 ms  |  1.235 ms  |   0    |
|  rust@pi4  |           ndarray (BLAS)            |  90   | 361.759 ms | 369.024 ms | 362.892 ms | 362.431 ms |  1.486 ms  |   0    |


## Roadmap

These are the next features that I would like to implement, in no specific order

- [ ] Add activation functions support in simd
- [ ] Add support for more layer types
- [ ] Add support to other optimization backends (cuda, ...)
- [ ] Export both the python library and rust library to pip/crates.io for easier usage
- [ ] Add support for non float models (right now only float is supported, we should allow export and run of models in int)
- [ ] Implement parser for pytorch and tensorflow models
  - I will focus mainly on tensorflow support at first, since pytorch has a good onnx exporter