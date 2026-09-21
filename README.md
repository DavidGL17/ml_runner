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
|  Runner  |                Backend                | Runs  |    Min    |    Max     |   Mean    |  Median   |   StdDev   | Errors |
| :------: | :-----------------------------------: | :---: | :-------: | :--------: | :-------: | :-------: | :--------: | :----: |
|  python  | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  990  | 2.687 ms  | 15.460 ms  | 4.885 ms  | 4.615 ms  |  1.583 ms  |   0    |
|   rust   |       ndarray (matrixmultiply)        |  990  | 5.008 ms  | 22.087 ms  | 7.947 ms  | 7.820 ms  |  1.820 ms  |   0    |
|   rust   |              simd (wide)              |  990  | 4.336 ms  | 24.089 ms  | 7.156 ms  | 6.824 ms  |  1.944 ms  |   0    |
|   rust   |            ndarray (BLAS)             |  990  | 2.774 ms  | 122.733 ms | 34.098 ms | 31.745 ms | 16.052 ms  |   0    |
| rust@pi4 |       ndarray (matrixmultiply)        |  990  | 15.886 ms | 24.844 ms  | 16.768 ms | 16.403 ms | 988.648 µs |   0    |
| rust@pi4 |              simd (wide)              |  990  | 16.142 ms | 22.196 ms  | 16.538 ms | 16.364 ms | 594.539 µs |   0    |

## Results for model : LongLinearModel
|  Runner  |                Backend                | Runs  |   Min    |    Max     |   Mean    |  Median   |   StdDev   | Errors |
| :------: | :-----------------------------------: | :---: | :------: | :--------: | :-------: | :-------: | :--------: | :----: |
|  python  | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  990  | 1.769 ms | 32.579 ms  | 2.751 ms  | 2.473 ms  |  1.678 ms  |   0    |
|   rust   |       ndarray (matrixmultiply)        |  990  | 2.973 ms |  6.049 ms  | 3.693 ms  | 3.531 ms  | 597.007 µs |   0    |
|   rust   |              simd (wide)              |  990  | 2.571 ms |  7.891 ms  | 3.473 ms  | 3.287 ms  | 776.920 µs |   0    |
|   rust   |            ndarray (BLAS)             |  990  | 1.395 ms | 197.835 ms | 32.119 ms | 16.692 ms | 37.198 ms  |   0    |
| rust@pi4 |       ndarray (matrixmultiply)        |  990  | 9.712 ms | 14.942 ms  | 10.122 ms | 9.894 ms  | 669.368 µs |   0    |
| rust@pi4 |              simd (wide)              |  990  | 9.617 ms | 16.837 ms  | 9.995 ms  | 9.822 ms  | 564.428 µs |   0    |



## Roadmap

These are the next features that I would like to implement, in no specific order

- [ ] Add activation functions support in simd
- [ ] Add support for more layer types
- [ ] Add support to other optimization backends (cuda, ...)
- [ ] Export both the python library and rust library to pip/c rates.io for easier usage
- [ ] Add support for non float models (right now only float is supported, we should allow export and run of models in int)
- [ ] Implement parser for pytorch and tensorflow models
  - I will focus mainly on tensorflow support at first, since pytorch has a good onnx exporter