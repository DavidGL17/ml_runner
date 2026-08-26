# ML_RUNNER

A small library that aims to provide a fast and device agnostic way of running python ML models in embedded platforms using rust


## Features

- A python library to parse ML models into an exported json format. Exports from : 
  - ONNX models
- A rust library to run the exported models on any platforms, with optimized functions depending on the platforms toolset

### Supported layers in python library

|  layer  | ONNX support |
| :-----: | :----------: |
| Linear  |     yes      |
| Conv2D  |     yes      |
| Flatten |     yes      |
|   RNN   |     yes      |
|   GRU   |     yes      |

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
python main.py --model-path simple_linear_model.onnx --output-path export.json
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

## Results for model : HugeLinearModel
| Runner |             Backend             | Runs  |   Min    |    Max     |   Mean    |  Median  |   StdDev   | Errors |
| :----: | :-----------------------------: | :---: | :------: | :--------: | :-------: | :------: | :--------: | :----: |
| python | PyTorch 2.7.0 (CPU, 14 threads) |  990  | 1.620 ms |  7.249 ms  | 4.670 ms  | 4.688 ms | 625.494 µs |   0    |
|  rust  |    ndarray (matrixmultiply)     |  990  | 4.452 ms | 25.582 ms  | 7.154 ms  | 5.987 ms |  3.014 ms  |   0    |
|  rust  |           simd (wide)           |  990  | 3.905 ms | 25.319 ms  | 6.739 ms  | 4.735 ms |  4.554 ms  |   0    |
|  rust  |         ndarray (BLAS)          |  990  | 2.835 ms | 112.862 ms | 15.437 ms | 8.690 ms | 16.322 ms  |   0    |


## Results for model : LongLinearModel
| Runner |             Backend             | Runs  |   Min    |    Max     |   Mean    |  Median  |   StdDev   | Errors |
| :----: | :-----------------------------: | :---: | :------: | :--------: | :-------: | :------: | :--------: | :----: |
| python | PyTorch 2.7.0 (CPU, 14 threads) |  990  | 3.721 ms | 15.246 ms  | 6.042 ms  | 5.654 ms |  1.526 ms  |   0    |
|  rust  |    ndarray (matrixmultiply)     |  990  | 3.185 ms | 16.646 ms  | 5.359 ms  | 4.825 ms |  2.141 ms  |   0    |
|  rust  |           simd (wide)           |  990  | 2.762 ms |  6.852 ms  | 3.541 ms  | 3.540 ms | 466.682 µs |   0    |
|  rust  |         ndarray (BLAS)          |  990  | 2.150 ms | 301.096 ms | 17.027 ms | 4.247 ms | 35.321 ms  |   0    |



## Roadmap

These are the next features that I would like to implement, in no specific order

- [ ] Add activation functions support in simd
- [ ] Add support for more layer types
- [ ] Add support for non linear models (meaning models that are not just a simple chaining of layers but that have multiple paths and potentially multiple inputs/outputs)
- [ ] Add support to other optimization backends (cuda, ...)
- [ ] Export both the python library and rust library to pip/c rates.io for easier usage
- [ ] Add support for non float models (right now only float is supported, we should allow export and run of models in int)
- [ ] Implement parser for pytorch and tensorflow models
  - I will focus mainly on tensorflow support at first, since pytorch has a good onnx exporter