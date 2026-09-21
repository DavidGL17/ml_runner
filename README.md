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
|   Runner   |                Backend                | Runs  |   Min    |    Max    |   Mean   |  Median  |   StdDev   |                                                                       Errors                                                                       |
| :--------: | :-----------------------------------: | :---: | :------: | :-------: | :------: | :------: | :--------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
|   python   | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  990  | 2.400 ms | 4.654 ms  | 2.762 ms | 2.709 ms | 225.574 µs |                                                                         0                                                                          |
|    rust    |       ndarray (matrixmultiply)        |  990  | 4.728 ms | 19.880 ms | 5.498 ms | 5.189 ms |  1.008 ms  |                                                                         0                                                                          |
|    rust    |              simd (wide)              |  990  | 4.157 ms | 11.540 ms | 4.844 ms | 4.665 ms | 744.508 µs |                                                                         0                                                                          |
|    rust    |            ndarray (BLAS)             |  990  | 2.471 ms | 53.235 ms | 5.019 ms | 3.469 ms |  4.196 ms  |                                                                         0                                                                          |
| python@pi4 |                  pi4                  |   0   |    -     |     -     |    -     |    -     |     -      | remote python benchmark failed (exit 2): python3: can't open file '/home/david/ml-runner/benchmark_models.py': [Errno 2] No such file or directory |
|  rust@pi4  |                default                |   0   |    -     |     -     |    -     |    -     |     -      |                                        remote cargo run failed (exit 127): zsh:1: command not found: cargo                                         |
|  rust@pi4  |                 simd                  |   0   |    -     |     -     |    -     |    -     |     -      |                                        remote cargo run failed (exit 127): zsh:1: command not found: cargo                                         |
|  rust@pi4  |                 blas                  |   0   |    -     |     -     |    -     |    -     |     -      |                                        remote cargo run failed (exit 127): zsh:1: command not found: cargo                                         |


## Results for model : LongLinearModel
|   Runner   |                Backend                | Runs  |   Min    |    Max     |   Mean   |  Median  |   StdDev   |                                                                       Errors                                                                       |
| :--------: | :-----------------------------------: | :---: | :------: | :--------: | :------: | :------: | :--------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
|   python   | PyTorch 2.13.0+cu130 (CPU, 6 threads) |  990  | 1.810 ms |  4.440 ms  | 2.509 ms | 2.437 ms | 347.071 µs |                                                                         0                                                                          |
|    rust    |       ndarray (matrixmultiply)        |  990  | 3.218 ms | 15.106 ms  | 4.587 ms | 4.270 ms |  1.279 ms  |                                                                         0                                                                          |
|    rust    |              simd (wide)              |  990  | 2.756 ms |  7.729 ms  | 3.485 ms | 3.220 ms | 768.727 µs |                                                                         0                                                                          |
|    rust    |            ndarray (BLAS)             |  990  | 1.501 ms | 139.782 ms | 9.494 ms | 2.486 ms | 16.498 ms  |                                                                         0                                                                          |
| python@pi4 |                  pi4                  |   0   |    -     |     -      |    -     |    -     |     -      | remote python benchmark failed (exit 2): python3: can't open file '/home/david/ml-runner/benchmark_models.py': [Errno 2] No such file or directory |
|  rust@pi4  |                default                |   0   |    -     |     -      |    -     |    -     |     -      |                                        remote cargo run failed (exit 127): zsh:1: command not found: cargo                                         |
|  rust@pi4  |                 simd                  |   0   |    -     |     -      |    -     |    -     |     -      |                                        remote cargo run failed (exit 127): zsh:1: command not found: cargo                                         |
|  rust@pi4  |                 blas                  |   0   |    -     |     -      |    -     |    -     |     -      |                                        remote cargo run failed (exit 127): zsh:1: command not found: cargo                                         |



## Roadmap

These are the next features that I would like to implement, in no specific order

- [ ] Add activation functions support in simd
- [ ] Add support for more layer types
- [ ] Add support to other optimization backends (cuda, ...)
- [ ] Export both the python library and rust library to pip/c rates.io for easier usage
- [ ] Add support for non float models (right now only float is supported, we should allow export and run of models in int)
- [ ] Implement parser for pytorch and tensorflow models
  - I will focus mainly on tensorflow support at first, since pytorch has a good onnx exporter