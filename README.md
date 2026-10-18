# ML_RUNNER

[![Crates.io](https://img.shields.io/crates/v/ml_runner)](https://crates.io/crates/ml_runner)
[![Crates.io downloads](https://img.shields.io/crates/d/ml_runner)](https://crates.io/crates/ml_runner)
[![docs.rs](https://img.shields.io/docsrs/ml_runner)](https://docs.rs/ml_runner)

[![PyPI](https://img.shields.io/pypi/v/ml-runner-exporter)](https://pypi.org/project/ml-runner-exporter/)
[![PyPI downloads](https://img.shields.io/pypi/dm/ml-runner-exporter)](https://pypi.org/project/ml-runner-exporter/)
[![Python versions](https://img.shields.io/pypi/pyversions/ml-runner-exporter)](https://pypi.org/project/ml-runner-exporter/)

[![License](https://img.shields.io/crates/l/ml_runner)](https://crates.io/crates/ml_runner)


A small library that aims to provide a fast and device agnostic way of running python ML models in embedded platforms using rust

Pypi package : [ml-runner-exporter](https://pypi.org/project/ml-runner-exporter)

Cargo package : [ml-runner](https://crates.io/crates/ml_runner)

### Current state of performance of the library

![speedup results](img/benchmark_rust_speedup.png)

See the [benchmark chapter](#benchmarks) for more details.

## Features

- A python library to parse ML models into an exported json format. Exports from : 
  - ONNX models
- A rust library to run the exported models on any platforms, with optimized functions depending on the platforms toolset

The library supports complex models with multiple inputs and outputs, and with multiple paths inside the model.

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

See [benchmark.md](benchmark.md)

## Supported layers


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
|  MaxPool  |       ✅        |   ✅   |                ✅                 |         |
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


## Roadmap

These are the next features that I would like to implement, in no specific order

- [ ] Add activation functions support in simd
- [ ] Add support for more layer types
- [ ] Add support to other optimization backends (cuda, ...)
- [ ] Export both the python library and rust library to pip/crates.io for easier usage
- [ ] Add support for non float models (right now only float is supported, we should allow export and run of models in int)
- [ ] Implement parser for pytorch and tensorflow models
  - I will focus mainly on tensorflow support at first, since pytorch has a good onnx exporter