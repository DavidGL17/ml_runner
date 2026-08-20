# ML_RUNNER

A small library that aims to provide a fast and device agnostic way of running python ML models in embedded platforms using rust


## Features

- A python library to parse ML models into an exported json format. Exports from : 
  - ONNX models
- A rust library to run the exported models on any platforms, with optimized functions depending on the platforms toolset

### Supported layers

|  layer  | simple forward | BLAS  | simd  |
| :-----: | :------------: | :---: | :---: |
| Linear  |      yes       |  yes  |  yes  |
| Conv2D  |      yes       |  yes  |  no   |
| Flatten |      yes       |  yes  |  no   |

### Supported activation functions

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


## Roadmap

These are the next features that I would like to implement, in no specific order

- Add activation functions support in simd
- Add support for more layer types (Conv, etc.)
- Export both the python library and rust library to pip/crates.io for easier usage
- Add support for non linear models (meaning models that are not just a simple chaining of layers but that have multiple paths and potentially multiple inputs/outputs)
- Add support to other optimization backends (cuda, ...)