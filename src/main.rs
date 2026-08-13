mod activation;
mod conv;
mod dense;
mod layers;
mod model;
mod tensor;

use crate::model::Model;
use crate::tensor::Tensor;
use std::env;
use std::fs;

fn main() {
    // Path to the exported model JSON. Defaults to "export.json" in the
    // current directory, or pass a path as the first CLI argument:
    //   cargo run -- path/to/export.json
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "export.json".to_string());

    let model_json: String = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(e) => {
            eprintln!("Error reading model file '{}': {}", path, e);
            return;
        }
    };

    // Load the model
    let model = match Model::from_json(&model_json) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing model JSON: {}", e);
            return;
        }
    };

    println!("Model loaded successfully from '{}'!", path);
    println!("Input size: {}", model.input_size);
    println!("Output size: {}", model.output_size);

    // Catch a misconfigured layer chain (e.g. mismatched dense sizes)
    // before running any data through it.
    if let Err(e) = model.validate_shapes() {
        eprintln!("Model shape validation failed: {}", e);
        return;
    }

    // Test input
    let input = Tensor::flat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    println!("Input: {:?}", input.data);

    // Run forward pass
    match model.forward(&input) {
        Ok(output) => println!("Output: {:?}", output.data),
        Err(e) => eprintln!("Error during forward pass: {}", e),
    }
}
