mod activation;
mod conv;
mod dense;
mod flatten;
mod layers;
mod model;
mod tensor;

use crate::model::Model;
use crate::tensor::{Tensor, TensorShape};
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
    // println!("Input size: {}", model.input_shape);
    // println!("Output size: {}", model.input_shape);

    // Validate the model
    match model.validate_shapes() {
        Ok(m) => m,
        Err(e) => {
            eprint!("Error when validating model shapes:\n{}", e);
            return;
        }
    }

    // Test input
    let input = Tensor::new(
        [
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]
        .to_vec(),
        TensorShape::CHW {
            channels: 1,
            height: 4,
            width: 4,
        },
    );
    println!("Input: {:?}", input.data);

    // Run forward pass
    match model.forward(&input) {
        Ok(output) => println!("Output: {:?}", output.data),
        Err(e) => eprintln!("Error during forward pass: {}", e),
    }
}
