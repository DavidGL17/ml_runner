mod dense;
mod layers;
mod model;

use crate::model::Model;

fn main() {
    // Example JSON representation of a simple model:
    // Input (2) -> Dense (2x2) -> Output (1)
    // Weights: [[0.5, 0.2], [0.1, 0.8]], Bias: [0.1, -0.2]
    // Then Dense (2x1) -> Output (1)
    // Weights: [[0.7, 0.3]], Bias: [0.5]
    let model_json = r#"
    {
        "input_size": 2,
        "output_size": 1,
        "layers": [
            {
                "type": "dense",
                "input_size": 2,
                "output_size": 2,
                "weights": [0.5, 0.2, 0.1, 0.8],
                "bias": [0.1, -0.2]
            },
            {
                "type": "dense",
                "input_size": 2,
                "output_size": 1,
                "weights": [0.7, 0.3],
                "bias": [0.5]
            }
        ]
    }
    "#;

    // Load the model
    let model = match Model::from_json(model_json) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing model JSON: {}", e);
            return;
        }
    };

    println!("Model loaded successfully!");
    println!("Input size: {}", model.input_size);
    println!("Output size: {}", model.output_size);

    // Test input
    let input = vec![1.0, 2.0];
    println!("Input: {:?}", input);

    // Run forward pass
    match model.forward(&input) {
        Ok(output) => println!("Output: {:?}", output),
        Err(e) => eprintln!("Error during forward pass: {}", e),
    }
}
