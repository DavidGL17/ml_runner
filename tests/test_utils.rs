use approx::AbsDiffEq;
use core::panic;
use ml_runner::tensor::Tensor;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

pub struct FixtureModelInput {
    pub model_json: String,
    pub test_input: Tensor,
    pub test_output: Vec<f32>,
}

impl FixtureModelInput {
    pub fn load_json(name: &str) -> FixtureModelInput {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);

        let json_content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Error reading fixture '{}': '{}'", path.display(), e));

        // Parse the JSON content
        let json_value: Value = serde_json::from_str(&json_content)
            .unwrap_or_else(|e| panic!("Error parsing JSON '{}': '{}'", path.display(), e));

        // Extract the model field and convert it back to a string
        let model_json = json_value["model"].to_string();

        // Extract test_input and test_output
        let test_input: Vec<f32> = json_value["test_input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let test_output = json_value["test_output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        FixtureModelInput {
            model_json,
            // Fixtures only ever describe flat vector inputs today, so
            // Tensor::flat is the right constructor here. If a fixture
            // ever needs to describe a CHW input (e.g. for a conv model),
            // this will need to read a "test_input_shape" field instead.
            test_input: Tensor::flat(test_input),
            test_output,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FloatVec(pub Vec<f32>);

impl AbsDiffEq for FloatVec {
    type Epsilon = f32;

    fn default_epsilon() -> f32 {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(other.0.iter())
                .all(|(a, b)| f32::abs_diff_eq(a, b, epsilon))
    }
}
