use approx::AbsDiffEq;
use core::panic;
use ml_runner::tensor::{Tensor, TensorShape};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

pub struct FixtureModelInput {
    pub model_json: String,
    pub input_name: String,
    pub test_input: Tensor,
    pub output_name: String,
    pub test_output: Vec<f32>,
}

impl FixtureModelInput {
    pub fn load_json(name: &str) -> FixtureModelInput {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);

        let json_content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Error reading fixture '{}': '{}'", path.display(), e));

        let json_value: Value = serde_json::from_str(&json_content)
            .unwrap_or_else(|e| panic!("Error parsing JSON '{}': '{}'", path.display(), e));

        let model_value = &json_value["model"];
        let model_json = model_value.to_string();

        // Every fixture is still single-input/single-output for now (every
        // `Layer` variant is - see `layers.rs`), so we just read the first
        // (only) entry of the model's `inputs`/`outputs` arrays and match
        // the fixture's flat `test_input`/`test_output` arrays against them.
        let inputs = model_value["inputs"].as_array().unwrap_or_else(|| {
            panic!(
                "Fixture '{}': model has no 'inputs' array - is this an old-format fixture?",
                path.display()
            )
        });
        let outputs = model_value["outputs"].as_array().unwrap_or_else(|| {
            panic!(
                "Fixture '{}': model has no 'outputs' array - is this an old-format fixture?",
                path.display()
            )
        });

        assert_eq!(
            inputs.len(),
            1,
            "Fixture '{}': multi-input models aren't supported by the test harness yet",
            path.display()
        );
        assert_eq!(
            outputs.len(),
            1,
            "Fixture '{}': multi-output models aren't supported by the test harness yet",
            path.display()
        );

        let input_name = inputs[0]["name"]
            .as_str()
            .unwrap_or_else(|| panic!("Fixture '{}': input spec missing 'name'", path.display()))
            .to_string();
        let output_name = outputs[0]["name"]
            .as_str()
            .unwrap_or_else(|| panic!("Fixture '{}': output spec missing 'name'", path.display()))
            .to_string();

        let input_shape: TensorShape = serde_json::from_value(inputs[0]["shape"].clone())
            .unwrap_or_else(|e| {
                panic!(
                    "Error parsing model input shape from fixture '{}': '{}'",
                    path.display(),
                    e
                )
            });

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
            input_name,
            // Tensor::new's debug_assert_eq on numel() will panic with a clear
            // message if input_shape and the flat test_input array disagree on
            // size, so no separate validation is needed here.
            test_input: Tensor::new(test_input, input_shape),
            output_name,
            test_output,
        }
    }

    /// Runs `model` against this fixture's `test_input`, returning the
    /// declared output tensor as a flat `Vec<f32>`. Wraps the `HashMap`
    /// plumbing `Model::forward` now needs so individual tests don't have
    /// to repeat it.
    pub fn run(&self, model: &ml_runner::model::Model) -> Vec<f32> {
        let mut inputs = HashMap::new();
        inputs.insert(self.input_name.clone(), self.test_input.clone());

        let outputs = model
            .forward(inputs)
            .unwrap_or_else(|e| panic!("Error running fixture model forward pass: {}", e));

        outputs[&self.output_name].to_vec()
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