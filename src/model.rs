use crate::layers::Layer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Model {
    pub input_size: usize,
    pub output_size: usize,
    pub layers: Vec<Layer>,
}

impl Model {
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        assert_eq!(input.len(), self.input_size, "Model input size mismatch");

        let mut current_input = input.to_vec();

        for layer in &self.layers {
            current_input = layer.forward(&current_input);
        }

        assert_eq!(
            current_input.len(),
            self.output_size,
            "Model output size mismatch"
        );

        Ok(current_input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_from_json() {
        let json = r#"
        {
            "input_size": 1,
            "output_size": 1,
            "layers": [
                {
                    "type": "dense",
                    "input_size": 1,
                    "output_size": 1,
                    "weights": [2.0],
                    "bias": [1.0]
                }
            ]
        }
        "#;
        let model = Model::from_json(json).unwrap();
        assert_eq!(model.input_size, 1);
        assert_eq!(model.output_size, 1);
        assert_eq!(model.layers.len(), 1);
    }

    #[test]
    fn test_model_forward_multi_layer() {
        let json = r#"
        {
            "input_size": 2,
            "output_size": 1,
            "layers": [
                {
                    "type": "dense",
                    "input_size": 2,
                    "output_size": 2,
                    "weights": [0.5, 0.5, 0.5, 0.5],
                    "bias": [0.0, 0.0]
                },
                {
                    "type": "dense",
                    "input_size": 2,
                    "output_size": 1,
                    "weights": [1.0, 1.0],
                    "bias": [0.5]
                }
            ]
        }
        "#;
        let model = Model::from_json(json).unwrap();
        let input = vec![1.0, 1.0];
        let output = model.forward(&input).unwrap();
        // Layer 1: [0.5*1 + 0.5*1, 0.5*1 + 0.5*1] = [1.0, 1.0]
        // Layer 2: [1.0*1 + 1.0*1 + 0.5] = [2.5]
        assert_eq!(output, vec![2.5]);
    }

    #[test]
    #[should_panic(expected = "Model input size mismatch")]
    fn test_model_wrong_input_size() {
        let json = r#"
        {
            "input_size": 2,
            "output_size": 1,
            "layers": []
        }
        "#;
        let model = Model::from_json(json).unwrap();
        let _ = model.forward(&[1.0]);
    }

    #[test]
    #[should_panic(expected = "Model output size mismatch")]
    fn test_model_wrong_output_size() {
        let json = r#"
        {
            "input_size": 1,
            "output_size": 2,
            "layers": [
                {
                    "type": "dense",
                    "input_size": 1,
                    "output_size": 1,
                    "weights": [1.0],
                    "bias": [0.0]
                }
            ]
        }
        "#;
        let model = Model::from_json(json).unwrap();
        let _ = model.forward(&[1.0]);
    }
}
