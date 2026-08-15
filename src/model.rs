use crate::layers::Layer;
use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Model {
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
    pub layers: Vec<Layer>,
}

impl Model {
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    /// Walks the layer chain and checks that each layer's declared output
    /// shape matches the next layer's declared input shape. This only
    /// looks at declared shapes, not real data, so it's cheap enough to
    /// run right after loading a model - catching a misconfigured model
    /// (e.g. a dense layer wired to the wrong size, or a conv layer feeding
    /// straight into a dense layer without a Flatten in between) before any
    /// forward pass runs.
    pub fn validate_shapes(&self) -> Result<(), String> {
        let mut expected = self.input_shape.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            if layer.input_shape() != expected {
                return Err(format!(
                    "Shape mismatch before layer {}: pipeline has {:?}, layer expects {:?}",
                    i,
                    expected,
                    layer.input_shape()
                ));
            }
            expected = layer.output_shape();
        }

        if expected != self.output_shape {
            return Err(format!(
                "Model output shape mismatch: layers produce {:?}, model declares {:?}",
                expected, self.output_shape
            ));
        }

        Ok(())
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        assert_eq!(input.shape, self.input_shape, "Model input shape mismatch");

        let mut current = Tensor::new(input.data.clone(), input.shape.clone());

        for layer in &self.layers {
            current = layer.forward(&current);
        }

        assert_eq!(
            current.shape, self.output_shape,
            "Model output shape mismatch"
        );

        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_from_json() {
        let json = r#"
        {
            "input_shape": { "Flat": 1 },
            "output_shape": { "Flat": 1 },
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
        assert_eq!(model.input_shape, TensorShape::Flat(1));
        assert_eq!(model.output_shape, TensorShape::Flat(1));
        assert_eq!(model.layers.len(), 1);
    }

    #[test]
    fn test_model_forward_multi_layer() {
        let json = r#"
        {
            "input_shape": { "Flat": 2 },
            "output_shape": { "Flat": 1 },
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
        let input = Tensor::flat(vec![1.0, 1.0]);
        let output = model.forward(&input).unwrap();
        // Layer 1: [0.5*1 + 0.5*1, 0.5*1 + 0.5*1] = [1.0, 1.0]
        // Layer 2: [1.0*1 + 1.0*1 + 0.5] = [2.5]
        assert_eq!(output.data, vec![2.5]);
    }

    #[test]
    fn test_model_forward_with_activation() {
        let json = r#"
        {
            "input_shape": { "Flat": 2 },
            "output_shape": { "Flat": 1 },
            "layers": [
                {
                    "type": "dense",
                    "input_size": 2,
                    "output_size": 2,
                    "weights": [0.5, 0.5, 0.5, 0.5],
                    "bias": [0.0, 0.0]
                },
                {
                    "type": "activation",
                    "activation_type": "relu",
                    "shape": { "Flat": 2 }
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
        let input = Tensor::flat(vec![1.0, 1.0]);
        let output = model.forward(&input).unwrap();
        // Layer 1: [0.5*1 + 0.5*1, 0.5*1 + 0.5*1] = [1.0, 1.0]
        // Layer 2 (ReLU): [1.0, 1.0] (no change since values are positive)
        // Layer 3: [1.0*1 + 1.0*1 + 0.5] = [2.5]
        assert_eq!(output.data, vec![2.5]);
    }

    /// Model whose input is CHW (a Conv2D layer first), goes through Flatten,
    /// and finishes as a flat output. Exercises the case that motivated moving
    /// `Model` from a flat `usize` to a full `TensorShape`.
    #[test]
    fn test_model_forward_conv_then_flatten() {
        let json = r#"
        {
            "input_shape": { "CHW": { "channels": 1, "height": 2, "width": 2 } },
            "output_shape": { "Flat": 1 },
            "layers": [
                {
                    "type": "conv2d",
                    "kernel_size": 2,
                    "stride": 1,
                    "padding": 0,
                    "input_channels": 1,
                    "output_channels": 1,
                    "height": 2,
                    "width": 2,
                    "weights": [1.0, 1.0, 1.0, 1.0],
                    "bias": [0.0]
                },
                {
                    "type": "flatten",
                    "shape": { "CHW": { "channels": 1, "height": 1, "width": 1 } }
                }
            ]
        }
        "#;
        let model = Model::from_json(json).unwrap();
        assert!(model.validate_shapes().is_ok());

        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            TensorShape::CHW {
                channels: 1,
                height: 2,
                width: 2,
            },
        );
        let output = model.forward(&input).unwrap();
        // conv2d sums the whole 2x2 window with an all-ones kernel: 1+2+3+4 = 10
        // flatten is then a no-op on the data
        assert_eq!(output.data, vec![10.0]);
    }

    #[test]
    #[should_panic(expected = "Model input shape mismatch")]
    fn test_model_wrong_input_size() {
        let json = r#"
        {
            "input_shape": { "Flat": 2 },
            "output_shape": { "Flat": 1 },
            "layers": []
        }
        "#;
        let model = Model::from_json(json).unwrap();
        let _ = model.forward(&Tensor::flat(vec![1.0]));
    }

    #[test]
    #[should_panic(expected = "Model output shape mismatch")]
    fn test_model_wrong_output_size() {
        let json = r#"
        {
            "input_shape": { "Flat": 1 },
            "output_shape": { "Flat": 2 },
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
        let _ = model.forward(&Tensor::flat(vec![1.0]));
    }

    #[test]
    fn test_validate_shapes_ok() {
        let json = r#"
        {
            "input_shape": { "Flat": 2 },
            "output_shape": { "Flat": 1 },
            "layers": [
                {
                    "type": "dense",
                    "input_size": 2,
                    "output_size": 1,
                    "weights": [1.0, 1.0],
                    "bias": [0.0]
                }
            ]
        }
        "#;
        let model = Model::from_json(json).unwrap();
        assert!(model.validate_shapes().is_ok());
    }

    #[test]
    fn test_validate_shapes_catches_mismatched_layer_chain() {
        // Layer 1 outputs size 2, but layer 2 expects size 3 - this
        // would previously only be caught mid-forward-pass via a panic.
        let json = r#"
        {
            "input_shape": { "Flat": 2 },
            "output_shape": { "Flat": 1 },
            "layers": [
                {
                    "type": "dense",
                    "input_size": 2,
                    "output_size": 2,
                    "weights": [1.0, 1.0, 1.0, 1.0],
                    "bias": [0.0, 0.0]
                },
                {
                    "type": "dense",
                    "input_size": 3,
                    "output_size": 1,
                    "weights": [1.0, 1.0, 1.0],
                    "bias": [0.0]
                }
            ]
        }
        "#;
        let model = Model::from_json(json).unwrap();
        assert!(model.validate_shapes().is_err());
    }
}
