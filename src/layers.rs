use crate::activation::ActivationLayer;
use crate::dense::DenseLayer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum Layer {
    #[serde(rename = "dense")]
    Dense(DenseLayer),
    #[serde(rename = "activation")]
    Activation(ActivationLayer),
}

impl Layer {
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        match self {
            Layer::Dense(layer) => layer.forward(input),
            Layer::Activation(layer) => layer.forward(input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::ActivationType;

    #[test]
    fn test_layer_enum_dispatch() {
        let layer = Layer::Dense(DenseLayer {
            input_size: 1,
            output_size: 1,
            weights: vec![2.0],
            bias: vec![1.0],
        });
        let input = vec![0.5];
        let output = layer.forward(&input);
        // (0.5 * 2.0) + 1.0 = 2.0
        assert_eq!(output, vec![2.0]);
    }

    #[test]
    fn test_activation_layer_enum_dispatch() {
        let layer = Layer::Activation(ActivationLayer {
            activation_type: ActivationType::Sigmoid,
            input_size: 1,
        });
        let input = vec![0.0];
        let output = layer.forward(&input);
        // Sigmoid(0) = 0.5
        assert_eq!(output, vec![0.5]);
    }
}
