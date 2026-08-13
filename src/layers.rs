use crate::activation::ActivationLayer;
use crate::conv::Conv2DLayer;
use crate::dense::DenseLayer;
use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum Layer {
    #[serde(rename = "dense")]
    Dense(DenseLayer),
    #[serde(rename = "activation")]
    Activation(ActivationLayer),
    #[serde(rename = "conv2d")]
    Conv2D(Conv2DLayer),
}

impl Layer {
    /// The shape this layer expects to receive. Used by
    /// `Model::validate_shapes` to check the layer chain before any data
    /// actually flows through it.
    pub fn input_shape(&self) -> TensorShape {
        match self {
            Layer::Dense(layer) => layer.input_shape(),
            Layer::Activation(layer) => layer.input_shape(),
            Layer::Conv2D(layer) => layer.input_shape(),
        }
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        match self {
            Layer::Dense(layer) => layer.output_shape(),
            Layer::Activation(layer) => layer.output_shape(),
            Layer::Conv2D(layer) => layer.output_shape(),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            Layer::Dense(layer) => layer.forward(input),
            Layer::Activation(layer) => layer.forward(input),
            Layer::Conv2D(layer) => layer.forward(input),
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
        let input = Tensor::flat(vec![0.5]);
        let output = layer.forward(&input);
        // (0.5 * 2.0) + 1.0 = 2.0
        assert_eq!(output.data, vec![2.0]);
    }

    #[test]
    fn test_activation_layer_enum_dispatch() {
        let layer = Layer::Activation(ActivationLayer {
            activation_type: ActivationType::Sigmoid,
            input_size: 1,
        });
        let input = Tensor::flat(vec![0.0]);
        let output = layer.forward(&input);
        // Sigmoid(0) = 0.5
        assert_eq!(output.data, vec![0.5]);
    }
}
