use crate::activation::ActivationLayer;
use crate::conv::Conv2DLayer;
use crate::dense::DenseLayer;
use crate::flatten::FlattenLayer;
use crate::rnn::{GRULayer, RNNLayer};
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
    #[serde(rename = "flatten")]
    Flatten(FlattenLayer),
    #[serde(rename = "rnn")]
    Rnn(RNNLayer),
    #[serde(rename = "gru")]
    Gru(GRULayer),
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
            Layer::Flatten(layer) => layer.input_shape(),
            Layer::Rnn(layer) => layer.input_shape(),
            Layer::Gru(layer) => layer.input_shape(),
        }
    }

    /// The shape this layer produces, given a matching input shape.
    pub fn output_shape(&self) -> TensorShape {
        match self {
            Layer::Dense(layer) => layer.output_shape(),
            Layer::Activation(layer) => layer.output_shape(),
            Layer::Conv2D(layer) => layer.output_shape(),
            Layer::Flatten(layer) => layer.output_shape(),
            Layer::Rnn(layer) => layer.output_shape(),
            Layer::Gru(layer) => layer.output_shape(),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            Layer::Dense(layer) => layer.forward(input),
            Layer::Activation(layer) => layer.forward(input),
            Layer::Conv2D(layer) => layer.forward(input),
            Layer::Flatten(layer) => layer.forward(input),
            Layer::Rnn(layer) => layer.forward(input),
            Layer::Gru(layer) => layer.forward(input),
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
        let input = Tensor::new(vec![0.5], TensorShape::Flat(1));
        let output = layer.forward(&input);
        // (0.5 * 2.0) + 1.0 = 2.0
        assert_eq!(output.to_vec(), vec![2.0]);
    }

    #[test]
    fn test_activation_layer_enum_dispatch() {
        let layer = Layer::Activation(ActivationLayer {
            activation_type: ActivationType::Sigmoid,
            shape: TensorShape::Flat(1),
        });
        let input = Tensor::new(vec![0.0], TensorShape::Flat(1));
        let output = layer.forward(&input);
        // Sigmoid(0) = 0.5
        assert_eq!(output.to_vec(), vec![0.5]);
    }

    #[test]
    fn test_conv2d_layer_enum_dispatch() {
        let layer = Layer::Conv2D(Conv2DLayer {
            kernel_size: 2,
            stride: 1,
            padding: 0,
            input_channels: 1,
            output_channels: 1,
            height: 2,
            width: 2,
            weights: vec![1.0; 4],
            bias: vec![1.0],
        });

        #[rustfmt::skip]
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            layer.input_shape(),
        );

        let output = layer.forward(&input);

        assert_eq!(
            output.shape(),
            TensorShape::D3 {
                dim1: 1,
                dim2: 1,
                dim3: 1,
            }
        );
        // (1+2+3+4) + bias(1.0) = 11.0
        assert_eq!(output.to_vec(), vec![11.0]);
    }

    #[test]
    fn test_flatten_layer_enum_dispatch() {
        let layer = Layer::Flatten(FlattenLayer {
            shape: TensorShape::D3 {
                dim1: 2,
                dim2: 1,
                dim3: 2,
            },
        });

        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], layer.input_shape());
        let output = layer.forward(&input);

        assert_eq!(output.shape(), TensorShape::Flat(4));
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rnn_layer_enum_dispatch() {
        let layer = Layer::Rnn(RNNLayer {
            seq_len: 1,
            input_size: 2,
            hidden_size: 1,
            weights_ih: vec![1.0, 1.0],
            weights_hh: vec![0.0],
            bias_ih: vec![0.0],
            bias_hh: vec![0.0],
            activation_type: ActivationType::Linear,
            return_sequences: false,
        });

        let input = Tensor::new(vec![1.0, 2.0], layer.input_shape());
        let output = layer.forward(&input);

        // seq_len = 1 means the hidden-to-hidden term is multiplied by the
        // zero initial state, so this reduces to a single Dense-like step:
        // 1*1 + 1*2 + 0 = 3.0
        assert_eq!(output.shape(), TensorShape::Flat(1));
        assert_eq!(output.to_vec(), vec![3.0]);
    }

    #[test]
    fn test_gru_layer_enum_dispatch() {
        let layer = Layer::Gru(GRULayer {
            seq_len: 1,
            input_size: 1,
            hidden_size: 1,
            weights_ir: vec![0.0],
            weights_hr: vec![0.0],
            bias_ir: vec![0.0],
            bias_hr: vec![0.0],
            weights_iz: vec![0.0],
            weights_hz: vec![0.0],
            bias_iz: vec![0.0],
            bias_hz: vec![0.0],
            weights_in: vec![1.0],
            weights_hn: vec![0.0],
            bias_in: vec![0.0],
            bias_hn: vec![0.0],
            recurrent_activation_type: ActivationType::Sigmoid,
            activation_type: ActivationType::Tanh,
            return_sequences: false,
        });

        let input = Tensor::new(vec![2.0], layer.input_shape());
        let output = layer.forward(&input);

        // r = sigmoid(0) = 0.5, z = sigmoid(0) = 0.5, hn_term = 0
        // n = tanh(1*2.0 + 0.5*0) = tanh(2.0)
        // h_1 = (1 - 0.5)*tanh(2.0) + 0.5*0 = 0.5*tanh(2.0)
        assert_eq!(output.shape(), TensorShape::Flat(1));
        assert_eq!(output.to_vec(), vec![0.5 * 2.0f32.tanh()]);
    }
}