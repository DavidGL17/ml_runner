use crate::activation::ActivationLayer;
use crate::add::AddLayer;
use crate::conv::Conv2DLayer;
use crate::dense::DenseLayer;
use crate::flatten::FlattenLayer;
use crate::gather::GatherLayer;
use crate::rnn::{GRULayer, RNNLayer};
use crate::shape::ShapeLayer;
use crate::tensor::{Tensor, TensorShape};
use crate::transpose::TransposeLayer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IoSpec {
    pub name: String,
    pub shape: TensorShape,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Node {
    pub id: String,
    pub inputs: Vec<String>,  // names of tensors this node reads
    pub outputs: Vec<String>, // names of tensors this node produces
    #[serde(flatten)]
    pub op: Layer,
}

macro_rules! define_layers {
    ($($variant:ident($ty:path) => $tag:literal),+ $(,)?) => {
        #[derive(Debug, Serialize, Deserialize, Clone)]
        #[serde(tag = "type")]
        pub enum Layer {
            $(#[serde(rename = $tag)] $variant($ty),)+
        }

        impl Layer {
            pub fn input_shape(&self) -> TensorShape {
                match self {
                    $(Layer::$variant(layer) => layer.input_shape(),)+
                }
            }

            pub fn output_shape(&self) -> TensorShape {
                match self {
                    $(Layer::$variant(layer) => layer.output_shape(),)+
                }
            }

            pub fn forward(&self, inputs: &[&Tensor]) -> Tensor {
                match self {
                    $(Layer::$variant(layer) => layer.forward(inputs[0]),)+
                }
            }
        }
    };
}

define_layers! {
    Dense(DenseLayer) => "dense",
    Activation(ActivationLayer) => "activation",
    Conv2D(Conv2DLayer) => "conv2d",
    Flatten(FlattenLayer) => "flatten",
    Rnn(RNNLayer) => "rnn",
    Gru(GRULayer) => "gru",
    Add(AddLayer) => "add",
    Gather(GatherLayer) => "gather",
    Shape(ShapeLayer) => "shape",
    Transpose(TransposeLayer) => "transpose",
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
        let output = layer.forward(&[&input]);
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
        let output = layer.forward(&[&input]);
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

        let output = layer.forward(&[&input]);

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
        let output = layer.forward(&[&input]);

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
        let output = layer.forward(&[&input]);

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
        let output = layer.forward(&[&input]);

        // r = sigmoid(0) = 0.5, z = sigmoid(0) = 0.5, hn_term = 0
        // n = tanh(1*2.0 + 0.5*0) = tanh(2.0)
        // h_1 = (1 - 0.5)*tanh(2.0) + 0.5*0 = 0.5*tanh(2.0)
        assert_eq!(output.shape(), TensorShape::Flat(1));
        assert_eq!(output.to_vec(), vec![0.5 * 2.0f32.tanh()]);
    }

    #[test]
    fn test_add_layer_enum_dispatch() {
        let layer = Layer::Add(AddLayer {
            shape: TensorShape::Flat(2),
            constants: vec![vec![1.0, 2.0]],
        });
        let input = Tensor::new(vec![10.0, 20.0], layer.input_shape());
        let output = layer.forward(&[&input]);
        assert_eq!(output.to_vec(), vec![11.0, 22.0]);
    }

    #[test]
    fn test_shape_layer_enum_dispatch() {
        let layer = Layer::Shape(ShapeLayer {
            input_shape: TensorShape::D3 {
                dim1: 2,
                dim2: 3,
                dim3: 4,
            },
            output_shape: TensorShape::Flat(3),
        });

        let input = Tensor::new(vec![0.0; 24], layer.input_shape());
        let output = layer.forward(&[&input]);

        assert_eq!(output.shape(), TensorShape::Flat(3));
        assert_eq!(output.to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gather_layer_enum_dispatch() {
        let layer = Layer::Gather(GatherLayer {
            input_shape: TensorShape::D2 { dim1: 3, dim2: 2 },
            output_shape: TensorShape::Flat(2),
            axis: 0,
            indices: Some(vec![1.0]),
        });

        #[rustfmt::skip]
    let input = Tensor::new(
        vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ],
        layer.input_shape(),
    );
        let output = layer.forward(&[&input]);

        assert_eq!(output.shape(), TensorShape::Flat(2));
        // row 1 of the input
        assert_eq!(output.to_vec(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_transpose_layer_enum_dispatch() {
        let layer = Layer::Transpose(TransposeLayer {
            input_shape: TensorShape::D2 { dim1: 2, dim2: 3 },
            perm: vec![1, 0],
        });
        #[rustfmt::skip]
    let input = Tensor::new(
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ],
        layer.input_shape(),
    );
        let output = layer.forward(&[&input]);
        assert_eq!(output.shape(), TensorShape::D2 { dim1: 3, dim2: 2 });
        assert_eq!(output.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    /// `Layer::forward` takes a slice to leave room for future multi-input
    /// ops, but every current variant still only reads `inputs[0]`. This
    /// documents that until a real multi-input layer lands, extra slice
    /// entries are simply ignored - so nobody is surprised later.
    #[test]
    fn test_layer_forward_only_uses_first_input_for_now() {
        let layer = Layer::Dense(DenseLayer {
            input_size: 1,
            output_size: 1,
            weights: vec![2.0],
            bias: vec![1.0],
        });
        let used = Tensor::new(vec![0.5], TensorShape::Flat(1));
        let ignored = Tensor::new(vec![999.0], TensorShape::Flat(1));

        let output = layer.forward(&[&used, &ignored]);
        assert_eq!(output.to_vec(), vec![2.0]); // 0.5*2 + 1 = 2.0, `ignored` unused
    }
}
