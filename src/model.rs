use std::collections::HashMap;

use crate::layers::{IoSpec, Node};
use crate::tensor::{Tensor, TensorShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Model {
    pub inputs: Vec<IoSpec>,  // multiple model inputs now allowed
    pub outputs: Vec<IoSpec>, // multiple model outputs now allowed
    pub nodes: Vec<Node>,
}

impl Model {
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    /// Walks the graph and checks that each node's declared input shapes
    /// match what upstream tensors actually produce. This only looks at
    /// declared shapes, not real data, so it's cheap enough to run right
    /// after loading a model - catching a misconfigured graph (wrong size,
    /// a typo'd tensor name, a conv layer feeding straight into a dense
    /// layer without a Flatten) before any forward pass runs.
    ///
    /// Every `Layer` variant is still single-input/single-output for now
    /// (see `layers.rs`), so a node with any other arity is rejected here
    /// rather than silently truncated - this is the one thing that has to
    /// change when multi-input layers (e.g. a merge/add-two-tensors op)
    /// are added later.
    pub fn validate_shapes(&self) -> Result<(), String> {
        let mut shapes: HashMap<String, TensorShape> = self
            .inputs
            .iter()
            .map(|s| (s.name.clone(), s.shape.clone()))
            .collect();

        for node in &self.nodes {
            if node.inputs.len() != 1 || node.outputs.len() != 1 {
                return Err(format!(
                    "node '{}': multi-input/output layers aren't supported yet \
                     (got {} inputs, {} outputs)",
                    node.id,
                    node.inputs.len(),
                    node.outputs.len()
                ));
            }

            let in_name = &node.inputs[0];
            let in_shape = shapes.get(in_name).ok_or_else(|| {
                format!("node '{}' reads undefined tensor '{}'", node.id, in_name)
            })?;

            if *in_shape != node.op.input_shape() {
                return Err(format!(
                    "node '{}': tensor '{}' has shape {:?}, layer expects {:?}",
                    node.id,
                    in_name,
                    in_shape,
                    node.op.input_shape()
                ));
            }

            shapes.insert(node.outputs[0].clone(), node.op.output_shape());
        }

        for spec in &self.outputs {
            match shapes.get(&spec.name) {
                Some(s) if *s == spec.shape => {}
                Some(s) => {
                    return Err(format!(
                        "output '{}': graph produces {:?}, model declares {:?}",
                        spec.name, s, spec.shape
                    ));
                }
                None => return Err(format!("declared output '{}' is never produced", spec.name)),
            }
        }

        Ok(())
    }

    pub fn forward(
        &self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, String> {
        // Sanity-check the caller handed us exactly what the graph declares.
        for spec in &self.inputs {
            let t = inputs
                .get(&spec.name)
                .ok_or_else(|| format!("missing model input '{}'", spec.name))?;
            assert_eq!(
                t.shape(),
                spec.shape,
                "shape mismatch for input '{}'",
                spec.name
            );
        }

        let mut env: HashMap<String, Tensor> = inputs;

        for node in &self.nodes {
            if node.inputs.len() != 1 || node.outputs.len() != 1 {
                return Err(format!(
                    "node '{}': multi-input/output layers aren't supported yet \
                     (got {} inputs, {} outputs)",
                    node.id,
                    node.inputs.len(),
                    node.outputs.len()
                ));
            }

            let in_tensor = env.get(&node.inputs[0]).ok_or_else(|| {
                format!(
                    "node '{}' needs tensor '{}', not yet computed",
                    node.id, node.inputs[0]
                )
            })?;
            let out = node.op.forward(&[in_tensor]);
            env.insert(node.outputs[0].clone(), out);
        }

        self.outputs
            .iter()
            .map(|spec| {
                let t = env
                    .remove(&spec.name)
                    .ok_or_else(|| format!("model output '{}' was never produced", spec.name))?;
                assert_eq!(
                    t.shape(),
                    spec.shape,
                    "Model output shape mismatch for '{}'",
                    spec.name
                );
                Ok((spec.name.clone(), t))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_from_json() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 1 } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "dense1",
                    "inputs": ["input"],
                    "outputs": ["output"],
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
        assert_eq!(model.inputs.len(), 1);
        assert_eq!(model.inputs[0].name, "input");
        assert_eq!(model.inputs[0].shape, TensorShape::Flat(1));
        assert_eq!(model.outputs.len(), 1);
        assert_eq!(model.outputs[0].name, "output");
        assert_eq!(model.outputs[0].shape, TensorShape::Flat(1));
        assert_eq!(model.nodes.len(), 1);
    }

    #[test]
    fn test_model_forward_single_dense() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 1 } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "dense1",
                    "inputs": ["input"],
                    "outputs": ["output"],
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
        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::new(vec![0.5], TensorShape::Flat(1)),
        );
        let outputs = model.forward(inputs).unwrap();
        // (0.5 * 2.0) + 1.0 = 2.0
        assert_eq!(outputs["output"].to_vec(), vec![2.0]);
    }

    #[test]
    fn test_model_forward_multi_layer() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 2 } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "dense1",
                    "inputs": ["input"],
                    "outputs": ["hidden"],
                    "type": "dense",
                    "input_size": 2,
                    "output_size": 2,
                    "weights": [0.5, 0.5, 0.5, 0.5],
                    "bias": [0.0, 0.0]
                },
                {
                    "id": "dense2",
                    "inputs": ["hidden"],
                    "outputs": ["output"],
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
        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::new(vec![1.0, 1.0], TensorShape::Flat(2)),
        );
        let outputs = model.forward(inputs).unwrap();
        // Layer 1: [0.5*1 + 0.5*1, 0.5*1 + 0.5*1] = [1.0, 1.0]
        // Layer 2: [1.0*1 + 1.0*1 + 0.5] = [2.5]
        assert_eq!(outputs["output"].to_vec(), vec![2.5]);
    }

    #[test]
    fn test_model_forward_with_activation() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 2 } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "dense1",
                    "inputs": ["input"],
                    "outputs": ["hidden"],
                    "type": "dense",
                    "input_size": 2,
                    "output_size": 2,
                    "weights": [0.5, 0.5, 0.5, 0.5],
                    "bias": [0.0, 0.0]
                },
                {
                    "id": "relu1",
                    "inputs": ["hidden"],
                    "outputs": ["hidden_act"],
                    "type": "activation",
                    "activation_type": "relu",
                    "shape": { "Flat": 2 }
                },
                {
                    "id": "dense2",
                    "inputs": ["hidden_act"],
                    "outputs": ["output"],
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
        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::new(vec![1.0, 1.0], TensorShape::Flat(2)),
        );
        let outputs = model.forward(inputs).unwrap();
        // Layer 1: [1.0, 1.0]; ReLU: no change; Layer 3: 1+1+0.5 = 2.5
        assert_eq!(outputs["output"].to_vec(), vec![2.5]);
    }

    /// Graph whose input is D3 (a Conv2D node first), goes through Flatten
    /// via a named intermediate tensor, and finishes as a flat output.
    #[test]
    fn test_model_forward_conv_then_flatten() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "D3": { "dim1": 1, "dim2": 2, "dim3": 2 } } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "conv1",
                    "inputs": ["input"],
                    "outputs": ["conv_out"],
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
                    "id": "flatten1",
                    "inputs": ["conv_out"],
                    "outputs": ["output"],
                    "type": "flatten",
                    "shape": { "D3": { "dim1": 1, "dim2": 1, "dim3": 1 } }
                }
            ]
        }
        "#;
        let model = Model::from_json(json).unwrap();
        assert!(model.validate_shapes().is_ok());

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::new(
                vec![1.0, 2.0, 3.0, 4.0],
                TensorShape::D3 {
                    dim1: 1,
                    dim2: 2,
                    dim3: 2,
                },
            ),
        );
        let outputs = model.forward(inputs).unwrap();
        // conv2d sums the whole 2x2 window with an all-ones kernel: 1+2+3+4 = 10
        assert_eq!(outputs["output"].to_vec(), vec![10.0]);
    }

    /// Two independent single-input/single-output chains sharing one
    /// `Model`, each with its own named model input and output. No merge
    /// layer involved - this exercises the actual capability this change
    /// adds at the graph level (multiple named inputs/outputs), without
    /// needing any individual `Layer` to become multi-input.
    #[test]
    fn test_model_forward_multiple_independent_branches() {
        let json = r#"
        {
            "inputs": [
                { "name": "input_a", "shape": { "Flat": 1 } },
                { "name": "input_b", "shape": { "Flat": 1 } }
            ],
            "outputs": [
                { "name": "output_a", "shape": { "Flat": 1 } },
                { "name": "output_b", "shape": { "Flat": 1 } }
            ],
            "nodes": [
                {
                    "id": "branch_a",
                    "inputs": ["input_a"],
                    "outputs": ["output_a"],
                    "type": "dense",
                    "input_size": 1,
                    "output_size": 1,
                    "weights": [2.0],
                    "bias": [0.0]
                },
                {
                    "id": "branch_b",
                    "inputs": ["input_b"],
                    "outputs": ["output_b"],
                    "type": "dense",
                    "input_size": 1,
                    "output_size": 1,
                    "weights": [10.0],
                    "bias": [1.0]
                }
            ]
        }
        "#;
        let model = Model::from_json(json).unwrap();
        assert!(model.validate_shapes().is_ok());

        let mut inputs = HashMap::new();
        inputs.insert(
            "input_a".to_string(),
            Tensor::new(vec![3.0], TensorShape::Flat(1)),
        );
        inputs.insert(
            "input_b".to_string(),
            Tensor::new(vec![3.0], TensorShape::Flat(1)),
        );

        let outputs = model.forward(inputs).unwrap();
        assert_eq!(outputs["output_a"].to_vec(), vec![6.0]); // 3*2 + 0
        assert_eq!(outputs["output_b"].to_vec(), vec![31.0]); // 3*10 + 1
    }

    #[test]
    fn test_model_forward_missing_input_returns_err() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 1 } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": []
        }
        "#;
        let model = Model::from_json(json).unwrap();
        let err = model.forward(HashMap::new()).unwrap_err();
        assert!(err.contains("input"));
    }

    #[test]
    #[should_panic(expected = "shape mismatch for input")]
    fn test_model_forward_wrong_input_shape_panics() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 2 } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "dense1",
                    "inputs": ["input"],
                    "outputs": ["output"],
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
        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::new(vec![1.0], TensorShape::Flat(1)), // wrong size
        );
        let _ = model.forward(inputs);
    }

    #[test]
    #[should_panic(expected = "Model output shape mismatch")]
    fn test_model_forward_wrong_output_shape_panics() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 1 } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 2 } }],
            "nodes": [
                {
                    "id": "dense1",
                    "inputs": ["input"],
                    "outputs": ["output"],
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
        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::new(vec![1.0], TensorShape::Flat(1)),
        );
        let _ = model.forward(inputs);
    }

    #[test]
    fn test_validate_shapes_ok() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 2 } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "dense1",
                    "inputs": ["input"],
                    "outputs": ["output"],
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
    fn test_validate_shapes_catches_mismatched_node_chain() {
        // dense1 outputs size 2, but dense2 expects size 3 - this would
        // previously only be caught mid-forward-pass via a panic.
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 2 } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "dense1",
                    "inputs": ["input"],
                    "outputs": ["hidden"],
                    "type": "dense",
                    "input_size": 2,
                    "output_size": 2,
                    "weights": [1.0, 1.0, 1.0, 1.0],
                    "bias": [0.0, 0.0]
                },
                {
                    "id": "dense2",
                    "inputs": ["hidden"],
                    "outputs": ["output"],
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

    #[test]
    fn test_validate_shapes_catches_undefined_input_tensor() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 1 } }],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "dense1",
                    "inputs": ["typo_input"],
                    "outputs": ["output"],
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
        let err = model.validate_shapes().unwrap_err();
        assert!(err.contains("typo_input"));
    }

    #[test]
    fn test_validate_shapes_catches_output_never_produced() {
        let json = r#"
        {
            "inputs": [{ "name": "input", "shape": { "Flat": 1 } }],
            "outputs": [{ "name": "missing_output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "dense1",
                    "inputs": ["input"],
                    "outputs": ["output"],
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
        let err = model.validate_shapes().unwrap_err();
        assert!(err.contains("missing_output"));
    }

    /// Nodes with more/less than one input or output aren't supported by
    /// any `Layer` variant yet - this is the seam multi-input layers will
    /// widen later. Until then, such a node should fail loudly at
    /// validation time rather than silently dropping extra tensor names.
    #[test]
    fn test_validate_shapes_rejects_multi_input_node() {
        let json = r#"
        {
            "inputs": [
                { "name": "a", "shape": { "Flat": 1 } },
                { "name": "b", "shape": { "Flat": 1 } }
            ],
            "outputs": [{ "name": "output", "shape": { "Flat": 1 } }],
            "nodes": [
                {
                    "id": "merge",
                    "inputs": ["a", "b"],
                    "outputs": ["output"],
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
        let err = model.validate_shapes().unwrap_err();
        assert!(err.contains("multi-input"));
    }
}
