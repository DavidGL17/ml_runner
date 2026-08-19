from .layer import LayerParser


def export_model(model_layers: list[LayerParser], input_shape: dict, output_shape: dict) -> dict:
    """Build the JSON-serializable dict describing the whole model, matching the `Model` struct the Rust runtime deserializes via `Model::from_json`.

    `input_shape`/`output_shape` must already be in TensorShape's serde
    representation, e.g. {"Flat": 10} or
    {"D3": {"dim1": 3, "dim2": 4, "dim3": 5}}.
    """
    return {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "layers": [layer.to_dict() for layer in model_layers],
    }
