from .layer import LayerParser


def export_model(model_layers: list[LayerParser], input_size: int, output_size: int) -> dict:
    """Build the JSON-serializable dict describing the whole model, matching the `Model` struct the Rust runtime deserializes via `Model::from_json`."""
    return {
        "input_size": input_size,
        "output_size": output_size,
        "layers": [layer.to_dict() for layer in model_layers],
    }
