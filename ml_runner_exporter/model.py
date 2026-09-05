def export_model(nodes: list[dict], inputs: list[dict], outputs: list[dict]) -> dict:
    """Build the JSON-serializable dict describing the whole model graph,
    matching the `Model` struct the Rust runtime deserializes via
    `Model::from_json`.

    `inputs`/`outputs` are lists of IoSpec dicts: {"name": ..., "shape": ...},
    where `shape` is already in TensorShape's serde representation (e.g.
    {"Flat": 10} or {"D3": {"dim1": 3, "dim2": 4, "dim3": 5}}).

    `nodes` are already-built graph node dicts (see `_build_node` in
    `onnx_exporter.py`): each has "id", "inputs", "outputs", plus whatever
    a `LayerParser.to_dict()` contributes (the "type" tag and the layer's
    own fields), matching the `Node` struct on the Rust side.
    """
    return {
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
    }
