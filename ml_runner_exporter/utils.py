def onnx_shape_to_tensor_shape(shape: tuple) -> dict:
    """Convert an ONNX (N, ...) shape into TensorShape's serde JSON representation.

    Drops the leading batch dimension. A single remaining dim becomes
    {"Flat": n}; three remaining dims (C, H, W) become
    {"D3": {"dim1": .., "dim2": .., "dim3": ..}}.
    """
    dims = tuple(shape[1:])

    if len(dims) == 1:
        return {"Flat": dims[0]}
    elif len(dims) == 3:
        dim1, dim2, dim3 = dims
        return {"D3": {"dim1": dim1, "dim2": dim2, "dim3": dim3}}
    else:
        raise ValueError(
            f"Unsupported model input/output shape {shape}: expected a single " "feature dim (N, F) or a D3 dim (N, C, H, W) after the batch dimension"
        )
