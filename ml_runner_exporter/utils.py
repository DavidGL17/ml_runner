def onnx_shape_to_tensor_shape(shape: tuple) -> dict:
    """Convert an ONNX (N, ...) shape into TensorShape's serde JSON representation.

    Drops the leading batch dimension. A single remaining dim becomes
    {"Flat": n}; two remaining dims (e.g. seq_len, features) become
    {"D2": {"dim1": .., "dim2": ..}}; three remaining dims (e.g. C, H, W)
    become {"D3": {"dim1": .., "dim2": .., "dim3": ..}}.
    """
    dims = tuple(shape[1:])

    if len(dims) == 1:
        return {"Flat": dims[0]}
    elif len(dims) == 2:
        dim1, dim2 = dims
        return {"D2": {"dim1": dim1, "dim2": dim2}}
    elif len(dims) == 3:
        dim1, dim2, dim3 = dims
        return {"D3": {"dim1": dim1, "dim2": dim2, "dim3": dim3}}
    else:
        raise ValueError(
            f"Unsupported model input/output shape {shape}: expected a single "
            "feature dim (N, F), a D2 dim (N, D1, D2), or a D3 dim "
            "(N, D1, D2, D3) after the batch dimension"
        )
