def dims_to_tensor_shape(dims: tuple) -> dict:
    """Convert a (already batch-stripped) dims tuple into TensorShape's serde JSON representation.

    A single dim becomes {"Flat": n}; two dims (e.g. seq_len, features) become
    {"D2": {"dim1": .., "dim2": ..}}; three dims (e.g. C, H, W) become
    {"D3": {"dim1": .., "dim2": .., "dim3": ..}}.
    """
    if len(dims) == 1:
        return {"Flat": dims[0]}
    if len(dims) == 2:
        dim1, dim2 = dims
        return {"D2": {"dim1": dim1, "dim2": dim2}}
    if len(dims) == 3:
        dim1, dim2, dim3 = dims
        return {"D3": {"dim1": dim1, "dim2": dim2, "dim3": dim3}}
    m = f"Unsupported shape {dims}: expected 1, 2, or 3 dims (Flat, D2, or D3) after dropping the leading dimension(s)"
    raise ValueError(m)
