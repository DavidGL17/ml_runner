import onnx
from onnx import numpy_helper
from c_exporter.c_exporter import export_model
from c_exporter.utils import Layer, LayerTypes

# Load the ONNX model
model_path = "simple_model.onnx"
model = onnx.load(model_path)

# Check model validity
onnx.checker.check_model(model)
graph = model.graph

print("=" * 60)
print(f"Model: {graph.name}  |  IR version: {model.ir_version}")
print("=" * 60)

# --- Inputs & Outputs ---
in_shape: int = 0
out_shape: int = 0
print("\nInputs:")
for inp in graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {shape}")
    in_shape = shape[1] if len(shape) > 1 else 1

print("\nOutputs:")
for out in graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {shape}")
    out_shape = shape[1] if len(shape) > 1 else 1

# --- Build a weights lookup for quick access ---
weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

# --- Iterate over layers (nodes) ---
print(f"\nLayers ({len(graph.node)} total):\n")
layers = []
for i, node in enumerate(graph.node):
    print(f"[{i:02d}] {node.op_type:20s} | name: {node.name or '(unnamed)'}")
    print(f"      inputs : {list(node.input)}")
    print(f"      outputs: {list(node.output)}")

    # Print attributes (e.g. kernel_shape, strides, etc.)
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            val = attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            val = attr.f
        elif attr.type == onnx.AttributeProto.INTS:
            val = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            val = list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRING:
            val = attr.s.decode()
        else:
            val = "(complex type)"
        print(f"      attr   : {attr.name} = {val}")

    # Print shapes of any learnable weights attached to this node
    for inp_name in node.input:
        if inp_name in weights:
            w = weights[inp_name]
            print(f"      weight : {inp_name} → shape {w.shape}, dtype {w.dtype}")

    # Extract weight and bias tensors from the node's inputs
    node_weights = [weights[inp] for inp in node.input if inp in weights]

    weight_matrix = node_weights[0] if len(node_weights) > 0 else None
    bias_vector = node_weights[1] if len(node_weights) > 1 else None

    # Derive input/output sizes from the weight matrix shape
    # Gemm/MatMul weight shape: (out_features, in_features)
    input_size = weight_matrix.shape[1] if weight_matrix is not None else None
    output_size = weight_matrix.shape[0] if weight_matrix is not None else None

    layer = Layer(
        layer_num=i,
        layerType=LayerTypes.from_string(node.op_type),
        input_size=input_size,
        output_size=output_size,
        weights=weight_matrix.T.tolist() if weight_matrix is not None else [],
        bias=bias_vector.T.tolist() if bias_vector is not None else [],
    )
    layers.append(layer)

    print()

model_export = export_model(layers, in_shape, out_shape)

with open("src/model_weights.h", "w") as f:
    f.write(model_export)
