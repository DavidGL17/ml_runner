import json

from c_exporter.onnx_exporter import export_onnx

# Load the ONNX model
model_path = "simple_linear_model.onnx"
output_path = "export.json"

output_model = export_onnx(model_path, output_path)
with open(output_path, "w") as f:
    json.dump(output_model, f, indent=2)
print(f"Model exported to {output_path}")
