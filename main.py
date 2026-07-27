from c_exporter.onnx_exporter import export_onnx

# Load the ONNX model
model_path = "simple_linear_model.onnx"
output_path = "export.json"

export_onnx(model_path, output_path)
print(f"Model exported to {output_path}")
