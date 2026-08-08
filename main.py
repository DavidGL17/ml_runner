import json
import argparse

from ml_runner_exporter.onnx_exporter import export_onnx

# Set up argument parsing
parser = argparse.ArgumentParser(description="Export ONNX model to JSON")
parser.add_argument("--model-path", type=str, required=True, help="Path to the input ONNX model")
parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSON file")

args = parser.parse_args()

# Load the ONNX model
model_path = args.model_path
output_path = args.output_path

output_model = export_onnx(model_path)
with open(output_path, "w") as f:
    json.dump(output_model, f, indent=2)
print(f"Model exported to {output_path}")
