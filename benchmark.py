"""
Benchmark HugeLinearModel and LongLinearModel, reporting results in Markdown.

Mirrors the methodology of the companion Rust benchmark:
  - a handful of warmup iterations are discarded (min(10, iterations - 1))
  - only the forward pass itself is timed (input generation happens outside
    the timed region)
  - stats reported: min / max / mean / median / population stddev
  - output is a Markdown report per model, in the same shape as the Rust one
"""

import argparse
import random
import statistics
import time
import torch
import torch.nn as nn
import torch.onnx as torch_onnx
import json

from python_fixtures.benchmark_fixtures import HugeLinearModel, LongLinearModel
from ml_runner_exporter import export_onnx

TMP_ONNX_MODEL_PATH = "tmp_model.onnx"
TMP_EXPORT_PATH = "tmp_export.json"


def fmt_ns(ns: float) -> str:
    """Match the Rust fmt_ns: ms >= 1e6, µs >= 1e3, else ns."""
    if ns >= 1_000_000.0:
        return f"{ns / 1_000_000.0:.3f} ms"
    elif ns >= 1_000.0:
        return f"{ns / 1_000.0:.3f} \u00b5s"
    else:
        return f"{ns:.1f} ns"


def compute_stats(samples_ns):
    runs = len(samples_ns)
    return {
        "runs": runs,
        "min": min(samples_ns),
        "max": max(samples_ns),
        "mean": statistics.mean(samples_ns),
        "median": statistics.median(samples_ns),
        # population stddev to match the Rust implementation, which divides
        # the sum of squared deviations by `runs` (not `runs - 1`)
        "std_dev": statistics.pstdev(samples_ns),
    }


def backend_name() -> str:
    return f"PyTorch {torch.__version__} (CPU, {torch.get_num_threads()} threads)"


def benchmark_model(model: nn.Module, input_dim: int, iterations: int, seed: int, name: str) -> str:
    warmup = min(10, max(iterations - 1, 0))

    gen = torch.Generator(device="cpu").manual_seed(seed)

    model.eval()
    samples_ns = []
    errors = 0
    output_shape = None

    with torch.no_grad():
        for i in range(iterations):
            # Input generation happens outside the timed region, matching
            # the Rust benchmark (random_tensor() is called before Instant::now()).
            x = torch.rand((1, input_dim), generator=gen) * 2.0 - 1.0

            start = time.perf_counter_ns()
            try:
                out = model(x)
                elapsed = time.perf_counter_ns() - start
            except Exception as e:
                errors += 1
                print(f"Error during forward pass (iteration {i}): {e}")
                continue

            if output_shape is None:
                output_shape = tuple(out.shape)

            if i >= warmup:
                samples_ns.append(float(elapsed))

    lines = [f"## Benchmark Report: {name}", ""]

    if not samples_ns:
        lines.append(f"No successful forward passes to report on (errors: {errors}).")
        return "\n".join(lines)

    stats = compute_stats(samples_ns)

    lines.append(f"- **Model:** `{name}`")
    lines.append(f"- **Backend:** {backend_name()}")
    lines.append(f"- **Input shape:** [1, {input_dim}]")
    lines.append(f"- **Output shape:** {list(output_shape)}")
    lines.append(f"- **Successful runs:** {stats['runs']} (warmup discarded: {warmup}, errors: {errors})")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Min    | {fmt_ns(stats['min'])} |")
    lines.append(f"| Max    | {fmt_ns(stats['max'])} |")
    lines.append(f"| Mean   | {fmt_ns(stats['mean'])} |")
    lines.append(f"| Median | {fmt_ns(stats['median'])} |")
    lines.append(f"| StdDev | {fmt_ns(stats['std_dev'])} |")
    lines.append("")

    return "\n".join(lines)


def export_and_run_model(model: nn.Module, input_shape: int) -> str:
    if isinstance(input_shape, tuple) and len(input_shape) == 3:
        dummy_input_data = torch.randn(1, *input_shape)
    elif isinstance(input_shape, tuple) and len(input_shape) == 2:
        seq_len, feature_size = input_shape
        dummy_input_data = torch.randn(seq_len, 1, feature_size)
    else:
        dummy_input_data = torch.randn(1, input_shape)
    model.eval()
    torch_onnx.export(model, dummy_input_data, TMP_ONNX_MODEL_PATH, export_params=True, opset_version=17, dynamo=False)
    output_model = export_onnx(TMP_ONNX_MODEL_PATH)
    with open(TMP_EXPORT_PATH, "w") as f:
        json.dump(output_model, f, indent=2)

    return ""


def main():
    parser = argparse.ArgumentParser(description="Benchmark HugeLinearModel and LongLinearModel, output in Markdown.")
    parser.add_argument("iterations", nargs="?", type=int, default=1000, help="Number of timed forward passes (default: 1000)")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="Base seed for input generation and layer sizing")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)  # seeds LongLinearModel's random.randint layer sizing indirectly via `random`
    random.seed(args.seed)

    warmup_preview = min(10, max(args.iterations - 1, 0))
    print(f"Running {args.iterations} iterations per model " f"({warmup_preview} warmup, discarded) on backend: {backend_name()}\n")

    huge = HugeLinearModel()
    print(benchmark_model(huge, input_dim=huge.get_input_dims(), iterations=args.iterations, seed=args.seed, name="HugeLinearModel"))

    long_model = LongLinearModel()
    print(benchmark_model(long_model, input_dim=long_model.get_input_dims(), iterations=args.iterations, seed=args.seed + 1, name="LongLinearModel"))


if __name__ == "__main__":
    main()
