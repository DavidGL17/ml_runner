"""
Benchmark HugeLinearModel and LongLinearModel in PyTorch (locally) and in the
Rust runner (via `cargo run`, across the `default`, `simd`, and `blas`
features), then print a single combined Markdown comparison table.

Flow per model:
  1. Run the PyTorch forward-pass benchmark locally.
  2. Export the model to the Rust runner's JSON format.
  3. For each Rust feature set (default / simd / blas):
       - `cargo run --release [--features <feature>] -- <export.json> <iters> <report.json>`
       - read back <report.json>
  4. Collect every run (1 python + 3 rust) into one list of dict rows.
  5. Print results for the model in md format

At the end, all rows (both models, all backends) are:
  - saved to a combined JSON file (so results persist across executions)

How to run :

# Python-only, no Rust build/run at all
python3 benchmark_models.py 200 --skip-rust

# Only build/run one or two Rust feature sets instead of all three
python3 benchmark_models.py --rust-features simd blas

# Control where exported models/reports and the combined results land
python3 benchmark_models.py --work-dir ./bench_work --results-file ./results/run1.json
"""

import argparse
import json
import random
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.onnx as torch_onnx
from torch import nn

from ml_runner_exporter import export_onnx
from python_fixtures.benchmark_fixtures import HugeLinearModel, LongLinearModel

RUST_FEATURES = ["default", "simd", "blas"]


def fmt_ns(ns: float) -> str:
    """Match the Rust fmt_ns: ms >= 1e6, µs >= 1e3, else ns."""
    if ns >= 1_000_000.0:
        return f"{ns / 1_000_000.0:.3f} ms"
    if ns >= 1_000.0:
        return f"{ns / 1_000.0:.3f} \u00b5s"
    return f"{ns:.1f} ns"


def compute_stats(samples_ns: list[float]) -> dict:
    runs = len(samples_ns)
    return {
        "runs": runs,
        "min_ns": min(samples_ns),
        "max_ns": max(samples_ns),
        "mean_ns": statistics.mean(samples_ns),
        "median_ns": statistics.median(samples_ns),
        # population stddev to match the Rust implementation, which divides
        # the sum of squared deviations by `runs` (not `runs - 1`)
        "std_dev_ns": statistics.pstdev(samples_ns),
    }


def backend_name() -> str:
    return f"PyTorch {torch.__version__} (CPU, {torch.get_num_threads()} threads)"


def benchmark_model_local(model: nn.Module, input_dim: int, iterations: int, seed: int, name: str) -> dict:
    """Run the PyTorch forward-pass benchmark and return one result row."""
    warmup = min(10, max(iterations - 1, 0))
    gen = torch.Generator(device="cpu").manual_seed(seed)

    model.eval()
    samples_ns: list[float] = []
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
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"[python/{name}] Error during forward pass (iteration {i}): {e}")
                continue

            if output_shape is None:
                output_shape = list(out.shape)

            if i >= warmup:
                samples_ns.append(float(elapsed))

    row = {
        "runner": "python",
        "backend": backend_name(),
        "input_shape": [1, input_dim],
        "output_shape": output_shape,
        "warmup": warmup,
        "errors": errors,
    }

    if not samples_ns:
        row["runs"] = 0
        row["error_message"] = f"No successful forward passes to report on (errors: {errors})."
        return row

    row.update(compute_stats(samples_ns))
    return row


def export_and_run_model(model: nn.Module, input_shape: Any, onnx_path: Path, export_path: Path) -> Path:
    """Export a PyTorch model to ONNX, then convert it to the Rust runner's
    JSON model format at `export_path`. Returns `export_path`."""
    if isinstance(input_shape, tuple) and len(input_shape) == 3:
        dummy_input_data = torch.randn(1, *input_shape)
    elif isinstance(input_shape, tuple) and len(input_shape) == 2:
        seq_len, feature_size = input_shape
        dummy_input_data = torch.randn(seq_len, 1, feature_size)
    else:
        dummy_input_data = torch.randn(1, input_shape)

    model.eval()
    torch_onnx.export(
        model,
        dummy_input_data,  # type: ignore[arg-type]
        str(onnx_path),
        export_params=True,
        opset_version=17,
        dynamo=False,
    )
    output_model = export_onnx(str(onnx_path))
    with open(export_path, "w") as f:
        json.dump(output_model, f, indent=2)

    return export_path


def run_rust_benchmark(rust_dir: Path, export_path: Path, iterations: int, feature: str, report_path: Path) -> dict:
    """Build+run the Rust runner for a single feature set and return the
    parsed JSON report as one result row. On failure, returns a row carrying
    the error instead of raising, so one bad feature build doesn't kill the
    rest of the comparison."""
    cmd = ["cargo", "run", "--release"]
    if feature != "default":
        cmd += ["--features", feature]
    cmd += ["--", str(export_path.resolve()), str(iterations), str(report_path.resolve())]

    try:
        result = subprocess.run(cmd, cwd=rust_dir, capture_output=True, text=True, check=True)
    except OSError as e:
        # Covers a missing --rust-dir and a missing `cargo` executable alike.
        return {
            "runner": "rust",
            "backend": feature,
            "runs": 0,
            "errors": None,
            "error_message": f"failed to launch cargo in '{rust_dir}': {e}",
        }

    if result.returncode != 0:
        return {
            "runner": "rust",
            "backend": feature,
            "runs": 0,
            "errors": None,
            "error_message": f"cargo run failed (exit {result.returncode}): {result.stderr.strip()[-500:]}",
        }

    if not report_path.exists():
        return {
            "runner": "rust",
            "backend": feature,
            "runs": 0,
            "errors": None,
            "error_message": f"cargo run succeeded but '{report_path}' was not written. stdout tail: {result.stdout.strip()[-500:]}",
        }

    with open(report_path) as f:
        report = json.load(f)

    return {
        "runner": "rust",
        "backend": report.get("backend", feature),
        "input_shape": report.get("input_shape"),
        "output_shape": report.get("output_shape"),
        "runs": report.get("runs"),
        "warmup": report.get("warmup"),
        "errors": report.get("errors"),
        "min_ns": report.get("min_ns"),
        "max_ns": report.get("max_ns"),
        "mean_ns": report.get("mean_ns"),
        "median_ns": report.get("median_ns"),
        "std_dev_ns": report.get("std_dev_ns"),
    }


def print_results(result: dict) -> str:
    header = "| Runner | Backend | Runs | Min | Max | Mean | Median | StdDev | Errors |"
    sep = "|:--------:|:---------:|:------:|:-----:|:-----:|:------:|:--------:|:-------:|:--------:|"
    lines = [header, sep]

    output = "\n"
    output += f"## Results for model : {result['model']}\n"

    for row in result["results"]:
        if not row.get("runs"):
            lines.append(f"| {row.get('runner', '?')} | {row.get('backend', '?')} | 0 | - | - | - | - | - | {row.get('error_message', 'no successful runs')} |")
            continue

        lines.append(
            f"| {row['runner']} | {row['backend']} | {row['runs']} "
            f"| {fmt_ns(row['min_ns'])} | {fmt_ns(row['max_ns'])} | {fmt_ns(row['mean_ns'])} "
            f"| {fmt_ns(row['median_ns'])} | {fmt_ns(row['std_dev_ns'])} | {row.get('errors', 0)} |"
        )
    return output + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark HugeLinearModel/LongLinearModel in PyTorch and in the Rust runner "
        "(default/simd/blas), and print one combined comparison table."
    )
    parser.add_argument("iterations", nargs="?", type=int, default=1000, help="Timed forward passes per run (default: 1000)")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="Base seed for input generation and layer sizing")
    parser.add_argument(
        "--rust-dir",
        type=Path,
        default=".",
        help="Path to the Rust runner's Cargo project directory (containing Cargo.toml)",
    )
    parser.add_argument(
        "--rust-features",
        nargs="+",
        default=RUST_FEATURES,
        help=f"Rust feature sets to build+run (default: {RUST_FEATURES})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("bench_work"),
        help="Scratch directory for exported models and JSON reports (default: ./bench_work)",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("combined_benchmark_results.json"),
        help="Where to save the combined results as JSON (default: ./combined_benchmark_results.json)",
    )
    parser.add_argument("--skip-rust", action="store_true", help="Only run the local PyTorch benchmarks")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    args.work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Iterations per run: {args.iterations} | seed: {args.seed}")
    print(f"PyTorch backend: {backend_name()}")
    if not args.skip_rust:
        print(f"Rust runner: {args.rust_dir} | features: {args.rust_features}")
    print()

    models = [
        ("HugeLinearModel", HugeLinearModel()),  # type: ignore[no-untyped-call]
        ("LongLinearModel", LongLinearModel([random.randint(100, 1000) for _ in range(30)])),  # noqa : S311
    ]

    results = []
    output_to_print = []

    for idx, (name, model) in enumerate(models):
        input_dim = model.get_input_dims()  # type: ignore[operator]
        seed = args.seed + idx
        rows = []

        # 1. Local PyTorch run.
        print(f"Running local PyTorch benchmark: {name}...")
        local_row = benchmark_model_local(model, input_dim=input_dim, iterations=args.iterations, seed=seed, name=name)
        rows.append(local_row)

        if args.skip_rust:
            continue

        # 2. Export to the Rust runner's JSON model format.
        onnx_path = args.work_dir / f"{name}.onnx"
        export_path = args.work_dir / f"{name}.json"
        print(f"Exporting {name} -> {export_path}...")
        export_and_run_model(model, input_dim, onnx_path=onnx_path, export_path=export_path)

        # 3. Run the Rust runner once per feature set.
        for feature in args.rust_features:
            report_path = args.work_dir / f"{name}_{feature}.report.json"
            print(f"Running Rust benchmark: {name} [{feature}]...")
            rust_row = run_rust_benchmark(
                rust_dir=args.rust_dir,
                export_path=export_path,
                iterations=args.iterations,
                feature=feature,
                report_path=report_path,
            )
            # Rust doesn't know the friendly model name (it only sees the
            # exported file path) - stamp it with ours for the combined table.
            rust_row["model"] = name
            rows.append(rust_row)

        result = {
            "model": name,
            "results": rows,
        }
        output_to_print.append(print_results(result))
        results.append(result)

    for output in output_to_print:
        print(output)

    with open(args.results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "iterations": args.iterations,
                "seed": args.seed,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved combined results to '{args.results_file}'.")


if __name__ == "__main__":
    main()
