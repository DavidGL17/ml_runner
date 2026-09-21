"""
Benchmark HugeLinearModel and LongLinearModel in PyTorch (locally) and in the
Rust runner (via `cargo run`, across the `default`, `simd`, and `blas`
features), then print a single combined Markdown comparison table.

Optionally, the same PyTorch + Rust benchmarks can also be run on one or more
remote devices over SSH (e.g. a Raspberry Pi) via `--device`. This is generic:
any host you can reach with a plain `ssh <host>` (no interactive login, e.g. an
SSH key + `~/.ssh/config` alias) can be plugged in.

Flow per model:
  1. Run the PyTorch forward-pass benchmark locally.
  2. Export the model to the Rust runner's JSON format.
  3. For each Rust feature set (default / simd / blas):
       - `cargo run --release [--features <feature>] -- <export.json> <iters> <report.json>`
       - read back <report.json>
  4. For each configured --device:
       - Run the PyTorch benchmark on the device (by re-invoking this same
         script remotely with --skip-rust) and pull back that model's row.
       - scp the exported model JSON to the device and run the Rust runner
         there for each of the device's feature sets, pulling back the report.
  5. Collect every run (python, rust x features, and remote equivalents per
     device) into one list of dict rows.
  6. Print results for the model in md format.

At the end, all rows (all models, all backends, all devices) are:
  - saved to a combined JSON file (so results persist across executions)

How to run :

# Python-only, no Rust build/run at all
python3 benchmark.py 200 --skip-rust

# Only build/run one or two Rust feature sets instead of all three
python3 benchmark.py --rust-features simd blas

# Control where exported models/reports and the combined results land
python3 benchmark.py --work-dir ./bench_work --results-file ./results/run1.json

# Also benchmark on a Raspberry Pi reachable as `ssh pi4` (no login prompt),
# with the project rsync'd to ~/ml-runner on the device
python3 benchmark.py --device "name=pi4,host=pi4,dir=/home/pi/ml-runner"

# Same, but skip blas on the Pi and don't re-sync the project each run
python3 benchmark.py \
    --device "name=pi4,host=pi4,dir=/home/pi/ml-runner,features=default;simd,sync=0"

# Multiple devices at once, each with their own overrides
python3 benchmark.py \
    --device "name=pi4,host=pi4,dir=/home/pi/ml-runner" \
    --device "name=jetson,host=jetson-nano,dir=/home/jetson/ml-runner,python=python3.10"
"""

import argparse
import json
import random
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.onnx as torch_onnx
from torch import nn

from ml_runner_exporter import export_onnx
from python_fixtures.benchmark_fixtures import HugeLinearModel, LongLinearModel

RUST_FEATURES = ["default", "simd", "blas"]

# ssh/scp options shared by every remote invocation: fail fast instead of
# hanging on a password prompt, and don't wait forever for a dead host.
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


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


# --------------------------------------------------------------------------
# Remote (SSH) device support
#
# A "device" is anything reachable with a plain, non-interactive
# `ssh <host> <command>` (key-based auth, or a ~/.ssh/config Host alias that
# needs no password) - a Raspberry Pi, another workstation, a cloud box, etc.
#
# The project directory is rsync'd to the device, then:
#   - the PyTorch benchmark is run by re-invoking *this same script* on the
#     device with --skip-rust, and reading back its results file.
#   - the Rust benchmark is run by scp-ing the exported model JSON over and
#     invoking `cargo run` on the device, then scp-ing the report back.
# --------------------------------------------------------------------------


@dataclass
class RemoteDevice:
    name: str
    host: str  # ssh destination: hostname, IP, or ~/.ssh/config alias
    remote_dir: str  # project root on the device (contains this script + the Rust crate)
    python_bin: str = "python3"
    rust_subdir: str = "."  # Cargo project dir, relative to remote_dir
    rust_features: list[str] | None = None  # None => reuse the top-level --rust-features
    sync: bool = True  # rsync the local project to remote_dir before running
    skip_python: bool = False
    skip_rust: bool = False


_DEVICE_SPEC_REQUIRED_FIELDS = ("name", "host", "dir")
_DEVICE_SPEC_OPTIONAL_FIELDS = ("python", "rust-dir", "features", "sync", "skip-python", "skip-rust")
_DEVICE_SPEC_ALL_FIELDS = _DEVICE_SPEC_REQUIRED_FIELDS + _DEVICE_SPEC_OPTIONAL_FIELDS
_DEVICE_SPEC_BOOL_TRUE = {"1", "true", "yes"}
_DEVICE_SPEC_BOOL_FALSE = {"0", "false", "no"}


def parse_device_spec(spec: str) -> RemoteDevice:
    """Parse a --device SPEC string of the form:

    'name=pi4,host=pi4,dir=/home/pi/ml-runner[,python=python3][,rust-dir=.]
     [,features=default;simd][,sync=1][,skip-python=0][,skip-rust=0]'

    Only name, host, and dir are required; everything else has a default.
    Unknown keys, empty required values, and unrecognized boolean values are
    all rejected up front rather than silently ignored/misinterpreted -
    a typo here (e.g. 'dirs=' instead of 'dir=', or a wrong path) is much
    easier to debug as an immediate error than as a confusing failure deep
    into a benchmark run.
    """
    fields: dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"invalid --device field '{chunk}' in spec '{spec}' (expected key=value)")
        key, value = chunk.split("=", 1)
        fields[key.strip().lower()] = value.strip()

    unknown = [k for k in fields if k not in _DEVICE_SPEC_ALL_FIELDS]
    if unknown:
        raise ValueError(f"--device spec '{spec}' has unrecognized field(s): {', '.join(unknown)}. " f"Recognized fields: {', '.join(_DEVICE_SPEC_ALL_FIELDS)}")

    missing = [k for k in _DEVICE_SPEC_REQUIRED_FIELDS if not fields.get(k)]
    if missing:
        raise ValueError(f"--device spec '{spec}' is missing required field(s): {', '.join(missing)}")

    def as_bool(field_name: str, v: str) -> bool:
        normalized = v.strip().lower()
        if normalized in _DEVICE_SPEC_BOOL_TRUE:
            return True
        if normalized in _DEVICE_SPEC_BOOL_FALSE:
            return False
        raise ValueError(
            f"--device spec '{spec}' has invalid value '{v}' for '{field_name}' "
            f"(expected one of: {', '.join(sorted(_DEVICE_SPEC_BOOL_TRUE | _DEVICE_SPEC_BOOL_FALSE))})"
        )

    features = fields.get("features")
    return RemoteDevice(
        name=fields["name"],
        host=fields["host"],
        remote_dir=fields["dir"].rstrip("/"),
        python_bin=fields.get("python", "python3"),
        rust_subdir=fields.get("rust-dir", "."),
        rust_features=[f for f in features.split(";") if f] if features else None,
        sync=as_bool("sync", fields.get("sync", "1")),
        skip_python=as_bool("skip-python", fields.get("skip-python", "0")),
        skip_rust=as_bool("skip-rust", fields.get("skip-rust", "0")),
    )


def run_remote_cmd(host: str, cwd: str, command: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    full_cmd = f"cd {shlex.quote(cwd)} && {command}"
    return subprocess.run(["ssh", *SSH_OPTS, host, full_cmd], capture_output=True, text=True, timeout=timeout)


def ensure_remote_dir(host: str, path: str, timeout: float | None = 30) -> None:
    subprocess.run(
        ["ssh", *SSH_OPTS, host, f"mkdir -p {shlex.quote(path)}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def scp_to_remote(local_path: Path, host: str, remote_path: str, timeout: float | None = None) -> None:
    subprocess.run(
        ["scp", *SSH_OPTS, str(local_path), f"{host}:{remote_path}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def scp_from_remote(host: str, remote_path: str, local_path: Path, timeout: float | None = None) -> None:
    subprocess.run(
        ["scp", *SSH_OPTS, f"{host}:{remote_path}", str(local_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def sync_project_to_remote(local_root: Path, device: RemoteDevice, timeout: float | None = None) -> None:
    """rsync the local project (this script, its Python deps, and the Rust
    crate) to device.remote_dir, so the device can run both benchmarks."""
    ensure_remote_dir(device.host, device.remote_dir, timeout=timeout)
    ssh_cmd = " ".join(["ssh", *SSH_OPTS])
    cmd = [
        "rsync",
        "-az",
        "--delete",
        "-e",
        ssh_cmd,
        "--exclude",
        ".git",
        "--exclude",
        "target",
        "--exclude",
        "__pycache__",
        "--exclude",
        "bench_work",
        "--exclude",
        "*.onnx",
        f"{local_root}/",
        f"{device.host}:{device.remote_dir}/",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)


def benchmark_model_remote_python(
    device: RemoteDevice, model_name: str, iterations: int, seed: int, local_work_dir: Path, timeout: float | None = None
) -> dict:
    """Run the PyTorch benchmark for one model on a remote device, by
    re-invoking this same script there with --skip-rust, then scp-ing the
    results file back and pulling out that model's python row."""
    remote_scratch = f"{device.remote_dir}/_bench_scratch"
    try:
        ensure_remote_dir(device.host, remote_scratch, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        return _remote_error_row(f"python@{device.name}", device.name, f"could not prepare remote scratch dir: {_err_tail(e)}")

    remote_results = f"{remote_scratch}/py_only_{model_name}.json"
    cmd = f"{device.python_bin} benchmark.py {iterations} --seed {seed} " f"--skip-rust --results-file {shlex.quote(remote_results)}"

    try:
        result = run_remote_cmd(device.host, device.remote_dir, cmd, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return _remote_error_row(f"python@{device.name}", device.name, f"failed to run remote python benchmark: {_err_tail(e)}")

    if result.returncode != 0:
        return _remote_error_row(
            f"python@{device.name}", device.name, f"remote python benchmark failed (exit {result.returncode}): {result.stderr.strip()[-500:]}"
        )

    local_report = local_work_dir / f"py_only_{model_name}_{device.name}.json"
    try:
        scp_from_remote(device.host, remote_results, local_report, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        return _remote_error_row(f"python@{device.name}", device.name, f"failed to fetch remote results: {_err_tail(e)}")

    with open(local_report) as f:
        data = json.load(f)

    for model_result in data.get("results", []):
        if model_result.get("model") != model_name:
            continue
        for row in model_result.get("results", []):
            if row.get("runner") == "python":
                row = dict(row)
                row["runner"] = f"python@{device.name}"
                return row

    return _remote_error_row(f"python@{device.name}", device.name, "remote results file did not contain a python row for this model")


def run_rust_benchmark_remote(
    device: RemoteDevice,
    model_name: str,
    export_path: Path,
    iterations: int,
    feature: str,
    local_work_dir: Path,
    timeout: float | None = None,
) -> dict:
    """scp the exported model JSON to the device, run `cargo run` for one
    feature set there, and scp the report back."""
    remote_rust_dir = f"{device.remote_dir}/{device.rust_subdir}".rstrip("/")
    remote_scratch = f"{device.remote_dir}/_bench_scratch"

    try:
        ensure_remote_dir(device.host, remote_scratch, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        return _remote_error_row(f"rust@{device.name}", feature, f"could not prepare remote scratch dir: {_err_tail(e)}")

    remote_export = f"{remote_scratch}/{export_path.name}"
    remote_report = f"{remote_scratch}/{model_name}_{feature}.report.json"

    try:
        scp_to_remote(export_path, device.host, remote_export, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        return _remote_error_row(f"rust@{device.name}", feature, f"failed to upload exported model: {_err_tail(e)}")

    cmd = "cargo run --release"
    if feature != "default":
        cmd += f" --features {feature}"
    cmd += f" -- {shlex.quote(remote_export)} {iterations} {shlex.quote(remote_report)}"

    try:
        result = run_remote_cmd(device.host, remote_rust_dir, cmd, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return _remote_error_row(f"rust@{device.name}", feature, f"failed to run remote cargo build/run: {_err_tail(e)}")

    if result.returncode != 0:
        return _remote_error_row(f"rust@{device.name}", feature, f"remote cargo run failed (exit {result.returncode}): {result.stderr.strip()[-500:]}")

    local_report = local_work_dir / f"{model_name}_{feature}_{device.name}.report.json"
    try:
        scp_from_remote(device.host, remote_report, local_report, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        return _remote_error_row(f"rust@{device.name}", feature, f"failed to fetch remote report: {_err_tail(e)}")

    with open(local_report) as f:
        report = json.load(f)

    return {
        "runner": f"rust@{device.name}",
        "backend": report.get("backend", f"{feature}@{device.name}"),
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


def _err_tail(e: Exception, n: int = 500) -> str:
    stderr = getattr(e, "stderr", None)
    if stderr:
        return str(stderr).strip()[-n:]
    return str(e)[-n:]


def _remote_error_row(runner: str, backend: str, message: str) -> dict:
    return {"runner": runner, "backend": backend, "runs": 0, "errors": None, "error_message": message}


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
        "(default/simd/blas), locally and optionally on remote SSH devices, and print one "
        "combined comparison table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help=f"Rust feature sets to build+run locally (default: {RUST_FEATURES})",
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
    parser.add_argument(
        "--device",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Also benchmark on a remote device reachable via a plain, non-interactive "
            "`ssh <host>` (e.g. an ~/.ssh/config alias like 'pi4'). Repeatable. SPEC is a "
            "comma-separated key=value list: name=<label>,host=<ssh host/alias>,dir=<remote "
            "project dir>[,python=<python bin, default python3>][,rust-dir=<Cargo project dir "
            "relative to 'dir', default '.'>][,features=<semicolon-separated feature list, "
            "default: same as --rust-features>][,sync=0|1 (default 1: rsync the local project "
            "to the device first)][,skip-python=0|1][,skip-rust=0|1]. Example: "
            '--device "name=pi4,host=pi4,dir=/home/pi/ml-runner"'
        ),
    )
    parser.add_argument(
        "--local-project-dir",
        type=Path,
        default=None,
        help="Local project directory to rsync to each --device (default: this script's directory)",
    )
    parser.add_argument(
        "--device-timeout",
        type=float,
        default=1800,
        help="Timeout in seconds for each remote ssh/scp/rsync operation (default: 1800)",
    )
    args = parser.parse_args()

    try:
        devices = [parse_device_spec(spec) for spec in args.device]
    except ValueError as e:
        parser.error(str(e))
        return  # unreachable, parser.error() exits

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    args.work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Iterations per run: {args.iterations} | seed: {args.seed}")
    print(f"PyTorch backend: {backend_name()}")
    if not args.skip_rust:
        print(f"Rust runner: {args.rust_dir} | features: {args.rust_features}")
    if devices:
        print(f"Devices: {[d.name for d in devices]}")
    print()

    # Sync each device once up front (not once per model) so we don't rsync
    # the whole project twice for two models. A device that fails to sync is
    # marked broken and every row for it becomes an error row, rather than
    # aborting the whole benchmark run.
    local_project_dir = args.local_project_dir or Path(__file__).resolve().parent
    device_errors: dict[str, str] = {}
    for device in devices:
        if not device.sync:
            continue
        print(f"Syncing project to device '{device.name}' ({device.host}:{device.remote_dir})...")
        try:
            sync_project_to_remote(local_project_dir, device, timeout=args.device_timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            msg = _err_tail(e)
            print(f"  WARNING: sync to '{device.name}' failed, will skip this device: {msg}")
            device_errors[device.name] = msg
    if devices:
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
        local_row["model"] = name
        rows.append(local_row)

        # 2. Export to the Rust runner's JSON model format, if needed either
        #    locally or by any device.
        needs_export = (not args.skip_rust) or any(not d.skip_rust and d.name not in device_errors for d in devices)
        export_path = None
        if needs_export:
            onnx_path = args.work_dir / f"{name}.onnx"
            export_path = args.work_dir / f"{name}.json"
            print(f"Exporting {name} -> {export_path}...")
            export_and_run_model(model, input_dim, onnx_path=onnx_path, export_path=export_path)

        # 3. Local Rust runner, once per feature set.
        if not args.skip_rust:
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

        # 4. Remote devices: PyTorch, then Rust per feature set.
        for device in devices:
            if device.name in device_errors:
                rows.append(
                    {
                        "runner": f"?@{device.name}",
                        "backend": device.name,
                        "runs": 0,
                        "errors": None,
                        "model": name,
                        "error_message": f"device sync failed, skipped: {device_errors[device.name]}",
                    }
                )
                continue

            if not device.skip_python:
                print(f"Running remote PyTorch benchmark: {name} on '{device.name}'...")
                remote_py_row = benchmark_model_remote_python(device, name, args.iterations, seed, args.work_dir, timeout=args.device_timeout)
                remote_py_row["model"] = name
                rows.append(remote_py_row)

            if not device.skip_rust:
                device_features = device.rust_features if device.rust_features is not None else args.rust_features
                for feature in device_features:
                    print(f"Running remote Rust benchmark: {name} [{feature}] on '{device.name}'...")
                    remote_rust_row = run_rust_benchmark_remote(device, name, export_path, args.iterations, feature, args.work_dir, timeout=args.device_timeout)
                    remote_rust_row["model"] = name
                    rows.append(remote_rust_row)

        result = {"model": name, "results": rows}
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
                "devices": [d.name for d in devices],
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved combined results to '{args.results_file}'.")


if __name__ == "__main__":
    main()
