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

Configuration
-------------
By default, every setting is read from a JSON config file next to this
script (benchmark_config.json), so once that file is set up you can just run:

    python3 benchmark.py

with no arguments at all. Example benchmark_config.json:

{
  "iterations": 1000,
  "seed": 42,
  "rust_dir": ".",
  "rust_features": ["default", "simd", "blas"],
  "work_dir": "bench_work",
  "results_file": "combined_benchmark_results.json",
  "skip_rust": false,
  "local_project_dir": null,
  "device_timeout": 1800,
  "devices": [
    {
      "name": "pi4",
      "host": "pi4",
      "dir": "/home/pi/ml-runner",
      "python": "python3",
      "rust_dir": ".",
      "features": ["default", "simd"],
      "sync": true,
      "skip_python": false,
      "skip_rust": false
    }
  ]
}

Any field can be omitted (falls back to the same hardcoded default as
before), and any CLI flag you do pass overrides the corresponding config
value for that run - so quick one-off tweaks don't require editing the file.
`--device` on the CLI replaces the "devices" list from the config entirely
(rather than merging) for ad hoc overrides. Use `--config <path>` to point at
a different config file; if no config file exists at all, the script just
falls back to CLI flags/defaults exactly as before.

How to run :

# Simplest case: everything comes from ./benchmark_config.json
python3 benchmark.py

# Point at a different config file
python3 benchmark.py --config ./configs/pi_only.json
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

# Default config file location: next to this script, so `python3
# benchmark.py` with no arguments works regardless of cwd.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "benchmark_config.json"

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


_DEVICE_CONFIG_REQUIRED_FIELDS = ("name", "host", "dir")
_DEVICE_CONFIG_OPTIONAL_FIELDS = ("python", "rust_dir", "features", "sync", "skip_python", "skip_rust")
_DEVICE_CONFIG_ALL_FIELDS = _DEVICE_CONFIG_REQUIRED_FIELDS + _DEVICE_CONFIG_OPTIONAL_FIELDS


def device_from_config(entry: dict, source: str) -> RemoteDevice:
    """Build a RemoteDevice from one object in a config file's "devices" list.
    Same validation spirit as parse_device_spec: unknown keys, missing
    required values, and wrong types are rejected immediately."""
    if not isinstance(entry, dict):
        raise ValueError(f"each item in \"devices\" in '{source}' must be an object, got {entry!r}")

    unknown = [k for k in entry if k not in _DEVICE_CONFIG_ALL_FIELDS]
    if unknown:
        raise ValueError(
            f"device entry in '{source}' has unrecognized field(s): {', '.join(unknown)}. " f"Recognized fields: {', '.join(_DEVICE_CONFIG_ALL_FIELDS)}"
        )

    missing = [k for k in _DEVICE_CONFIG_REQUIRED_FIELDS if not entry.get(k)]
    if missing:
        raise ValueError(f"device entry in '{source}' is missing required field(s): {', '.join(missing)}: {entry}")

    name = entry["name"]
    features = entry.get("features")
    if features is not None and (not isinstance(features, list) or not all(isinstance(x, str) for x in features)):
        raise ValueError(f"device '{name}' in '{source}': \"features\" must be a list of strings, got {features!r}")

    for bool_key in ("sync", "skip_python", "skip_rust"):
        if bool_key in entry and not isinstance(entry[bool_key], bool):
            raise ValueError(f"device '{name}' in '{source}': \"{bool_key}\" must be true/false, got {entry[bool_key]!r}")

    return RemoteDevice(
        name=name,
        host=entry["host"],
        remote_dir=str(entry["dir"]).rstrip("/"),
        python_bin=entry.get("python", "python3"),
        rust_subdir=entry.get("rust_dir", "."),
        rust_features=list(features) if features else None,
        sync=entry.get("sync", True),
        skip_python=entry.get("skip_python", False),
        skip_rust=entry.get("skip_rust", False),
    )


_CONFIG_ALL_FIELDS = (
    "iterations",
    "seed",
    "rust_dir",
    "rust_features",
    "work_dir",
    "results_file",
    "skip_rust",
    "local_project_dir",
    "device_timeout",
    "devices",
)


def load_config_file(path: Path) -> dict:
    """Load and lightly validate a benchmark_config.json. Raises ValueError
    with a clear message on any problem (missing file, bad JSON, unrecognized
    top-level key) rather than letting a typo pass silently."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"config file '{path}' does not exist") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"failed to parse config file '{path}': {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"config file '{path}' must contain a JSON object at the top level")

    unknown = [k for k in data if k not in _CONFIG_ALL_FIELDS]
    if unknown:
        raise ValueError(
            f"config file '{path}' has unrecognized top-level field(s): {', '.join(unknown)}. " f"Recognized fields: {', '.join(_CONFIG_ALL_FIELDS)}"
        )

    devices_raw = data.get("devices")
    if devices_raw is not None:
        if not isinstance(devices_raw, list):
            raise ValueError(f"config file '{path}': \"devices\" must be a list")
        # Validate eagerly (rather than lazily at use-time) so a bad device
        # entry is reported before any benchmarking has started.
        for entry in devices_raw:
            device_from_config(entry, source=str(path))

    return data


def load_config(path: str) -> tuple[dict, list[RemoteDevice], Path | None]:
    """Merge CLI args (highest priority when explicitly given) with the
    config file (fallback), and hardcoded defaults (final fallback).
    Returns (resolved_settings, devices, config_path_used)."""
    config: dict = load_config_file(path)
    devices = [device_from_config(entry, source=str(path)) for entry in config.get("devices", [])]

    # post processing
    resolved = {
        "iterations": config.get("iterations", 1000),
        "seed": config.get("seed") if config.get("seed") is not None else int(time.time()),
        "rust_dir": Path(config.get("rust_dir", ".")),
        "rust_features": config.get("rust_features", RUST_FEATURES),
        "work_dir": Path(config.get("work_dir", "bench_work")),
        "results_file": Path(config.get("results_file", "combined_benchmark_results.json")),
        "skip_rust": config.get("skip_rust", False),
        "local_project_dir": (Path(config["local_project_dir"]) if config.get("local_project_dir") else None),
        "device_timeout": config.get("device_timeout", 1800),
    }

    return resolved, devices


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


def preflight_check_device(device: RemoteDevice, timeout: float | None = None) -> str | None:
    """Run cheap remote checks before committing to a full rsync + benchmark
    run: is the host reachable over ssh, and does it have what this device's
    config says it will need (cargo for the Rust side, the configured Python
    binary plus a `torch` install for the Python side)?

    Returns an error message describing the first thing that's missing, or
    None if the device looks ready. This exists so a typo'd python binary or
    a Pi that never got `cargo`/`torch` installed fails in a few seconds,
    instead of after several minutes of rsync-ing and building."""
    checks = []
    if not device.skip_rust:
        checks.append("command -v cargo >/dev/null 2>&1 || echo __MISSING__:cargo")
    if not device.skip_python:
        py = shlex.quote(device.python_bin)
        checks.append(f"command -v {py} >/dev/null 2>&1 || echo __MISSING__:'{device.python_bin}' (python interpreter)")
        checks.append(f"{py} -c 'import torch' >/dev/null 2>&1 || echo __MISSING__:'torch' (python module)")
    if not checks:
        return None

    remote_script = " ; ".join(checks)
    try:
        result = subprocess.run(["ssh", *SSH_OPTS, device.host, remote_script], capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"could not reach device for preflight check: {_err_tail(e)}"

    missing = [line[len("__MISSING__:") :] for line in result.stdout.splitlines() if line.startswith("__MISSING__:")]
    if missing:
        return f"missing on device: {', '.join(missing)}"
    if result.returncode != 0:
        # ssh (or the shell it ran) failed outright - auth, unknown host, etc.
        return f"ssh preflight check failed (exit {result.returncode}): {result.stderr.strip()[-300:]}"
    return None


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
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        metavar="PATH",
        help=f"Path to a JSON config file with all settings (default: {DEFAULT_CONFIG_PATH.name} next to this script, if present)",
    )
    args = parser.parse_args()

    settings, devices = load_config(args.config)
    print(settings)

    torch.set_grad_enabled(False)
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])

    settings["work_dir"].mkdir(parents=True, exist_ok=True)

    print(f"Config: {args.config}")
    print(f"Iterations per run: {settings['iterations']} | seed: {settings['seed']}")
    print(f"PyTorch backend: {backend_name()}")
    if not settings["skip_rust"]:
        print(f"Rust runner: {settings['rust_dir']} | features: {settings['rust_features']}")
    if devices:
        print(f"Devices: {[d.name for d in devices]}")
    print()

    # Preflight + sync each device once up front (not once per model) so we
    # don't rsync the whole project twice for two models, and so a device
    # that's missing cargo/python/torch fails in seconds rather than after a
    # full build. A device that fails either step is marked broken and every
    # row for it becomes an error row, rather than aborting the whole run.
    local_project_dir = settings["local_project_dir"] or Path(__file__).resolve().parent
    device_errors: dict[str, str] = {}
    for device in devices:
        print(f"Checking device '{device.name}' ({device.host})...")
        preflight_error = preflight_check_device(device, timeout=settings["device_timeout"])
        if preflight_error:
            print(f"  WARNING: preflight check for '{device.name}' failed, will skip this device: {preflight_error}")
            device_errors[device.name] = preflight_error
            continue

        if not device.sync:
            continue
        print(f"Syncing project to device '{device.name}' ({device.host}:{device.remote_dir})...")
        try:
            sync_project_to_remote(local_project_dir, device, timeout=settings["device_timeout"])
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
        seed = settings["seed"] + idx
        rows = []

        # 1. Local PyTorch run.
        print(f"Running local PyTorch benchmark: {name}...")
        local_row = benchmark_model_local(model, input_dim=input_dim, iterations=settings["iterations"], seed=seed, name=name)
        local_row["model"] = name
        rows.append(local_row)

        # 2. Export to the Rust runner's JSON model format, if needed either
        #    locally or by any device.
        needs_export = (not settings["skip_rust"]) or any(not d.skip_rust and d.name not in device_errors for d in devices)
        export_path = None
        if needs_export:
            onnx_path = settings["work_dir"] / f"{name}.onnx"
            export_path = settings["work_dir"] / f"{name}.json"
            print(f"Exporting {name} -> {export_path}...")
            export_and_run_model(model, input_dim, onnx_path=onnx_path, export_path=export_path)

        # 3. Local Rust runner, once per feature set.
        if not settings["skip_rust"]:
            for feature in settings["rust_features"]:
                report_path = settings["work_dir"] / f"{name}_{feature}.report.json"
                print(f"Running Rust benchmark: {name} [{feature}]...")
                rust_row = run_rust_benchmark(
                    rust_dir=settings["rust_dir"],
                    export_path=export_path,
                    iterations=settings["iterations"],
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
                        "error_message": f"device unavailable, skipped: {device_errors[device.name]}",
                    }
                )
                continue

            if not device.skip_python:
                print(f"Running remote PyTorch benchmark: {name} on '{device.name}'...")
                remote_py_row = benchmark_model_remote_python(
                    device, name, settings["iterations"], seed, settings["work_dir"], timeout=settings["device_timeout"]
                )
                remote_py_row["model"] = name
                rows.append(remote_py_row)

            if not device.skip_rust:
                device_features = device.rust_features if device.rust_features is not None else settings["rust_features"]
                for feature in device_features:
                    print(f"Running remote Rust benchmark: {name} [{feature}] on '{device.name}'...")
                    remote_rust_row = run_rust_benchmark_remote(
                        device, name, export_path, settings["iterations"], feature, settings["work_dir"], timeout=settings["device_timeout"]
                    )
                    remote_rust_row["model"] = name
                    rows.append(remote_rust_row)

        result = {"model": name, "results": rows}
        output_to_print.append(print_results(result))
        results.append(result)

    for output in output_to_print:
        print(output)

    with open(settings["results_file"], "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "iterations": settings["iterations"],
                "seed": settings["seed"],
                "devices": [d.name for d in devices],
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved combined results to '{settings['results_file']}'.")


if __name__ == "__main__":
    main()
