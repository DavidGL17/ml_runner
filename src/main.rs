mod activation;
mod conv;
mod dense;
mod flatten;
mod layers;
mod model;
mod rnn;
mod tensor;

use crate::model::Model;
use crate::tensor::{Tensor, TensorShape};
use std::env;
use std::fs;
use std::time::{Duration, Instant};

/// Minimal xorshift64* PRNG so we don't need to pull in the `rand` crate
/// just to generate benchmark input tensors.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // xorshift has a fixed point at 0, so make sure we never start there.
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Pseudo-random f32 in [-1.0, 1.0), a reasonable generic range for
    /// model inputs.
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits of entropy
        let unit = bits as f32 / (1u32 << 24) as f32; // [0, 1)
        unit * 2.0 - 1.0
    }
}

fn random_tensor(shape: &TensorShape, rng: &mut Rng) -> Tensor {
    let n = shape.total_size();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    Tensor::new(data, shape.clone())
}

/// Summary statistics for a batch of timings, all in nanoseconds.
struct Stats {
    runs: usize,
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
    std_dev: f64,
}

fn compute_stats(mut samples_ns: Vec<f64>) -> Stats {
    samples_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let runs = samples_ns.len();
    let min = samples_ns[0];
    let max = samples_ns[runs - 1];
    let mean = samples_ns.iter().sum::<f64>() / runs as f64;
    let median = if runs.is_multiple_of(2) {
        (samples_ns[runs / 2 - 1] + samples_ns[runs / 2]) / 2.0
    } else {
        samples_ns[runs / 2]
    };
    let variance = samples_ns.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / runs as f64;
    let std_dev = variance.sqrt();

    Stats {
        runs,
        min,
        max,
        mean,
        median,
        std_dev,
    }
}

fn fmt_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.3} ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.3} µs", ns / 1_000.0)
    } else {
        format!("{:.1} ns", ns)
    }
}

fn backend_name() -> &'static str {
    if cfg!(feature = "simd") {
        "simd (wide)"
    } else if cfg!(feature = "blas") {
        "ndarray (BLAS)"
    } else {
        "ndarray (matrixmultiply)"
    }
}

fn main() {
    // Usage: cargo run -- [path/to/export.json] [iterations]
    //   path/to/export.json  Model file to benchmark. Defaults to "export.json".
    //   iterations           Number of timed forward passes. Defaults to 1000.
    let mut args = env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "export.json".to_string());
    let iterations: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1000);

    // Discard a handful of warmup runs (cache warm-up, allocator settling,
    // etc.) before recording timings, but never so many that nothing is left.
    let warmup = std::cmp::min(10, iterations.saturating_sub(1));

    let model_json = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(e) => {
            eprintln!("Error reading model file '{}': {}", path, e);
            return;
        }
    };

    let model = match Model::from_json(&model_json) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing model JSON: {}", e);
            return;
        }
    };

    if let Err(e) = model.validate_shapes() {
        eprintln!("Error when validating model shapes:\n{}", e);
        return;
    }

    println!("Model loaded successfully from '{}'.", path);
    println!(
        "Running {} iterations ({} warmup, discarded) on backend: {}\n",
        iterations,
        warmup,
        backend_name()
    );

    let mut rng = Rng::new(0x5EED_C0FF_EE15_D00D);
    let mut samples_ns: Vec<f64> = Vec::with_capacity(iterations.saturating_sub(warmup));
    let mut errors = 0usize;

    for i in 0..iterations {
        let input = random_tensor(&model.input_shape, &mut rng);

        let start = Instant::now();
        let result = model.forward(&input);
        let elapsed: Duration = start.elapsed();

        match result {
            Ok(_) => {
                if i >= warmup {
                    samples_ns.push(elapsed.as_nanos() as f64);
                }
            }
            Err(e) => {
                errors += 1;
                eprintln!("Error during forward pass (iteration {}): {}", i, e);
            }
        }
    }

    if samples_ns.is_empty() {
        eprintln!(
            "No successful forward passes to report on (errors: {}).",
            errors
        );
        return;
    }

    let stats = compute_stats(samples_ns);

    // ---- Markdown report ----
    println!("## Benchmark Report");
    println!();
    println!("- **Model:** `{}`", path);
    println!("- **Backend:** {}", backend_name());
    println!("- **Input shape:** {:?}", model.input_shape);
    println!("- **Output shape:** {:?}", model.output_shape);
    println!(
        "- **Successful runs:** {} (warmup discarded: {}, errors: {})",
        stats.runs, warmup, errors
    );
    println!();
    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| Min    | {} |", fmt_ns(stats.min));
    println!("| Max    | {} |", fmt_ns(stats.max));
    println!("| Mean   | {} |", fmt_ns(stats.mean));
    println!("| Median | {} |", fmt_ns(stats.median));
    println!("| StdDev | {} |", fmt_ns(stats.std_dev));
}