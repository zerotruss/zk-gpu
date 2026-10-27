//! Benchmark runner: GPU vs multi-threaded arkworks for the NTT and MSM, written to JSON so that
//! runs can be compared over time with `examples/compare.rs`.
//!
//!   cargo run --release --example bench -- [--ntt 14,16,18] [--msm 14,16] [--runs 5]
//!                                          [--no-stages] [--warmup-ms 300] [--out bench/results/latest.json]
//!
//! Every size is first checked against arkworks, so a wrong result can never be recorded as a time.
//! The CPU is timed in one block, then the GPU is warmed up and timed in a second block; the fastest run of
//! each is kept, because interference only ever adds time. G1 only (the G2 MSM has no entry here).
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use serde_json::{json, Value};
use std::{path::Path, time::Instant};
use zk_gpu::{
    gpu::Gpu,
    host,
    msm::{xyzz_to_affine, Msm},
    ntt::Ntt,
};

struct Args {
    ntt: Vec<u32>,
    msm: Vec<u32>,
    runs: usize,
    stages: bool,
    warmup_ms: u64,
    out: String,
}

fn parse_args() -> Args {
    let mut args = Args {
        ntt: vec![14, 16, 18, 20, 22],
        msm: vec![14, 16, 18, 20],
        runs: 5,
        stages: true,
        warmup_ms: 300,
        out: "bench/results/latest.json".into(),
    };
    let list = |v: &str| -> Vec<u32> {
        v.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().expect("sizes are log2 integers")).collect()
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--ntt" => args.ntt = list(&value()),
            "--msm" => args.msm = list(&value()),
            "--runs" => args.runs = value().parse().expect("--runs needs a number"),
            "--warmup-ms" => args.warmup_ms = value().parse().expect("--warmup-ms needs a number"),
            "--out" => args.out = value(),
            "--no-stages" => args.stages = false,
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(args.runs >= 1, "--runs must be at least 1");
    args
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// median / min / max of a set of timings, in milliseconds.
fn stats(samples: &[f64]) -> Value {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if s.len() % 2 == 1 { s[s.len() / 2] } else { (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0 };
    json!({ "median": median, "min": s[0], "max": s[s.len() - 1] })
}

/// Times `runs` calls of `f`, in milliseconds.
fn time_runs(runs: usize, mut f: impl FnMut()) -> Vec<f64> {
    (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            ms(t)
        })
        .collect()
}

/// Keeps the GPU busy for `warmup_ms`. An idle GPU runs at a reduced clock and needs tens to
/// hundreds of milliseconds of sustained work to speed up: the first runs after a pause measured
/// exactly 2x slower on an M1. Timing them would measure the power state, not the kernels.
fn warm_up(warmup_ms: u64, mut f: impl FnMut()) {
    let start = Instant::now();
    let mut runs = 0;
    while runs < 2 || start.elapsed().as_millis() < warmup_ms as u128 {
        f();
        runs += 1;
    }
}

fn median_of(v: &Value) -> f64 {
    v["median"].as_f64().unwrap()
}

fn min_of(v: &Value) -> f64 {
    v["min"].as_f64().unwrap()
}

/// FNV-1a over every source and shader file: identifies exactly which code produced a result
/// (there is no git history to ask).
fn source_hash() -> String {
    fn walk(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, files);
            } else if matches!(path.extension().and_then(|e| e.to_str()), Some("rs" | "wgsl")) {
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
    files.sort();
    let mut hash = 0xcbf29ce484222325u64;
    for byte in files.iter().flat_map(|f| std::fs::read(f).unwrap()) {
        hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// "ac", "battery" or "unknown". Recorded because laptops change clocks with the power source.
/// (On the M1 used here, AC and battery runs gave the same timings; other machines may differ.)
fn power_source() -> &'static str {
    let output = std::process::Command::new("pmset").args(["-g", "batt"]).output();
    match output.map(|o| String::from_utf8_lossy(&o.stdout).to_string()) {
        Ok(text) if text.contains("AC Power") => "ac",
        Ok(text) if text.contains("Battery Power") => "battery",
        _ => "unknown",
    }
}

fn slug(text: &str) -> String {
    let s: String = text.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-")
}

fn bench_ntt(gpu: &Gpu, log_n: u32, runs: usize, warmup_ms: u64) -> Value {
    let n = 1usize << log_n;
    let mut rng = ark_std::test_rng();
    let input: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
    let ntt = Ntt::new(gpu, log_n);
    let words: Vec<u32> = input.iter().flat_map(|x| host::words32(&x.into_bigint().0)).collect();
    let buffer = gpu.buffer(&words);

    // Correctness gate (also the warm-up for both sides).
    ntt.forward(gpu, &buffer);
    let expected: Vec<u32> = domain.fft(&input).iter().flat_map(|x| host::words32(&x.into_bigint().0)).collect();
    assert!(
        gpu.read(&buffer, 8 * n) == expected,
        "NTT 2^{log_n}: GPU result differs from arkworks, refusing to benchmark"
    );

    let mut scratch = input.clone();
    let cpu = time_runs(runs, || {
        scratch.copy_from_slice(&input);
        domain.fft_in_place(&mut scratch);
    });
    let mut gpu_run = || {
        ntt.forward(gpu, &buffer);
        gpu.wait();
    };
    warm_up(warmup_ms, &mut gpu_run);
    let gpu_ms = time_runs(runs, &mut gpu_run);
    let (cpu, gpu_ms) = (stats(&cpu), stats(&gpu_ms));
    println!(
        "ntt 2^{log_n:<2} cpu {:>9.2} ms   gpu {:>9.2} ms   {:>5.2}x",
        median_of(&cpu),
        median_of(&gpu_ms),
        median_of(&cpu) / median_of(&gpu_ms)
    );
    json!({
        "op": "ntt", "log_n": log_n, "cpu_ms": cpu, "gpu_ms": gpu_ms,
        "speedup": median_of(&cpu) / median_of(&gpu_ms), "speedup_best": min_of(&cpu) / min_of(&gpu_ms),
    })
}

fn bench_msm(gpu: &Gpu, log_n: u32, runs: usize, stages: bool, warmup_ms: u64) -> Value {
    let n = 1usize << log_n;
    let c = Msm::default_window(n);
    let mut rng = ark_std::test_rng();
    // Distinct bases by chaining additions: much cheaper to build than n random points.
    let (mut p, g) = (G1Projective::rand(&mut rng), G1Projective::rand(&mut rng));
    let proj: Vec<G1Projective> = (0..n)
        .map(|_| {
            p += g;
            p
        })
        .collect();
    let bases: Vec<G1Affine> = G1Projective::normalize_batch(&proj);
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    let plain: Vec<u32> = bases
        .iter()
        .flat_map(|b| [host::words32(&b.x.into_bigint().0), host::words32(&b.y.into_bigint().0)].concat())
        .collect();
    let scalar_words: Vec<u32> = scalars.iter().flat_map(|s| host::words32(&s.into_bigint().0)).collect();
    let base_buf = Msm::upload_bases(gpu, &plain);
    let scalar_buf = gpu.buffer(&scalar_words);
    let msm = Msm::new(gpu, n, c);

    // Correctness gate (also the warm-up).
    let expected = G1Projective::msm(&bases, &scalars).unwrap().into_affine();
    let got = xyzz_to_affine(&gpu.read(msm.run(gpu, &base_buf, &scalar_buf), 32)).expect("MSM returned infinity");
    assert!(
        got == (expected.x.into_bigint().0, expected.y.into_bigint().0),
        "MSM 2^{log_n}: GPU result differs from arkworks, refusing to benchmark"
    );

    let cpu = time_runs(runs, || {
        let _ = std::hint::black_box(G1Projective::msm(&bases, &scalars).unwrap().into_affine());
    });
    let mut kernel_run = || {
        msm.run(gpu, &base_buf, &scalar_buf);
        gpu.wait();
    };
    warm_up(warmup_ms, &mut kernel_run);
    let kernel = time_runs(runs, &mut kernel_run);
    // End to end: upload the scalars, compute, read the result back.
    let e2e = time_runs(runs, || {
        let fresh = gpu.buffer(&scalar_words);
        gpu.read(msm.run(gpu, &base_buf, &fresh), 32);
    });
    let mut stage_samples: Vec<(&'static str, Vec<f64>)> = Vec::new();
    if stages {
        for _ in 0..runs {
            for (i, (name, time)) in msm.profile(gpu, &base_buf, &scalar_buf).into_iter().enumerate() {
                if stage_samples.len() <= i {
                    stage_samples.push((name, Vec::new()));
                }
                stage_samples[i].1.push(time);
            }
        }
    }
    let (cpu, kernel, e2e) = (stats(&cpu), stats(&kernel), stats(&e2e));
    println!(
        "msm 2^{log_n:<2} cpu {:>9.2} ms   gpu {:>9.2} ms (kernel) {:>9.2} ms (end to end)   {:>5.2}x",
        median_of(&cpu),
        median_of(&kernel),
        median_of(&e2e),
        median_of(&cpu) / median_of(&e2e)
    );
    let stage_json: serde_json::Map<String, Value> =
        stage_samples.iter().map(|(name, s)| (name.to_string(), stats(s))).collect();
    json!({
        "op": "msm", "log_n": log_n, "window_bits": c,
        "cpu_ms": cpu, "gpu_ms": kernel, "gpu_e2e_ms": e2e,
        "speedup": median_of(&cpu) / median_of(&e2e), "speedup_best": min_of(&cpu) / min_of(&e2e),
        "stages_ms": stage_json,
    })
}

fn main() {
    let args = parse_args();
    let gpu = Gpu::new();
    let device = format!("{} ({:?})", gpu.adapter.name, gpu.adapter.backend);
    println!(
        "device: {device}   cpu threads: {}   runs per size: {}   power: {}",
        rayon::current_num_threads(),
        args.runs,
        power_source()
    );

    let mut results = Vec::new();
    for &log_n in &args.ntt {
        results.push(bench_ntt(&gpu, log_n, args.runs, args.warmup_ms));
    }
    for &log_n in &args.msm {
        results.push(bench_msm(&gpu, log_n, args.runs, args.stages, args.warmup_ms));
    }

    let report = json!({
        "meta": {
            "unix_time": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            "device": device,
            "device_slug": slug(&device),
            "cpu_threads": rayon::current_num_threads(),
            "power": power_source(),
            "os": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            "source_hash": source_hash(),
            "runs": args.runs,
            "warmup_ms": args.warmup_ms,
        },
        "results": results,
    });
    if let Some(dir) = Path::new(&args.out).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    println!("wrote {}", args.out);
}
