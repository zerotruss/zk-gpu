//! Raw modular-multiplication throughput: GPU kernel vs multi-threaded arkworks.
//! Also checks the GPU result against arkworks.

use ark_bn254::Fq;
use ark_ff::{BigInteger, Field, PrimeField, UniformRand};
use rayon::prelude::*;
use std::time::Instant;
use zk_gpu::{field, gpu::Gpu};

/// `ways` independent multiplication chains per thread (instruction-level parallelism).
fn kernel(wg: u32, ways: u32) -> String {
    if ways == 1 {
        return format!(
            "
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>; // [n, iters]
@compute @workgroup_size({wg})
fn bench(@builtin(global_invocation_id) id: vec3<u32>) {{
  let i = id.x;
  if (i >= params[0]) {{ return; }}
  var a: Fq;
  for (var j = 0u; j < 20u; j++) {{ a.l[j] = data[i * 20u + j]; }}
  var x = a;
  for (var k = 0u; k < params[1]; k++) {{ x = fq_mul(x, a); }}
  for (var j = 0u; j < 20u; j++) {{ data[i * 20u + j] = x.l[j]; }}
}}"
        );
    }
    // two chains: element i and element i + n/2 per thread (n/2 threads)
    format!(
        "
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>; // [n, iters]
@compute @workgroup_size({wg})
fn bench(@builtin(global_invocation_id) id: vec3<u32>) {{
  let i = id.x;
  let h = params[0] / 2u;
  if (i >= h) {{ return; }}
  var a: Fq; var b: Fq;
  for (var j = 0u; j < 20u; j++) {{ a.l[j] = data[i * 20u + j]; b.l[j] = data[(i + h) * 20u + j]; }}
  var x = a; var y = b;
  for (var k = 0u; k < params[1]; k++) {{ x = fq_mul(x, a); y = fq_mul(y, b); }}
  for (var j = 0u; j < 20u; j++) {{ data[i * 20u + j] = x.l[j]; data[(i + h) * 20u + j] = y.l[j]; }}
}}"
    )
}

fn to_words(x: Fq) -> [u64; 4] {
    let b = x.into_bigint().to_bytes_le();
    std::array::from_fn(|i| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()))
}

fn main() {
    let (n, iters) = (1usize << 18, 256u32);
    let mut rng = ark_std::test_rng();
    let inputs: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();

    let gpu = Gpu::new();
    println!("GPU: {} ({:?})", gpu.adapter.name, gpu.adapter.backend);
    let env = |k: &str, d: u32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (wg, ways) = (env("WG", 64), env("WAYS", 1));
    println!("workgroup size {wg}, {ways} chain(s) per thread");
    let shader = format!("{}\n{}", field::wgsl("fq", field::FQ), kernel(wg, ways));
    let t = Instant::now();
    let pipeline = gpu.pipeline(&shader, "bench");
    println!("shader compile: {:?}", t.elapsed());

    let flat: Vec<u32> = inputs.iter().flat_map(|x| field::to_limbs(to_words(*x))).collect();
    let data = gpu.buffer(&flat);
    let params = gpu.buffer(&[n as u32, iters]);
    let groups = ((n as u32) / ways).div_ceil(wg);

    gpu.run(&pipeline, &[&data, &params], groups); // warm-up (also the run we verify)
    let out = gpu.read(&data, n * 20);

    // Verify: each step computes x * a * R^-1 with R = 2^260.
    let r_inv = Fq::from(2u64).pow([260u64]).inverse().unwrap();
    for i in (0..n).step_by(n / 16) {
        let a = inputs[i];
        let mut x = a;
        for _ in 0..iters {
            x = x * a * r_inv;
        }
        let got = field::from_limbs(&out[i * 20..i * 20 + 20]);
        assert_eq!(got, to_words(x), "mismatch at element {i}");
    }
    println!("GPU result matches arkworks ({} samples)", 16);

    // GPU timing (data keeps evolving; values stay valid field elements).
    let runs = 5;
    let t = Instant::now();
    for _ in 0..runs {
        gpu.run(&pipeline, &[&data, &params], groups);
    }
    let gpu_s = t.elapsed().as_secs_f64() / runs as f64;
    let muls = (n as u64 * iters as u64) as f64;
    println!("GPU: {:.1} ms  -> {:.0} M mulmod/s", gpu_s * 1e3, muls / gpu_s / 1e6);

    // CPU: same chain, all cores.
    let t = Instant::now();
    let sum: Fq = inputs
        .par_iter()
        .map(|&a| {
            let mut x = a;
            for _ in 0..iters {
                x *= a;
            }
            x
        })
        .sum();
    let cpu_s = t.elapsed().as_secs_f64();
    std::hint::black_box(sum);
    println!(
        "CPU ({} threads): {:.1} ms -> {:.0} M mulmod/s",
        rayon::current_num_threads(),
        cpu_s * 1e3,
        muls / cpu_s / 1e6
    );
    println!("GPU/CPU throughput ratio: {:.2}x", cpu_s / gpu_s);
}
