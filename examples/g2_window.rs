//! Times the G2 MSM for several window sizes (to tune the default). Bases are multiples of the
//! generator: the running time does not depend on which points they are.
//!   cargo run --release --example g2_window -- 17
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{PrimeField, UniformRand};
use std::time::Instant;
use zk_gpu::{gpu::Gpu, group::Group, host, msm::Msm};

fn main() {
    let log_n: u32 = std::env::args().nth(1).map_or(17, |a| a.parse().unwrap());
    let group = if std::env::var("G1").is_ok() { Group::G1 } else { Group::G2 };
    let n = 1usize << log_n;
    let mut rng = ark_std::test_rng();
    let words: Vec<u32> = if group == Group::G1 {
        let g = G1Projective::from(G1Affine::generator());
        let (mut acc, mut points) = (g, Vec::with_capacity(n));
        for _ in 0..n {
            points.push(acc);
            acc += g;
        }
        G1Projective::normalize_batch(&points)
            .iter()
            .flat_map(|p| {
                let (x, y) = p.xy().unwrap();
                [x, y].into_iter().flat_map(|c| host::words32(&c.into_bigint().0))
            })
            .collect()
    } else {
        let g = G2Projective::from(G2Affine::generator());
        let (mut acc, mut points) = (g, Vec::with_capacity(n));
        for _ in 0..n {
            points.push(acc);
            acc += g;
        }
        G2Projective::normalize_batch(&points)
            .iter()
            .flat_map(|p| {
                let (x, y) = p.xy().unwrap();
                [x.c0, x.c1, y.c0, y.c1].into_iter().flat_map(|c| host::words32(&c.into_bigint().0))
            })
            .collect()
    };
    let scalars: Vec<u32> = (0..n).flat_map(|_| host::words32(&Fr::rand(&mut rng).into_bigint().0)).collect();

    let gpu = Gpu::new();
    let base_buf = Msm::upload_bases(&gpu, &words);
    let scalar_buf = gpu.buffer(&scalars);
    println!("{group:?} MSM, n = 2^{log_n}  ({})", gpu.adapter.name);
    for c in 10..=16u32 {
        let msm = Msm::for_group(&gpu, group, n, c, None);
        let mut best = f64::MAX;
        for run in 0..4 {
            let t = Instant::now();
            let out = msm.run(&gpu, &base_buf, &scalar_buf);
            gpu.read(out, group.xyzz_words());
            if run > 0 {
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        println!("  c = {c:2}: {best:8.1} ms{}", if c == Msm::default_window(n) { "   <- default" } else { "" });
        if std::env::var("STAGES").is_ok() {
            let stages = msm.profile(&gpu, &base_buf, &scalar_buf);
            println!(
                "          {}",
                stages.iter().map(|(name, ms)| format!("{name} {ms:.0}")).collect::<Vec<_>>().join(", ")
            );
        }
    }
}
