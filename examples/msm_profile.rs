//! Per-stage timing of the GPU MSM.  LOG=<log2 n> C=<window bits>
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::{PrimeField, UniformRand};
use zk_gpu::{gpu::Gpu, host, msm::Msm};

fn main() {
    let gpu = Gpu::new();
    let log_n: u32 = std::env::var("LOG").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let n = 1usize << log_n;
    let c: u32 = std::env::var("C").ok().and_then(|v| v.parse().ok()).unwrap_or(Msm::default_window(n));
    let mut rng = ark_std::test_rng();
    let (mut p, g) = (G1Projective::rand(&mut rng), G1Projective::rand(&mut rng));
    let proj: Vec<G1Projective> = (0..n)
        .map(|_| {
            p += g;
            p
        })
        .collect();
    let bases: Vec<G1Affine> = G1Projective::normalize_batch(&proj);
    let plain: Vec<u32> = bases
        .iter()
        .flat_map(|b| [host::words32(&b.x.into_bigint().0), host::words32(&b.y.into_bigint().0)].concat())
        .collect();
    let sc: Vec<u32> = (0..n).flat_map(|_| host::words32(&Fr::rand(&mut rng).into_bigint().0)).collect();
    let (base_buf, scalar_buf) = (Msm::upload_bases(&gpu, &plain), gpu.buffer(&sc));
    let items: u32 = std::env::var("ITEM").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let msm = Msm::with_item_points(&gpu, n, c, items);
    msm.run(&gpu, &base_buf, &scalar_buf);
    gpu.wait();
    println!("n = 2^{log_n}, c = {c}, item = {items}");
    let mut total = 0.0;
    for (name, ms) in msm.profile(&gpu, &base_buf, &scalar_buf) {
        println!("  {name:16} {ms:9.2} ms");
        total += ms;
    }
    println!("  {:16} {total:9.2} ms", "total");
}
