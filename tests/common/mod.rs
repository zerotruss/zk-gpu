//! Helpers shared by the native test files: conversions between arkworks types and the
//! GPU's plain `u32`-word format, and one-call wrappers around the GPU MSM and NTT.
#![allow(dead_code)]
use ark_bn254::{Fq, Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::{BigInt, PrimeField};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use zk_gpu::{
    gpu::Gpu,
    host,
    msm::{xyzz_to_affine, Msm},
    ntt::Ntt,
};

/// A reproducible random generator: a failing seed can be re-run.
pub fn rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// `ZK_GPU_QUICK=1` trims the largest sizes and iteration counts (see `check.sh quick`).
pub fn quick() -> bool {
    std::env::var("ZK_GPU_QUICK").is_ok_and(|v| v == "1")
}

pub fn fr_words(v: &[Fr]) -> Vec<u32> {
    v.iter().flat_map(|x| host::words32(&x.into_bigint().0)).collect()
}

pub fn fr_from_words(w: &[u32]) -> Vec<Fr> {
    w.chunks(8).map(|c| Fr::from_bigint(BigInt::new(host::from_words32(c))).expect("value >= r")).collect()
}

pub fn point_words(p: &G1Affine) -> Vec<u32> {
    let (x, y) = p.xy().map(|(x, y)| (x.into_bigint().0, y.into_bigint().0)).unwrap_or(([0; 4], [0; 4]));
    [host::words32(&x), host::words32(&y)].concat()
}

pub fn affine_from_coords(coords: Option<([u64; 4], [u64; 4])>) -> G1Affine {
    match coords {
        None => G1Affine::zero(),
        Some((x, y)) => {
            let fq = |v| Fq::from_bigint(BigInt::new(v)).expect("coordinate >= p");
            G1Affine::new(fq(x), fq(y))
        }
    }
}

/// `plan = None` uses the library defaults for window and work-item size,
/// `Some((window_bits, item_points))` overrides them.
pub fn gpu_msm(gpu: &Gpu, bases: &[G1Affine], scalars: &[Fr], plan: Option<(u32, u32)>) -> G1Affine {
    let plain: Vec<u32> = bases.iter().flat_map(point_words).collect();
    let base_buf = Msm::upload_bases(gpu, &plain);
    let scalar_buf = gpu.buffer(&fr_words(scalars));
    let msm = match plan {
        Some((c, item_points)) => Msm::with_item_points(gpu, bases.len(), c, item_points),
        None => Msm::new(gpu, bases.len(), Msm::default_window(bases.len())),
    };
    let result = msm.run(gpu, &base_buf, &scalar_buf);
    affine_from_coords(xyzz_to_affine(&gpu.read(result, 32)))
}

pub fn gpu_ntt_forward(gpu: &Gpu, ntt: &Ntt, input: &[Fr]) -> Vec<Fr> {
    let buffer = gpu.buffer(&fr_words(input));
    ntt.forward(gpu, &buffer);
    fr_from_words(&gpu.read(&buffer, 8 * input.len()))
}

pub fn gpu_ntt_inverse(gpu: &Gpu, ntt: &Ntt, input: &[Fr]) -> Vec<Fr> {
    let buffer = gpu.buffer(&fr_words(input));
    ntt.inverse(gpu, &buffer);
    fr_from_words(&gpu.read(&buffer, 8 * input.len()))
}
