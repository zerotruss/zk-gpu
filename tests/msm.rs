#![cfg(not(target_arch = "wasm32"))]
mod common;
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{UniformRand, Zero};
use common::gpu_msm;
use zk_gpu::gpu::Gpu;

/// Checks the default work-item size and a tiny one (many multi-item buckets).
fn check(gpu: &Gpu, bases: &[G1Affine], scalars: &[Fr], c: u32, what: &str) {
    let expected = G1Projective::msm(bases, scalars).unwrap().into_affine();
    for item_points in [1024, 7] {
        let got = gpu_msm(gpu, bases, scalars, Some((c, item_points)));
        assert_eq!(got, expected, "{what} (n = {}, c = {c}, item_points = {item_points})", bases.len());
    }
}

#[test]
fn random_inputs() {
    let gpu = Gpu::new();
    let mut rng = ark_std::test_rng();
    for (n, c) in [(1000usize, 9u32), (4096, 10), (20_000, 12), (100_000, 14)] {
        let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        check(&gpu, &bases, &scalars, c, "random");
    }
}

#[test]
fn edge_cases() {
    let gpu = Gpu::new();
    let mut rng = ark_std::test_rng();
    let n = 3000;
    let c = 10;
    let p = G1Affine::rand(&mut rng);
    let random: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
    let rand_s: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    check(&gpu, &random, &vec![Fr::zero(); n], c, "all-zero scalars");
    check(&gpu, &random, &vec![Fr::from(1u64); n], c, "all-one scalars (one huge bucket)");
    check(&gpu, &random, &vec![-Fr::from(1u64); n], c, "scalars = r - 1");
    check(&gpu, &vec![p; n], &rand_s, c, "same base everywhere (forces doublings)");
    check(&gpu, &vec![p; n], &vec![Fr::from(5u64); n], c, "same base, same scalar");
    let small: Vec<Fr> = (0..n).map(|i| Fr::from((i % 7) as u64)).collect();
    check(&gpu, &random, &small, c, "tiny scalars");
    let mut with_inf = random.clone();
    with_inf[3] = G1Affine::zero();
    with_inf[100] = G1Affine::zero();
    check(&gpu, &with_inf, &rand_s, c, "bases containing infinity");
    // P*s + (-P)*s cancels to infinity.
    let mut cancel = vec![p, -p];
    let mut cs = vec![rand_s[0], rand_s[0]];
    cancel.extend(random.iter().take(8).copied());
    cs.extend(vec![Fr::zero(); 8]);
    check(&gpu, &cancel, &cs, c, "cancels to the point at infinity");
}

#[test]
fn default_parameters() {
    let gpu = Gpu::new();
    let mut rng = ark_std::test_rng();
    let n = 1 << 17;
    let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let expected = G1Projective::msm(&bases, &scalars).unwrap().into_affine();
    assert_eq!(gpu_msm(&gpu, &bases, &scalars, None), expected);
}
