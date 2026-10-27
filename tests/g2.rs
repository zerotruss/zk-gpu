#![cfg(not(target_arch = "wasm32"))]
//! The G2 MSM (points over Fq2) against arkworks.
mod common;
use ark_bn254::{Fq, Fq2, Fr, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, Field, PrimeField, UniformRand, Zero};
use common::*;
use zk_gpu::{
    gpu::Gpu,
    group::Group,
    host,
    msm::{xyzz_to_affine_g2, Fq2Words, Msm},
};

fn coordinate_words(c: &Fq2) -> Vec<u32> {
    [c.c0, c.c1].iter().flat_map(|f| host::words32(&f.into_bigint().0)).collect()
}

/// x then y, each as (a, b); the point at infinity is all zeros.
fn point_words_g2(p: &G2Affine) -> Vec<u32> {
    match p.xy() {
        Some((x, y)) => [coordinate_words(&x), coordinate_words(&y)].concat(),
        None => vec![0; 32],
    }
}

fn affine_g2(coords: Option<(Fq2Words, Fq2Words)>) -> G2Affine {
    let Some((x, y)) = coords else { return G2Affine::zero() };
    let fq = |v: [u64; 4]| Fq::from_bigint(BigInt::new(v)).expect("coordinate >= p");
    G2Affine::new(Fq2::new(fq(x[0]), fq(x[1])), Fq2::new(fq(y[0]), fq(y[1])))
}

fn gpu_msm_g2(gpu: &Gpu, bases: &[G2Affine], scalars: &[Fr], item_points: Option<u32>, c: u32) -> G2Affine {
    let plain: Vec<u32> = bases.iter().flat_map(point_words_g2).collect();
    let base_buf = Msm::upload_bases(gpu, &plain);
    let scalar_buf = gpu.buffer(&fr_words(scalars));
    let msm = Msm::for_group(gpu, Group::G2, bases.len(), c, item_points);
    let result = msm.run(gpu, &base_buf, &scalar_buf);
    affine_g2(xyzz_to_affine_g2(&gpu.read(result, Group::G2.xyzz_words())))
}

fn reference(bases: &[G2Affine], scalars: &[Fr]) -> G2Affine {
    G2Projective::msm(bases, scalars).unwrap().into_affine()
}

fn random_points(rng: &mut impl ark_std::rand::Rng, n: usize) -> Vec<G2Affine> {
    (0..n).map(|_| G2Affine::rand(rng)).collect()
}

#[test]
fn g2_msm_matches_arkworks() {
    let gpu = Gpu::new();
    let mut rng = rng(21);
    for n in [1usize, 2, 3, 17, 255, 1000, 4000] {
        let bases = random_points(&mut rng, n);
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        for (c, items) in [(13u32, None), (9, Some(1)), (10, Some(64))] {
            assert_eq!(gpu_msm_g2(&gpu, &bases, &scalars, items, c), reference(&bases, &scalars), "n = {n}, c = {c}");
        }
    }
}

#[test]
fn g2_msm_handles_special_scalars_and_infinity() {
    let gpu = Gpu::new();
    let mut rng = rng(22);
    let n = 600;
    let mut bases = random_points(&mut rng, n);
    bases[7] = G2Affine::zero();
    bases[8] = bases[9]; // equal points force the doubling branch of the addition
    let two = Fr::from(2u64);
    let mut scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    scalars[0] = Fr::zero();
    scalars[1] = Fr::from(1u64);
    scalars[2] = -Fr::from(1u64);
    scalars[3] = two.pow([200u64]);
    scalars[8] = Fr::from(5u64);
    scalars[9] = Fr::from(5u64);
    assert_eq!(gpu_msm_g2(&gpu, &bases, &scalars, None, 10), reference(&bases, &scalars));
    // All-zero scalars sum to infinity.
    let zeros = vec![Fr::zero(); n];
    assert!(gpu_msm_g2(&gpu, &bases, &zeros, None, 10).is_zero());
    // P + (-P) = infinity
    let p = bases[20];
    let s = Fr::rand(&mut rng);
    assert!(gpu_msm_g2(&gpu, &[p, p], &[s, -s], None, 9).is_zero());
}

#[test]
fn g2_bases_in_ffjavascript_montgomery_form_work() {
    let gpu = Gpu::new();
    let mut rng = rng(23);
    let r256 = Fq::from(2u64).pow([256u64]);
    let n = 500;
    let mut bases = random_points(&mut rng, n);
    bases[3] = G2Affine::zero();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let words: Vec<u32> = bases
        .iter()
        .flat_map(|p| match p.xy() {
            Some((x, y)) => vec![x.c0 * r256, x.c1 * r256, y.c0 * r256, y.c1 * r256],
            None => vec![Fq::zero(); 4],
        })
        .flat_map(|c| host::words32(&c.into_bigint().0))
        .collect();
    let buffer = Msm::upload_bases_montgomery_256(&gpu, &words);
    let scalar_buf = gpu.buffer(&fr_words(&scalars));
    let msm = Msm::for_group(&gpu, Group::G2, n, 10, None);
    let got = affine_g2(xyzz_to_affine_g2(&gpu.read(msm.run(&gpu, &buffer, &scalar_buf), 64)));
    assert_eq!(got, reference(&bases, &scalars));
}

#[test]
fn g2_msm_with_huge_buckets() {
    let gpu = Gpu::new();
    let mut rng = rng(24);
    for n in [300usize, 3000] {
        let bases = random_points(&mut rng, n);
        let same = vec![Fr::from(7u64); n];
        let mostly_equal: Vec<Fr> =
            (0..n).map(|i| if i % 50 == 0 { Fr::rand(&mut rng) } else { Fr::from(3u64) }).collect();
        for scalars in [&same, &mostly_equal] {
            for (c, items) in [(9u32, Some(1)), (10, Some(3)), (13, None)] {
                assert_eq!(gpu_msm_g2(&gpu, &bases, scalars, items, c), reference(&bases, scalars), "n = {n}, c = {c}");
            }
        }
    }
}

#[test]
fn g2_msm_cut_into_several_gpu_submissions() {
    // G2 costs about 5x more per point, so 2^16 points already need several slices.
    let gpu = Gpu::new();
    let mut rng = rng(25);
    let n = 1 << 16;
    let g = G2Projective::from(G2Affine::generator());
    let mut acc = g;
    let points: Vec<G2Projective> = (0..n)
        .map(|_| {
            let p = acc;
            acc += g;
            p
        })
        .collect();
    let bases = G2Projective::normalize_batch(&points);
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    assert_eq!(gpu_msm_g2(&gpu, &bases, &scalars, None, 13), reference(&bases, &scalars));
}
