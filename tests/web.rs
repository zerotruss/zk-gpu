//! Browser test: runs the real NTT and MSM on WebGPU and checks them against arkworks.
//!   wasm-pack test --headless --chrome --release
#![cfg(target_arch = "wasm32")]

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use wasm_bindgen_test::*;
use zk_gpu::{
    gpu::Gpu,
    host,
    msm::{xyzz_to_affine, Msm},
    ntt::Ntt,
};

mod golden_vectors;

wasm_bindgen_test_configure!(run_in_browser);

fn fr_words(v: &[Fr]) -> Vec<u32> {
    v.iter().flat_map(|x| host::words32(&x.into_bigint().0)).collect()
}

#[wasm_bindgen_test]
async fn ntt_matches_arkworks() {
    let gpu = Gpu::new_async().await.expect("WebGPU unavailable");
    let log_n = 12;
    let n = 1usize << log_n;
    let mut rng = ark_std::test_rng();
    let input: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let ntt = Ntt::new(&gpu, log_n);
    let buffer = gpu.buffer(&fr_words(&input));
    ntt.forward(&gpu, &buffer);
    let got = gpu.read_async(&buffer, 8 * n).await;
    let expected = Radix2EvaluationDomain::<Fr>::new(n).unwrap().fft(&input);
    assert_eq!(got, fr_words(&expected), "forward NTT differs from arkworks");
    ntt.inverse(&gpu, &buffer);
    assert_eq!(gpu.read_async(&buffer, 8 * n).await, fr_words(&input), "inverse NTT did not restore the input");
}

#[wasm_bindgen_test]
async fn msm_matches_arkworks() {
    let gpu = Gpu::new_async().await.expect("WebGPU unavailable");
    let n = 1usize << 12;
    let mut rng = ark_std::test_rng();
    let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let plain: Vec<u32> = bases
        .iter()
        .flat_map(|p| {
            let (x, y) = p.xy().unwrap();
            [host::words32(&x.into_bigint().0), host::words32(&y.into_bigint().0)].concat()
        })
        .collect();
    let base_buf = Msm::upload_bases(&gpu, &plain);
    let scalar_buf = gpu.buffer(&fr_words(&scalars));
    let msm = Msm::new(&gpu, n, 11);
    let result = msm.run(&gpu, &base_buf, &scalar_buf);
    let (x, y) = xyzz_to_affine(&gpu.read_async(result, 32).await).unwrap();
    let expected = G1Projective::msm(&bases, &scalars).unwrap().into_affine();
    let fq = |v| ark_bn254::Fq::from_bigint(BigInt::new(v)).unwrap();
    assert_eq!(G1Affine::new(fq(x), fq(y)), expected, "MSM differs from arkworks");
}

/// Independent check: fixed vectors produced by ffjavascript (the library behind snarkjs).
#[wasm_bindgen_test]
async fn ntt_matches_ffjavascript() {
    let gpu = Gpu::new_async().await.expect("WebGPU unavailable");
    for case in golden_vectors::ntt_cases() {
        let ntt = Ntt::new(&gpu, case.log_n);
        let buffer = gpu.buffer(&case.input);
        if case.inverse {
            ntt.inverse(&gpu, &buffer)
        } else {
            ntt.forward(&gpu, &buffer)
        }
        let got = gpu.read_async(&buffer, case.input.len()).await;
        assert!(
            got == case.output,
            "NTT (log_n = {}, inverse = {}) differs from ffjavascript",
            case.log_n,
            case.inverse
        );
    }
}

#[wasm_bindgen_test]
async fn msm_matches_ffjavascript() {
    let gpu = Gpu::new_async().await.expect("WebGPU unavailable");
    for case in golden_vectors::msm_cases() {
        let n = case.scalars.len() / 8;
        let bases = Msm::upload_bases(&gpu, &case.bases);
        let scalars = gpu.buffer(&case.scalars);
        for plan in [None, Some((9, 7)), Some((12, 1024))] {
            let msm = match plan {
                None => Msm::new(&gpu, n, Msm::default_window(n)),
                Some((c, items)) => Msm::with_item_points(&gpu, n, c, items),
            };
            let out = msm.run(&gpu, &bases, &scalars);
            let got = match xyzz_to_affine(&gpu.read_async(out, 32).await) {
                None => vec![0; 16],
                Some((x, y)) => [host::words32(&x), host::words32(&y)].concat(),
            };
            assert!(got == case.result, "MSM '{}' (n = {n}, plan = {plan:?}) differs from ffjavascript", case.name);
        }
    }
}
