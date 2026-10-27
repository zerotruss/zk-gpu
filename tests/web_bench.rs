//! Browser benchmark: WebGPU vs single-threaded WASM CPU (arkworks), printed to the console.
//!   WASM_BINDGEN_TEST_TIMEOUT=900 wasm-pack test --headless --chrome --release --test web_bench -- --nocapture
#![cfg(target_arch = "wasm32")]

use ark_bn254::{Fr, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;
use zk_gpu::{gpu::Gpu, host, msm::Msm, ntt::Ntt};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

#[wasm_bindgen_test]
async fn bench() {
    let gpu = Gpu::new_async().await.expect("WebGPU unavailable");
    console_log!("adapter: {} ({:?})", gpu.adapter.name, gpu.adapter.backend);
    let mut rng = ark_std::test_rng();

    console_log!("-- NTT: log_n | CPU (wasm) ms | WebGPU ms | speedup");
    for log_n in [14u32, 16, 18, 20] {
        let n = 1usize << log_n;
        let mut v: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let t = now();
        domain.fft_in_place(&mut v);
        let cpu = now() - t;

        let ntt = Ntt::new(&gpu, log_n.max(10));
        let buf = gpu.empty_buffer(8 * n);
        ntt.forward(&gpu, &buf);
        gpu.read_async(&buf, 8).await; // warm-up + sync
        let t = now();
        for _ in 0..3 {
            ntt.forward(&gpu, &buf);
            gpu.read_async(&buf, 8).await; // one call at a time: queued calls would hide the per-call cost
        }
        let g = (now() - t) / 3.0;
        console_log!("   {log_n:>2} | {cpu:>9.1} | {g:>8.1} | {:.1}x", cpu / g);
    }

    console_log!("-- MSM: log_n | CPU (wasm) ms | WebGPU ms | speedup");
    let max = 1usize << 18;
    let (mut p, g) = (G1Projective::rand(&mut rng), G1Projective::rand(&mut rng));
    let proj: Vec<G1Projective> = (0..max)
        .map(|_| {
            p += g;
            p
        })
        .collect();
    let bases = G1Projective::normalize_batch(&proj);
    let scalars: Vec<Fr> = (0..max).map(|_| Fr::rand(&mut rng)).collect();
    for log_n in [14u32, 16, 18] {
        let n = 1usize << log_n;
        let t = now();
        let _ = G1Projective::msm(&bases[..n], &scalars[..n]).unwrap().into_affine();
        let cpu = now() - t;

        let plain: Vec<u32> = bases[..n]
            .iter()
            .flat_map(|b| {
                let (x, y) = b.xy().unwrap();
                [host::words32(&x.into_bigint().0), host::words32(&y.into_bigint().0)].concat()
            })
            .collect();
        let sc: Vec<u32> = scalars[..n].iter().flat_map(|s| host::words32(&s.into_bigint().0)).collect();
        let base_buf = Msm::upload_bases(&gpu, &plain);
        let scalar_buf = gpu.buffer(&sc);
        let msm = Msm::new(&gpu, n, Msm::default_window(n));
        gpu.read_async(msm.run(&gpu, &base_buf, &scalar_buf), 32).await; // warm-up
        let t = now();
        for _ in 0..3 {
            gpu.read_async(msm.run(&gpu, &base_buf, &scalar_buf), 32).await;
        }
        let gpu_ms = (now() - t) / 3.0;
        console_log!("   {log_n:>2} | {cpu:>9.1} | {gpu_ms:>8.1} | {:.1}x", cpu / gpu_ms);
    }
}
