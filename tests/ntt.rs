#![cfg(not(target_arch = "wasm32"))]
mod common;
use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use common::{fr_from_words, fr_words};
use zk_gpu::{gpu::Gpu, ntt::Ntt};

fn check(gpu: &Gpu, log_n: u32) {
    let n = 1usize << log_n;
    let mut rng = ark_std::test_rng();
    let input: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();

    let ntt = Ntt::new(gpu, log_n);
    let buffer = gpu.buffer(&fr_words(&input));

    ntt.forward(gpu, &buffer);
    let got = fr_from_words(&gpu.read(&buffer, 8 * n));
    assert!(got == domain.fft(&input), "forward NTT mismatch at log_n={log_n}");

    ntt.inverse(gpu, &buffer);
    let back = fr_from_words(&gpu.read(&buffer, 8 * n));
    assert!(back == input, "inverse NTT did not restore the input at log_n={log_n}");
}

#[test]
fn matches_arkworks() {
    let gpu = Gpu::new();
    for log_n in [10, 11, 12, 14, 16, 18, 20, 22] {
        check(&gpu, log_n);
        println!("log_n = {log_n}: ok");
    }
}
