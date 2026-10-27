//! The smallest complete use of zk-gpu: one MSM and one NTT, each checked against arkworks.
//!   cargo run --release --example quickstart
use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, BigInteger, PrimeField, UniformRand};
use zk_gpu::{
    gpu::Gpu,
    host,
    msm::{xyzz_to_affine, Msm},
    ntt::Ntt,
};

/// A field element is 8 little-endian u32 words (plain integer, fully reduced).
fn words(x: impl PrimeField) -> Vec<u32> {
    x.into_bigint().to_bytes_le().chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    let gpu = Gpu::new();
    let mut rng = ark_std::test_rng();

    // NTT: transform 2^12 random values forward and back; we must get the input again.
    let log_n = 12;
    let input: Vec<Fr> = (0..1 << log_n).map(|_| Fr::rand(&mut rng)).collect();
    let ntt = Ntt::new(&gpu, log_n);
    let data = gpu.buffer(&input.iter().flat_map(|x| words(*x)).collect::<Vec<u32>>());
    ntt.forward(&gpu, &data);
    ntt.inverse(&gpu, &data);
    let output: Vec<Fr> = gpu
        .read(&data, 8 << log_n)
        .chunks(8)
        .map(|w| Fr::from_bigint(BigInt::new(host::from_words32(w))).unwrap())
        .collect();
    assert_eq!(output, input);
    println!("NTT 2^{log_n}: forward then inverse gives the input back");

    // MSM: sum of scalar_i * point_i over 4096 random pairs, against arkworks on the CPU.
    let n = 4096;
    let points: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let point_words: Vec<u32> = points
        .iter()
        .flat_map(|p| {
            let (x, y) = p.xy().unwrap();
            [words(x), words(y)].concat()
        })
        .collect();
    let bases = Msm::upload_bases(&gpu, &point_words); // convert once; reuse for many proofs
    let scalar_buf = gpu.buffer(&scalars.iter().flat_map(|s| words(*s)).collect::<Vec<u32>>());
    let msm = Msm::new(&gpu, n, Msm::default_window(n));
    let result = msm.run(&gpu, &bases, &scalar_buf);
    let (x, y) = xyzz_to_affine(&gpu.read(result, 32)).expect("not the point at infinity");
    let fq = |v| Fq::from_bigint(BigInt::new(v)).unwrap();
    let gpu_point = G1Affine::new(fq(x), fq(y));
    assert_eq!(gpu_point, G1Projective::msm(&points, &scalars).unwrap().into_affine());
    println!("MSM n = {n}: the GPU result equals arkworks");
}
