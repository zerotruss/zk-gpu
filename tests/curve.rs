#![cfg(not(target_arch = "wasm32"))]
//! GPU curve arithmetic vs arkworks, including every special case.
use ark_bn254::{Fq, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInt, PrimeField, UniformRand};
use zk_gpu::{field, gpu::Gpu, host};

const KERNEL: &str = "
@group(0) @binding(0) var<storage, read> input: array<vec4<u32>>;       // P.x P.y Q.x Q.y
@group(0) @binding(1) var<storage, read_write> out: array<vec4<u32>>;    // 3 x Xyzz
@group(0) @binding(2) var<storage, read> n: array<u32>;

fn aff(i: u32) -> Affine { return Affine(fq_load(input[i], input[i + 1u]), fq_load(input[i + 2u], input[i + 3u])); }
fn put(i: u32, p: Xyzz) {
  let a = fq_words(p.x); let b = fq_words(p.y); let c = fq_words(p.zz); let d = fq_words(p.zzz);
  out[i] = a.lo; out[i + 1u] = a.hi; out[i + 2u] = b.lo; out[i + 3u] = b.hi;
  out[i + 4u] = c.lo; out[i + 5u] = c.hi; out[i + 6u] = d.lo; out[i + 7u] = d.hi;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x;
  if (c >= n[0]) { return; }
  let p = aff(c * 8u);
  let q = aff(c * 8u + 4u);
  let sum = xyzz_add_affine(xyzz_from_affine(p), q);       // P + Q
  let dbl = xyzz_double(xyzz_from_affine(p));              // 2P
  let both = xyzz_add(sum, dbl);                           // (P + Q) + 2P
  put(c * 24u, sum); put(c * 24u + 8u, dbl); put(c * 24u + 16u, both);
}";

fn mont(x: Fq) -> [u64; 4] {
    host::mul_mod(&x.into_bigint().0, &host::montgomery_r(&field::FQ), &field::FQ)
}

fn words(x: [u64; 4]) -> [u32; 8] {
    host::words32(&x)
}

fn affine_words(p: &G1Affine) -> Vec<u32> {
    let (x, y) = p.xy().map(|(x, y)| (mont(x), mont(y))).unwrap_or(([0; 4], [0; 4]));
    [words(x), words(y)].concat()
}

/// XYZZ (Montgomery words) -> arkworks point.
fn from_xyzz(w: &[u32]) -> G1Affine {
    let f = |i: usize| host::from_montgomery(&host::from_words32(&w[i * 8..i * 8 + 8]), &field::FQ);
    let (x, y, zz, zzz) = (f(0), f(1), f(2), f(3));
    if zz == [0; 4] {
        return G1Affine::zero();
    }
    let fq = |v: [u64; 4]| Fq::from_bigint(BigInt::new(v)).unwrap();
    let (x, y) = (fq(x) / fq(zz), fq(y) / fq(zzz));
    G1Affine::new(x, y)
}

#[test]
fn point_ops_match_arkworks() {
    let mut rng = ark_std::test_rng();
    let inf = G1Affine::zero();
    let p = G1Affine::rand(&mut rng);
    let q = G1Affine::rand(&mut rng);
    // (P, Q) pairs: generic, P = Q, P = -Q, P = inf, Q = inf, both inf, and random ones.
    let mut cases = vec![(p, q), (p, p), (p, -p), (inf, q), (p, inf), (inf, inf)];
    cases.extend((0..20).map(|_| (G1Affine::rand(&mut rng), G1Affine::rand(&mut rng))));

    let gpu = Gpu::new();
    let mut input = Vec::new();
    for (a, b) in &cases {
        input.extend(affine_words(a));
        input.extend(affine_words(b));
    }
    let n = cases.len();
    let pipeline = gpu.pipeline(&format!("{}\n{KERNEL}", zk_gpu::group::Group::G1.wgsl()), "main");
    let (input, out, count) = (gpu.buffer(&input), gpu.empty_buffer(n * 24 * 4), gpu.buffer(&[n as u32]));
    gpu.run(&pipeline, &[&input, &out, &count], n as u32);
    let out = gpu.read(&out, n * 24 * 4);

    for (i, (a, b)) in cases.iter().enumerate() {
        let got = |k: usize| from_xyzz(&out[(i * 24 + k * 8) * 4..(i * 24 + k * 8 + 8) * 4]);
        let sum = (G1Projective::from(*a) + b).into_affine();
        let dbl = (G1Projective::from(*a) + a).into_affine();
        assert_eq!(got(0), sum, "case {i}: P + Q");
        assert_eq!(got(1), dbl, "case {i}: 2P");
        assert_eq!(got(2), (G1Projective::from(sum) + dbl).into_affine(), "case {i}: (P+Q) + 2P");
    }
}
