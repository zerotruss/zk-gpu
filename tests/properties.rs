#![cfg(not(target_arch = "wasm32"))]
//! Mathematical properties of the NTT and MSM that must hold for ANY correct implementation.
//! They need no reference implementation of the operation under test: the arkworks calls
//! below are only used for ordinary field and curve arithmetic on the host.
mod common;
use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{FftField, Field, One, PrimeField, UniformRand, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::rand::{seq::SliceRandom, Rng};
use common::*;
use zk_gpu::{
    gpu::Gpu,
    host,
    msm::{xyzz_to_affine, Msm},
    ntt::Ntt,
};

// ---------------------------------------------------------------- NTT

fn random_fr(rng: &mut impl Rng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::rand(rng)).collect()
}

#[test]
fn ntt_of_a_delta_is_a_power_sequence() {
    // F(delta_k)[j] = w^(j*k), and F(constant c) = [n*c, 0, 0, ...].
    let gpu = Gpu::new();
    for log_n in [10u32, 11, 13] {
        let n = 1usize << log_n;
        let ntt = Ntt::new(&gpu, log_n);
        let w = Fr::get_root_of_unity(n as u64).unwrap();
        for k in [0usize, 1, 2, 511, 512, n - 1] {
            let mut delta = vec![Fr::zero(); n];
            delta[k] = Fr::one();
            let got = gpu_ntt_forward(&gpu, &ntt, &delta);
            let expected: Vec<Fr> = (0..n).map(|j| w.pow([(j * k) as u64])).collect();
            assert!(got == expected, "NTT of delta_{k} wrong at log_n={log_n}");
        }
        let c = Fr::from(123456789u64);
        let got = gpu_ntt_forward(&gpu, &ntt, &vec![c; n]);
        let mut expected = vec![Fr::zero(); n];
        expected[0] = c * Fr::from(n as u64);
        assert!(got == expected, "NTT of a constant wrong at log_n={log_n}");
    }
}

#[test]
fn ntt_is_linear() {
    let gpu = Gpu::new();
    let mut rng = rng(1);
    for log_n in [10u32, 12, 14] {
        let n = 1usize << log_n;
        let ntt = Ntt::new(&gpu, log_n);
        let (a, b) = (random_fr(&mut rng, n), random_fr(&mut rng, n));
        let c = Fr::rand(&mut rng);
        let combined: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| *x * c + y).collect();
        let (fa, fb) = (gpu_ntt_forward(&gpu, &ntt, &a), gpu_ntt_forward(&gpu, &ntt, &b));
        let expected: Vec<Fr> = fa.iter().zip(&fb).map(|(x, y)| *x * c + y).collect();
        assert!(gpu_ntt_forward(&gpu, &ntt, &combined) == expected, "linearity failed at log_n={log_n}");
    }
}

#[test]
fn ntt_convolution_theorem() {
    // inverse(F(a) * F(b)) must equal the cyclic convolution computed the slow, obvious way.
    let gpu = Gpu::new();
    let mut rng = rng(2);
    let log_n = 10;
    let n = 1usize << log_n;
    let ntt = Ntt::new(&gpu, log_n);
    let (a, b) = (random_fr(&mut rng, n), random_fr(&mut rng, n));
    let mut expected = vec![Fr::zero(); n];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            expected[(i + j) % n] += *x * y;
        }
    }
    let product: Vec<Fr> =
        gpu_ntt_forward(&gpu, &ntt, &a).iter().zip(gpu_ntt_forward(&gpu, &ntt, &b)).map(|(x, y)| *x * y).collect();
    assert!(gpu_ntt_inverse(&gpu, &ntt, &product) == expected, "convolution theorem failed");
}

#[test]
fn ntt_round_trips_in_both_directions() {
    let gpu = Gpu::new();
    let mut rng = rng(3);
    for log_n in [10u32, 15, 18] {
        let n = 1usize << log_n;
        let ntt = Ntt::new(&gpu, log_n);
        let x = random_fr(&mut rng, n);
        assert!(
            gpu_ntt_inverse(&gpu, &ntt, &gpu_ntt_forward(&gpu, &ntt, &x)) == x,
            "inverse(forward(x)) != x at {log_n}"
        );
        assert!(
            gpu_ntt_forward(&gpu, &ntt, &gpu_ntt_inverse(&gpu, &ntt, &x)) == x,
            "forward(inverse(x)) != x at {log_n}"
        );
    }
}

#[test]
fn ntt_extreme_values() {
    // Zeros, ones, r - 1 and mixes stress the modular reduction at its boundaries.
    let gpu = Gpu::new();
    let log_n = 12;
    let n = 1usize << log_n;
    let ntt = Ntt::new(&gpu, log_n);
    let mut rng = rng(4);
    let minus_one = -Fr::one();
    let mixed: Vec<Fr> = (0..n).map(|_| [Fr::zero(), Fr::one(), minus_one][rng.gen_range(0..3)]).collect();
    for (name, input) in
        [("zeros", vec![Fr::zero(); n]), ("ones", vec![Fr::one(); n]), ("r-1", vec![minus_one; n]), ("mixed", mixed)]
    {
        let got = gpu_ntt_forward(&gpu, &ntt, &input);
        let expected = Radix2EvaluationDomain::<Fr>::new(n).unwrap().fft(&input);
        assert!(got == expected, "extreme input '{name}' wrong");
        assert!(gpu_ntt_inverse(&gpu, &ntt, &got) == input, "extreme input '{name}' does not round-trip");
    }
}

#[test]
fn ntt_plan_can_be_reused() {
    // Leftover state in the scratch buffer or twiddles would show up as a wrong second result.
    let gpu = Gpu::new();
    let mut rng = rng(5);
    let log_n = 13;
    let n = 1usize << log_n;
    let ntt = Ntt::new(&gpu, log_n);
    let inputs: Vec<Vec<Fr>> = (0..4).map(|_| random_fr(&mut rng, n)).collect();
    let first: Vec<Vec<Fr>> = inputs.iter().map(|x| gpu_ntt_forward(&gpu, &ntt, x)).collect();
    for (i, x) in inputs.iter().enumerate().rev() {
        assert!(gpu_ntt_forward(&gpu, &ntt, x) == first[i], "second run of input {i} differs from the first");
    }
    // Two independent plans of the same size agree.
    assert!(gpu_ntt_forward(&gpu, &Ntt::new(&gpu, log_n), &inputs[0]) == first[0]);
}

// ---------------------------------------------------------------- MSM

fn random_points(rng: &mut impl Rng, n: usize) -> Vec<G1Affine> {
    (0..n).map(|_| G1Affine::rand(rng)).collect()
}

fn msm_ref(bases: &[G1Affine], scalars: &[Fr]) -> G1Affine {
    G1Projective::msm(bases, scalars).unwrap().into_affine()
}

#[test]
fn msm_is_linear_in_the_scalars() {
    let gpu = Gpu::new();
    let mut rng = rng(10);
    let n = 3000;
    let bases = random_points(&mut rng, n);
    let (s, t) = (random_fr(&mut rng, n), random_fr(&mut rng, n));
    let sum: Vec<Fr> = s.iter().zip(&t).map(|(a, b)| *a + b).collect();
    let expected = (gpu_msm(&gpu, &bases, &s, None) + gpu_msm(&gpu, &bases, &t, None)).into_affine();
    assert_eq!(gpu_msm(&gpu, &bases, &sum, None), expected, "MSM(s + t) != MSM(s) + MSM(t)");
}

#[test]
fn msm_is_linear_in_the_bases() {
    let gpu = Gpu::new();
    let mut rng = rng(11);
    let n = 3000;
    let (p, q) = (random_points(&mut rng, n), random_points(&mut rng, n));
    let s = random_fr(&mut rng, n);
    let sum: Vec<G1Affine> = p.iter().zip(&q).map(|(a, b)| (*a + b).into_affine()).collect();
    let expected = (gpu_msm(&gpu, &p, &s, None) + gpu_msm(&gpu, &q, &s, None)).into_affine();
    assert_eq!(gpu_msm(&gpu, &sum, &s, None), expected, "MSM(P + Q) != MSM(P) + MSM(Q)");
}

#[test]
fn msm_scales_with_a_constant() {
    let gpu = Gpu::new();
    let mut rng = rng(12);
    let n = 3000;
    let bases = random_points(&mut rng, n);
    let s = random_fr(&mut rng, n);
    let c = Fr::rand(&mut rng);
    let scaled: Vec<Fr> = s.iter().map(|x| *x * c).collect();
    let expected = (gpu_msm(&gpu, &bases, &s, None) * c).into_affine();
    assert_eq!(gpu_msm(&gpu, &bases, &scaled, None), expected, "MSM(c * s) != c * MSM(s)");
}

#[test]
fn msm_ignores_the_order_of_the_pairs() {
    let gpu = Gpu::new();
    let mut rng = rng(13);
    let n = 5000;
    let bases = random_points(&mut rng, n);
    let s = random_fr(&mut rng, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut rng);
    let (b2, s2): (Vec<G1Affine>, Vec<Fr>) = order.iter().map(|&i| (bases[i], s[i])).unzip();
    assert_eq!(
        gpu_msm(&gpu, &b2, &s2, None),
        gpu_msm(&gpu, &bases, &s, None),
        "shuffling the pairs changed the result"
    );
}

#[test]
fn msm_of_tiny_inputs() {
    // Sizes that are not powers of two, down to a single point; and the n = 1 identity MSM = s * P.
    let gpu = Gpu::new();
    let mut rng = rng(14);
    for n in [1usize, 2, 3, 7, 17, 255, 256, 257, 1023, 1025] {
        let bases = random_points(&mut rng, n);
        let s = random_fr(&mut rng, n);
        for plan in [None, Some((9, 1024)), Some((9, 1))] {
            assert_eq!(gpu_msm(&gpu, &bases, &s, plan), msm_ref(&bases, &s), "n = {n}, plan = {plan:?}");
        }
    }
    let p = G1Affine::rand(&mut rng);
    let s = Fr::rand(&mut rng);
    assert_eq!(gpu_msm(&gpu, &[p], &[s], None), (p * s).into_affine(), "single-point MSM != s * P");
}

#[test]
fn bases_already_in_montgomery_form_give_the_same_msm() {
    // snarkjs/ffjavascript keep points as x * 2^256 mod p. They must be accepted as they are,
    // including the point at infinity (0, 0).
    let gpu = Gpu::new();
    let mut rng = rng(15);
    let r256 = Fq::from(2u64).pow([256u64]);
    for n in [1usize, 5, 300, 2000] {
        let mut bases = random_points(&mut rng, n);
        bases[n / 2] = G1Affine::zero();
        let scalars = random_fr(&mut rng, n);
        let words: Vec<u32> = bases
            .iter()
            .flat_map(|p| match p.xy() {
                Some((x, y)) => [x * r256, y * r256],
                None => (Fq::zero(), Fq::zero()).into(),
            })
            .flat_map(|c| host::words32(&c.into_bigint().0))
            .collect();
        let buffer = Msm::upload_bases_montgomery_256(&gpu, &words);
        let scalar_buf = gpu.buffer(&fr_words(&scalars));
        let msm = Msm::new(&gpu, n, Msm::default_window(n));
        let got = affine_from_coords(xyzz_to_affine(&gpu.read(msm.run(&gpu, &buffer, &scalar_buf), 32)));
        assert_eq!(got, msm_ref(&bases, &scalars), "n = {n}");
    }
}

#[test]
fn msm_with_huge_buckets() {
    // Equal scalars put every point into one bucket; with one point per work item that is
    // thousands of items, which the merge rounds must sum in the right order and amount.
    let gpu = Gpu::new();
    let mut rng = rng(16);
    for n in [17usize, 300, 5000] {
        let bases = random_points(&mut rng, n);
        let same = vec![Fr::from(7u64); n];
        let ones = vec![Fr::one(); n];
        let mostly_equal: Vec<Fr> =
            (0..n).map(|i| if i % 50 == 0 { Fr::rand(&mut rng) } else { Fr::from(3u64) }).collect();
        for scalars in [&same, &ones, &mostly_equal] {
            for plan in [Some((9, 1)), Some((10, 3)), None] {
                assert_eq!(gpu_msm(&gpu, &bases, scalars, plan), msm_ref(&bases, scalars), "n = {n}, plan = {plan:?}");
            }
        }
    }
}

#[test]
fn msm_with_a_nearly_empty_top_window() {
    // 255 = 14 * 18 + 3: with 14-bit windows the top window has 3 bits, so all points crowd into
    // a handful of buckets of thousands of points each.
    let gpu = Gpu::new();
    let mut rng = rng(17);
    let n = 20_000;
    let bases = random_points(&mut rng, n);
    let scalars = random_fr(&mut rng, n);
    for plan in [Some((14, 8)), Some((14, 1)), Some((11, 8))] {
        assert_eq!(gpu_msm(&gpu, &bases, &scalars, plan), msm_ref(&bases, &scalars), "plan = {plan:?}");
    }
}

/// Points that are cheap to produce: multiples of the generator.
fn generator_multiples(n: usize) -> Vec<G1Affine> {
    let g = G1Projective::from(G1Affine::generator());
    let mut acc = g;
    let points: Vec<G1Projective> = (0..n)
        .map(|_| {
            let p = acc;
            acc += g;
            p
        })
        .collect();
    G1Projective::normalize_batch(&points)
}

#[test]
fn msm_cut_into_several_gpu_submissions() {
    // Big enough that the accumulate and bucket-reduction stages are cut into several slices
    // (see ACCUMULATE_BUDGET in msm.rs); the answer must not depend on how they are cut.
    let gpu = Gpu::new();
    let mut rng = rng(18);
    let n = 1 << 18;
    let bases = generator_multiples(n);
    let scalars = random_fr(&mut rng, n);
    let want = msm_ref(&bases, &scalars);
    for plan in [None, Some((15, 8))] {
        assert_eq!(gpu_msm(&gpu, &bases, &scalars, plan), want, "plan = {plan:?}");
    }
}

/// Scalars that sit exactly on the signed-digit window boundaries for window size `c`:
/// 2^k - 1, 2^k, 2^k + 1 for every k that is a multiple of c or one less, plus r - 1, r - 2.
fn boundary_scalars(c: u32) -> Vec<Fr> {
    let two = Fr::from(2u64);
    let mut out = vec![Fr::zero(), Fr::one(), -Fr::one(), -Fr::from(2u64), -two.pow([c as u64])];
    for j in 1..=254u32.div_ceil(c) {
        for k in [c * j - 1, c * j] {
            let p = two.pow([k as u64]);
            out.extend([p - Fr::one(), p, p + Fr::one()]);
        }
    }
    out
}

#[test]
fn msm_at_window_boundaries() {
    let gpu = Gpu::new();
    let mut rng = rng(15);
    for c in 9..=16u32 {
        let mut scalars = boundary_scalars(c);
        scalars.extend(boundary_scalars(c)); // every value twice, so buckets hold several points
        let bases = random_points(&mut rng, scalars.len());
        let expected = msm_ref(&bases, &scalars);
        for item_points in [1024, 3] {
            assert_eq!(
                gpu_msm(&gpu, &bases, &scalars, Some((c, item_points))),
                expected,
                "c = {c}, item_points = {item_points}"
            );
        }
    }
}

#[test]
fn msm_result_does_not_depend_on_the_window_size() {
    let gpu = Gpu::new();
    let mut rng = rng(16);
    let n = 6000;
    let bases = random_points(&mut rng, n);
    let s = random_fr(&mut rng, n);
    let expected = msm_ref(&bases, &s);
    for c in 9..=16u32 {
        assert_eq!(gpu_msm(&gpu, &bases, &s, Some((c, 1024))), expected, "window size c = {c}");
    }
}

#[test]
fn msm_plan_can_be_reused() {
    // One Msm object, several different scalar sets: leftover bucket state would corrupt later runs.
    let gpu = Gpu::new();
    let mut rng = rng(17);
    let n = 4000;
    let bases = random_points(&mut rng, n);
    let base_buf = Msm::upload_bases(&gpu, &bases.iter().flat_map(point_words).collect::<Vec<_>>());
    let msm = Msm::new(&gpu, n, 11);
    let sets: Vec<Vec<Fr>> =
        vec![random_fr(&mut rng, n), vec![Fr::zero(); n], vec![Fr::one(); n], random_fr(&mut rng, n)];
    for round in 0..2 {
        for (i, s) in sets.iter().enumerate() {
            let out = msm.run(&gpu, &base_buf, &gpu.buffer(&fr_words(s)));
            let got = affine_from_coords(zk_gpu::msm::xyzz_to_affine(&gpu.read(out, 32)));
            assert_eq!(got, msm_ref(&bases, s), "round {round}, scalar set {i}");
        }
    }
}

/// One randomized MSM case: random size, window, work-item size, and scalar/base distributions
/// chosen to hit the awkward shapes (many zeros, many duplicates, tiny scalars, infinities).
fn fuzz_case(gpu: &Gpu, seed: u64) {
    let mut rng = rng(seed);
    let n = [rng.gen_range(1..64), rng.gen_range(1..1500), rng.gen_range(1..6000)][rng.gen_range(0..3)];
    let c = rng.gen_range(9..=16u32);
    let item_points = [1u32, 2, 5, 64, 1024][rng.gen_range(0..5)];
    let scalar_kind = rng.gen_range(0..5);
    let base_kind = rng.gen_range(0..3);

    let pool = random_points(&mut rng, 4);
    let bases: Vec<G1Affine> = (0..n)
        .map(|_| match base_kind {
            0 => G1Affine::rand(&mut rng),
            1 => pool[rng.gen_range(0..pool.len())], // few distinct points: doublings and cancellations
            _ if rng.gen_bool(0.1) => G1Affine::zero(),
            _ => G1Affine::rand(&mut rng),
        })
        .collect();
    let scalars: Vec<Fr> = (0..n)
        .map(|_| match scalar_kind {
            0 => Fr::rand(&mut rng),
            1 => Fr::from(rng.gen_range(0..16u64)),
            2 => {
                if rng.gen_bool(0.9) {
                    Fr::zero()
                } else {
                    Fr::rand(&mut rng)
                }
            }
            3 => Fr::from(rng.gen::<u64>()),
            _ => [Fr::zero(), Fr::one(), -Fr::one()][rng.gen_range(0..3)],
        })
        .collect();
    let got = gpu_msm(gpu, &bases, &scalars, Some((c, item_points)));
    assert_eq!(
        got,
        msm_ref(&bases, &scalars),
        "FUZZ FAILURE: rerun with ZK_GPU_FUZZ_SEED={seed} (n={n}, c={c}, item_points={item_points}, scalars={scalar_kind}, bases={base_kind})"
    );
}

#[test]
fn msm_fuzz() {
    // ZK_GPU_FUZZ_SEED=<n> reruns one case; ZK_GPU_FUZZ_ITERS=<count> sets the number of cases.
    let gpu = Gpu::new();
    if let Ok(seed) = std::env::var("ZK_GPU_FUZZ_SEED") {
        return fuzz_case(&gpu, seed.parse().expect("ZK_GPU_FUZZ_SEED must be a number"));
    }
    let default_iters = if quick() { 20 } else { 200 };
    let iters: u64 = std::env::var("ZK_GPU_FUZZ_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(default_iters);
    for seed in 1000..1000 + iters {
        fuzz_case(&gpu, seed);
    }
}

#[test]
fn the_msm_oracle_itself_agrees_with_naive_scalar_multiplication() {
    // Guards the guard: arkworks' MSM must equal the sum of individual scalar multiplications.
    let mut rng = rng(18);
    let bases = random_points(&mut rng, 50);
    let s = random_fr(&mut rng, 50);
    let naive = bases.iter().zip(&s).fold(G1Projective::zero(), |acc, (p, k)| acc + *p * k).into_affine();
    assert_eq!(msm_ref(&bases, &s), naive);
    assert!(msm_ref(&bases, &vec![Fr::zero(); 50]).is_zero());
}
