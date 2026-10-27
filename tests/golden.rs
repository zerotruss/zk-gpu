#![cfg(not(target_arch = "wasm32"))]
//! Checks the GPU against fixed vectors produced by ffjavascript (the library behind snarkjs).
//! Unlike the other tests this uses no arkworks at all, so it also catches an arkworks bug,
//! or a wrong shared assumption (root of unity, byte order) between arkworks and this library.
mod golden_vectors;
use golden_vectors::{msm_cases, ntt_cases};
use zk_gpu::{
    gpu::Gpu,
    host,
    msm::{xyzz_to_affine, Msm},
    ntt::Ntt,
};

#[test]
fn ntt_matches_ffjavascript() {
    let gpu = Gpu::new();
    for case in ntt_cases() {
        let ntt = Ntt::new(&gpu, case.log_n);
        let buffer = gpu.buffer(&case.input);
        if case.inverse {
            ntt.inverse(&gpu, &buffer)
        } else {
            ntt.forward(&gpu, &buffer)
        }
        let got = gpu.read(&buffer, case.input.len());
        assert!(
            got == case.output,
            "NTT (log_n = {}, inverse = {}) differs from ffjavascript",
            case.log_n,
            case.inverse
        );
    }
}

#[test]
fn msm_matches_ffjavascript() {
    let gpu = Gpu::new();
    for case in msm_cases() {
        let n = case.scalars.len() / 8;
        let bases = Msm::upload_bases(&gpu, &case.bases);
        let scalars = gpu.buffer(&case.scalars);
        // Library defaults, a small window with tiny work items, and a large window.
        for plan in [None, Some((9, 7)), Some((12, 1024))] {
            let msm = match plan {
                None => Msm::new(&gpu, n, Msm::default_window(n)),
                Some((c, items)) => Msm::with_item_points(&gpu, n, c, items),
            };
            let out = msm.run(&gpu, &bases, &scalars);
            let got = match xyzz_to_affine(&gpu.read(out, 32)) {
                None => vec![0; 16],
                Some((x, y)) => [host::words32(&x), host::words32(&y)].concat(),
            };
            assert!(got == case.result, "MSM '{}' (n = {n}, plan = {plan:?}) differs from ffjavascript", case.name);
        }
    }
}
