//! Wrong sizes must fail loudly: the GPU reads zeros past the end of a buffer, so without these
//! checks a too-small buffer silently gives a wrong answer.
#![cfg(not(target_arch = "wasm32"))]
use zk_gpu::{gpu::Gpu, msm::Msm, ntt::Ntt};

const N: usize = 4096;

#[test]
#[should_panic(expected = "at least one point")]
fn msm_with_no_points() {
    Msm::new(&Gpu::new(), 0, 10);
}

#[test]
#[should_panic(expected = "window size c must be in 9..=20")]
fn msm_window_out_of_range() {
    Msm::new(&Gpu::new(), N, 8);
}

#[test]
#[should_panic(expected = "scalars buffer holds")]
fn msm_scalars_too_small() {
    let gpu = Gpu::new();
    let bases = Msm::upload_bases(&gpu, &vec![0u32; 16 * N]);
    let scalars = gpu.buffer(&vec![1u32; 4 * N]); // half of what 4096 scalars need
    Msm::new(&gpu, N, 10).run(&gpu, &bases, &scalars);
}

#[test]
#[should_panic(expected = "bases buffer holds")]
fn msm_bases_too_small() {
    let gpu = Gpu::new();
    let bases = Msm::upload_bases(&gpu, &vec![0u32; 8 * N]); // half of what 4096 points need
    let scalars = gpu.buffer(&vec![1u32; 8 * N]);
    Msm::new(&gpu, N, 10).run(&gpu, &bases, &scalars);
}

#[test]
#[should_panic(expected = "log_n must be in 10..=27")]
fn ntt_size_out_of_range() {
    Ntt::new(&Gpu::new(), 9);
}

#[test]
#[should_panic(expected = "data buffer holds")]
fn ntt_data_too_small() {
    let gpu = Gpu::new();
    let data = gpu.buffer(&vec![1u32; 8 << 11]); // half of a 2^12 transform
    Ntt::new(&gpu, 12).forward(&gpu, &data);
}

#[test]
fn buffers_larger_than_needed_are_accepted() {
    let gpu = Gpu::new();
    let data = gpu.buffer(&vec![1u32; 8 << 13]); // room for 2^13, transform of 2^12
    let ntt = Ntt::new(&gpu, 12);
    ntt.forward(&gpu, &data);
    ntt.inverse(&gpu, &data);
}
