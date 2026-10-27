//! Number-theoretic transform over the BN254 scalar field.
//!
//! Decimation-in-frequency radix-2, executed as a few passes; each pass fuses up to 9
//! stages in workgroup memory. Data stays in plain (non-Montgomery) form: twiddles carry
//! the Montgomery factor, so `mul(x, twiddle)` returns exactly `x * w`.

use crate::{field, gpu::Gpu, host};

const TILE_LOG: u32 = 9; // 512 elements = 16 KB of workgroup memory (WebGPU baseline limit)

struct Direction {
    twiddles: wgpu::Buffer,
    params: Vec<wgpu::Buffer>, // one per pass
}

pub struct Ntt {
    pipeline: wgpu::ComputePipeline,
    scratch: wgpu::Buffer,
    forward: Direction,
    inverse: Direction,
    pub log_n: u32,
}

impl Ntt {
    /// Plan for transforms of size 2^log_n (10 <= log_n <= 27). Buffers are 8 u32 per element.
    pub fn new(gpu: &Gpu, log_n: u32) -> Self {
        assert!((TILE_LOG + 1..=27).contains(&log_n), "log_n must be in 10..=27");
        let source = format!("{}\n{}", field::wgsl("fr", field::FR), include_str!("shaders/ntt.wgsl"));
        let r = host::montgomery_r(&field::FR);
        let w = host::root_of_unity(log_n);
        let w_inv = host::inverse(&w, &field::FR);
        let n_inv = host::mul_mod(&host::inverse(&[1u64 << log_n, 0, 0, 0], &field::FR), &r, &field::FR);
        Self {
            pipeline: gpu.pipeline(&source, "ntt_pass"),
            scratch: gpu.empty_buffer(8 << log_n),
            forward: Direction::new(gpu, log_n, &w, None),
            inverse: Direction::new(gpu, log_n, &w_inv, Some(n_inv)),
            log_n,
        }
    }

    /// In-place forward transform: natural order in, natural order out.
    pub fn forward(&self, gpu: &Gpu, data: &wgpu::Buffer) {
        self.run(gpu, &self.forward, data);
    }

    /// In-place inverse transform (includes the 1/n scaling).
    pub fn inverse(&self, gpu: &Gpu, data: &wgpu::Buffer) {
        self.run(gpu, &self.inverse, data);
    }

    fn run(&self, gpu: &Gpu, dir: &Direction, data: &wgpu::Buffer) {
        let need = 32u64 << self.log_n;
        assert!(
            data.size() >= need,
            "data buffer holds {} bytes; a 2^{} transform needs {need}",
            data.size(),
            self.log_n
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let groups = 1u32 << (self.log_n - TILE_LOG);
        for (i, params) in dir.params.iter().enumerate() {
            // Ping-pong; an even pass count makes the last pass land back in `data`.
            let (src, dst) = if i % 2 == 0 { (data, &self.scratch) } else { (&self.scratch, data) };
            gpu.encode(&mut encoder, &self.pipeline, &[src, &dir.twiddles, params, dst], groups);
        }
        gpu.queue.submit([encoder.finish()]);
    }
}

impl Direction {
    fn new(gpu: &Gpu, log_n: u32, root: &host::U256, scale: Option<host::U256>) -> Self {
        // Split the log_n stages over an even number of passes of at most TILE_LOG stages.
        let mut passes = log_n.div_ceil(TILE_LOG);
        passes += passes % 2;
        let mut params = Vec::new();
        let mut remaining = log_n;
        for i in 0..passes {
            let s = log_n / passes + u32::from(i < log_n % passes);
            remaining -= s;
            let last = i == passes - 1;
            let mut p = vec![remaining, s, log_n, last as u32, scale.is_some() as u32 & last as u32];
            p.extend(host::words32(&scale.unwrap_or([0; 4])));
            params.push(gpu.buffer(&p));
        }
        Self { twiddles: twiddle_table(gpu, log_n, root), params }
    }
}

/// Table of root^i (Montgomery form), i < n/2. Built as hi[i / L] * lo[i % L] on the GPU,
/// with the two small tables computed here.
fn twiddle_table(gpu: &Gpu, log_n: u32, root: &host::U256) -> wgpu::Buffer {
    let m = &field::FR;
    let r = host::montgomery_r(m);
    let count = 1usize << (log_n - 1);
    let lo_bits = (log_n - 1).div_ceil(2);
    let lo_len = 1usize << lo_bits;
    let powers = |step: &host::U256, len: usize| -> Vec<u32> {
        let mut x = [1, 0, 0, 0];
        let mut out = Vec::new();
        for _ in 0..len {
            out.extend(host::words32(&host::mul_mod(&x, &r, m)));
            x = host::mul_mod(&x, step, m);
        }
        out
    };
    let lo = powers(root, lo_len);
    let hi = powers(&host::pow_mod(root, &[lo_len as u64, 0, 0, 0], m), count.div_ceil(lo_len));

    let source = format!("{}\n{}", field::wgsl("fr", field::FR), include_str!("shaders/twiddles.wgsl"));
    let pipeline = gpu.pipeline(&source, "twiddles");
    let out = gpu.empty_buffer(8 * count);
    let params = gpu.buffer(&[lo_bits, count as u32]);
    let (hi, lo) = (gpu.buffer(&hi), gpu.buffer(&lo));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    gpu.encode(&mut encoder, &pipeline, &[&hi, &lo, &out, &params], (count as u32).div_ceil(64));
    gpu.queue.submit([encoder.finish()]);
    out
}
