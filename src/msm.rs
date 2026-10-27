//! Multi-scalar multiplication on BN254 G1: signed-digit Pippenger, entirely on the GPU.
//!
//! A single GPU thread is slow (one point addition takes tens of microseconds of latency),
//! so every stage is built from short per-thread chains and lots of threads:
//!  1. scalars become signed c-bit digits; a counting sort (atomics + prefix scan) groups the
//!     points by (window, bucket),
//!  2. buckets are cut into work items of at most `chunk` points, one thread each, so a huge
//!     bucket (scalars that repeat, or the sparse top window) cannot become a serial tail,
//!  3. each window's weighted bucket sum comes from many small running sums plus a tree sum,
//!  4. windows are combined with Horner's rule.

use crate::{field, gpu::Gpu, group::Group, host};

const REDUCE_CHUNK: u32 = 16; // buckets per running-sum thread
const SUM_GROUP: u32 = 16; // items per tree-sum thread

// Operating systems stop GPU work that runs for too long without letting the screen update (macOS
// reports "Impacting Interactivity" and the browser then loses the device). The two heavy stages
// are therefore cut into slices of whole windows and submitted one slice at a time. A slice is
// sized to roughly 100 ms on an Apple M1: this many point additions (G1; G2 costs about 5x more)
// in the accumulate stage and this many buckets in the bucket reduction.
const ACCUMULATE_BUDGET: u64 = 1 << 20;
const REDUCE_BUDGET: u64 = 1 << 17;

pub struct Msm {
    group: Group,
    n: u32,
    windows: u32,
    buckets: u32, // per window
    items_per_window: u32,
    pass: Passes,
    buf: Buffers,
    sum_levels: Vec<(u32, wgpu::Buffer)>, // (input length, params) per tree-sum level
    merge_rounds: Vec<wgpu::Buffer>,      // params (the stride) of each merge round
    accumulate_slices: Vec<Slice>,
    chunk_slices: Vec<Slice>,
}

struct Passes {
    count: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    accumulate: wgpu::ComputePipeline,
    merge: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
    chunks: wgpu::ComputePipeline,
    sum: wgpu::ComputePipeline,
    combine: wgpu::ComputePipeline,
}

struct Buffers {
    counts: wgpu::Buffer,
    starts: wgpu::Buffer,
    item_starts: wgpu::Buffer,
    item_totals: wgpu::Buffer,
    sorted: wgpu::Buffer,
    item_results: wgpu::Buffer,
    item_info: wgpu::Buffer, // per item: [position in its bucket, items in its bucket]
    buckets: wgpu::Buffer,
    dims: wgpu::Buffer,
    sizes: wgpu::Buffer, // [n, items per window, points per item]: run-time values, so shaders are shared across sizes
    sums: [wgpu::Buffer; 2],
    result: wgpu::Buffer,
}

/// One dispatch of a stage that is cut into pieces by window, so that no single GPU command
/// buffer runs long (see `SLICE_BUDGET`): the first window or thread it covers, and its grid.
struct Slice {
    first: wgpu::Buffer,
    grid: [u32; 3],
}

type Stage<'a> = (&'static str, &'a wgpu::ComputePipeline, Vec<&'a wgpu::Buffer>, [u32; 3]);

impl Msm {
    /// Window size that balances bucket-accumulation work against bucket-reduction work
    /// (tuned on an Apple M1 for G1 and G2, n from 2^16 to 2^20).
    pub fn default_window(n: usize) -> u32 {
        (n.max(1).ilog2().saturating_sub(5)).clamp(11, 16)
    }

    /// Plan for G1 MSMs of `n` points with window size `c` bits (9 <= c <= 20).
    ///
    /// Work items hold about a quarter of the average bucket: small enough that the lanes of
    /// a SIMD group finish together, large enough that combining items stays cheap.
    pub fn new(gpu: &Gpu, n: usize, c: u32) -> Self {
        Self::for_group(gpu, Group::G1, n, c, None)
    }

    /// Like `new`, with an explicit maximum number of points per work item.
    pub fn with_item_points(gpu: &Gpu, n: usize, c: u32, item_points: u32) -> Self {
        Self::for_group(gpu, Group::G1, n, c, Some(item_points))
    }

    /// Plan for MSMs in `group`; `item_points` = None picks the work item size automatically.
    pub fn for_group(gpu: &Gpu, group: Group, n: usize, c: u32, item_points: Option<u32>) -> Self {
        assert!(n >= 1, "an MSM needs at least one point");
        assert!((9..=20).contains(&c), "window size c must be in 9..=20, got {c}");
        let mean_bucket = (n as u32 >> (c - 1)).max(1);
        let item_points = item_points.unwrap_or_else(|| (mean_bucket / 4).next_power_of_two().clamp(8, 1024));
        let windows = 255u32.div_ceil(c);
        let b = 1u32 << (c - 1);
        let t = b / REDUCE_CHUNK;
        let ipw = b + (n as u32).div_ceil(item_points) + 1;
        let header = format!(
            "const C: u32 = {c}u; const W: u32 = {windows}u; const B: u32 = {b}u; \
             const L: u32 = {REDUCE_CHUNK}u; const T: u32 = {t}u; const G: u32 = {SUM_GROUP}u;\n"
        );
        let math = group.wgsl();
        let plain = |src: &str, extra: &str| gpu.pipeline(&format!("{header}{extra}{src}"), "main");
        let curve = |src: &str| gpu.pipeline(&format!("{header}{math}\n{src}"), "main");

        let digits = include_str!("shaders/msm_digits.wgsl");
        let pass = Passes {
            count: plain(digits, "const SCATTER: bool = false;\n"),
            scatter: plain(digits, "const SCATTER: bool = true;\n"),
            scan: plain(include_str!("shaders/msm_scan.wgsl"), ""),
            accumulate: curve(include_str!("shaders/msm_accumulate.wgsl")),
            merge: curve(include_str!("shaders/msm_merge.wgsl")),
            finalize: curve(include_str!("shaders/msm_finalize.wgsl")),
            chunks: curve(include_str!("shaders/msm_reduce_chunks.wgsl")),
            sum: curve(include_str!("shaders/msm_sum.wgsl")),
            combine: curve(include_str!("shaders/msm_combine.wgsl")),
        };

        let (w, bk, ip, xw) = (windows as usize, b as usize, ipw as usize, group.xyzz_words());
        let buf = Buffers {
            counts: gpu.empty_buffer(w * bk),
            starts: gpu.empty_buffer(w * bk),
            item_starts: gpu.empty_buffer(w * bk),
            item_totals: gpu.empty_buffer(w),
            sorted: gpu.empty_buffer(w * n),
            item_results: gpu.empty_buffer(w * ip * xw),
            item_info: gpu.empty_buffer(w * ip * 2),
            buckets: gpu.empty_buffer(w * bk * xw),
            dims: gpu.buffer(&[windows * b]),
            sizes: gpu.buffer(&[n as u32, ipw, item_points]),
            sums: [gpu.empty_buffer(w * t as usize * xw), gpu.empty_buffer(w * t as usize * xw)],
            result: gpu.empty_buffer(xw),
        };
        // Merge rounds with strides 1, G, G^2, ... until G^rounds covers the largest possible item count.
        let mut merge_rounds = Vec::new();
        let (mut stride, max_items) = (1u32, (n as u32).div_ceil(item_points));
        while stride < max_items {
            merge_rounds.push(gpu.buffer(&[stride]));
            stride = stride.saturating_mul(SUM_GROUP);
        }
        let mut sum_levels = Vec::new();
        let mut len = t;
        while len > 1 {
            sum_levels.push((len, gpu.buffer(&[len])));
            len = len.div_ceil(SUM_GROUP);
        }
        let cost = if group == Group::G2 { 5 } else { 1 };
        let per_slice = |budget: u64, per_window: u64| ((budget / cost / per_window.max(1)).max(1) as u32).min(windows);
        let mut accumulate_slices = Vec::new();
        let mut chunk_slices = Vec::new();
        let (acc_windows, chunk_windows) = (per_slice(ACCUMULATE_BUDGET, n as u64), per_slice(REDUCE_BUDGET, b as u64));
        for first in (0..windows).step_by(acc_windows as usize) {
            let count = acc_windows.min(windows - first);
            accumulate_slices.push(Slice { first: gpu.buffer(&[first]), grid: [ipw.div_ceil(64), count, 1] });
        }
        for first in (0..windows).step_by(chunk_windows as usize) {
            let count = chunk_windows.min(windows - first);
            chunk_slices.push(Slice { first: gpu.buffer(&[first * t]), grid: [(count * t).div_ceil(64), 1, 1] });
        }
        Self {
            group,
            n: n as u32,
            windows,
            buckets: b,
            items_per_window: ipw,
            pass,
            buf,
            sum_levels,
            merge_rounds,
            accumulate_slices,
            chunk_slices,
        }
    }

    /// Upload affine points given as plain coordinates (x then y, 8 u32 words each; the
    /// point at infinity is (0, 0)) and convert them to the GPU's Montgomery form.
    pub fn upload_bases(gpu: &Gpu, plain_xy: &[u32]) -> wgpu::Buffer {
        let m = field::FQ;
        let r = host::montgomery_r(&m);
        Self::upload_scaled(gpu, plain_xy, &host::mul_mod(&r, &r, &m))
    }

    /// Like `upload_bases`, but the coordinates are already in Montgomery form with R = 2^256
    /// (the form ffjavascript and snarkjs keep points in), so they can be sent as they are.
    pub fn upload_bases_montgomery_256(gpu: &Gpu, xy: &[u32]) -> wgpu::Buffer {
        let m = field::FQ;
        // The kernel computes v * scale / R. With v = a * 2^256 and R = 2^260 we want a * R,
        // so scale = 2^264 = R * 16.
        Self::upload_scaled(gpu, xy, &host::mul_mod(&host::montgomery_r(&m), &[16, 0, 0, 0], &m))
    }

    /// Replace every coordinate v by v * scale / R (mod p) on the GPU.
    fn upload_scaled(gpu: &Gpu, xy: &[u32], scale: &host::U256) -> wgpu::Buffer {
        let buffer = gpu.buffer(xy);
        let coords = (xy.len() / 8) as u32;
        let params = gpu.buffer(&[&[coords][..], &host::words32(scale)[..]].concat());
        let source = format!("{}\n{}", field::wgsl("fq", field::FQ), include_str!("shaders/to_montgomery.wgsl"));
        let pipeline = gpu.pipeline(&source, "main");
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        gpu.encode_grid(&mut encoder, &pipeline, &[&buffer, &params], Gpu::grid_1d(coords.div_ceil(64)));
        gpu.queue.submit([encoder.finish()]);
        buffer
    }

    fn stages<'a>(&'a self, bases: &'a wgpu::Buffer, scalars: &'a wgpu::Buffer) -> Vec<Stage<'a>> {
        let (p, b) = (&self.pass, &self.buf);
        let groups = |threads: u32| [threads.div_ceil(64), 1, 1];
        let point_groups = Gpu::grid_1d(self.n.div_ceil(64));
        let slots = self.windows * self.buckets;
        let mut stages: Vec<Stage> = vec![
            ("count", &p.count, vec![scalars, &b.counts, &b.starts, &b.sorted, &b.sizes], point_groups),
            (
                "scan",
                &p.scan,
                vec![&b.counts, &b.starts, &b.item_starts, &b.item_totals, &b.sizes],
                [self.windows, 1, 1],
            ),
            ("scatter", &p.scatter, vec![scalars, &b.counts, &b.starts, &b.sorted, &b.sizes], point_groups),
        ];
        for slice in &self.accumulate_slices {
            let buffers = vec![
                bases,
                &b.sorted,
                &b.starts,
                &b.counts,
                &b.item_starts,
                &b.item_totals,
                &b.item_results,
                &b.sizes,
                &b.item_info,
                &slice.first,
            ];
            stages.push(("accumulate", &p.accumulate, buffers, slice.grid));
        }
        const MERGE_NAMES: [&str; 6] = ["merge 1", "merge 2", "merge 3", "merge 4", "merge 5", "merge 6"];
        for (round, params) in self.merge_rounds.iter().enumerate() {
            let buffers = vec![&b.item_results, &b.item_info, &b.item_totals, &b.sizes, params];
            stages.push((
                MERGE_NAMES[round.min(5)],
                &p.merge,
                buffers,
                [self.items_per_window.div_ceil(64), self.windows, 1],
            ));
        }
        let finalize = vec![&b.item_results, &b.counts, &b.item_starts, &b.buckets, &b.dims, &b.sizes];
        stages.push(("finalize", &p.finalize, finalize, groups(slots)));
        for slice in &self.chunk_slices {
            stages.push(("reduce chunks", &p.chunks, vec![&b.buckets, &b.sums[0], &slice.first], slice.grid));
        }
        for (i, (len, params)) in self.sum_levels.iter().enumerate() {
            let (src, dst) = (&b.sums[i % 2], &b.sums[(i + 1) % 2]);
            let threads = self.windows * len.div_ceil(SUM_GROUP);
            stages.push(("tree sum", &p.sum, vec![src, dst, params], groups(threads)));
        }
        let window_sums = &b.sums[self.sum_levels.len() % 2];
        stages.push(("combine", &p.combine, vec![window_sums, &b.result], [1, 1, 1]));
        stages
    }

    pub fn group(&self) -> Group {
        self.group
    }

    /// Panics with a clear message when a buffer is too small for this plan. Without the check the
    /// GPU reads zeros past the end of a buffer and silently returns a wrong answer.
    fn check_buffers(&self, bases: &wgpu::Buffer, scalars: &wgpu::Buffer) {
        let (n, point_bytes) = (self.n as u64, 4 * self.group.affine_words() as u64);
        assert!(scalars.size() >= n * 32, "scalars buffer holds {} bytes; {n} scalars need {}", scalars.size(), n * 32);
        assert!(
            bases.size() >= n * point_bytes,
            "bases buffer holds {} bytes; {n} points need {}",
            bases.size(),
            n * point_bytes
        );
    }

    /// Compute `sum(scalars[i] * bases[i])`. `scalars` holds 8 plain u32 words per scalar.
    /// Returns the result buffer: one XYZZ point (`group().xyzz_words()` words, Montgomery form).
    pub fn run(&self, gpu: &Gpu, bases: &wgpu::Buffer, scalars: &wgpu::Buffer) -> &wgpu::Buffer {
        self.check_buffers(bases, scalars);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.buf.counts, 0, None);
        gpu.queue.submit([encoder.finish()]);
        // One command buffer per stage (and per slice of the heavy stages), so that none runs long.
        for (_, pipeline, buffers, grid) in self.stages(bases, scalars) {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            gpu.encode_grid(&mut encoder, pipeline, &buffers, grid);
            gpu.queue.submit([encoder.finish()]);
        }
        &self.buf.result
    }

    /// Like `run`, but waits for the GPU to finish each heavy slice before queuing the next. A
    /// burst of back-to-back submissions keeps the GPU busy for a second or more without a gap;
    /// on macOS that gets the whole device killed ("Impacting Interactivity") when the screen needs
    /// it. The short pauses let the system schedule its own work in between. Use this in programs
    /// that share the GPU with a display, such as browsers; `run` is fastest otherwise.
    pub async fn run_yielding(&self, gpu: &Gpu, bases: &wgpu::Buffer, scalars: &wgpu::Buffer) -> wgpu::Buffer {
        self.check_buffers(bases, scalars);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.buf.counts, 0, None);
        gpu.queue.submit([encoder.finish()]);
        for (name, pipeline, buffers, grid) in self.stages(bases, scalars) {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            gpu.encode_grid(&mut encoder, pipeline, &buffers, grid);
            gpu.queue.submit([encoder.finish()]);
            if matches!(name, "accumulate" | "reduce chunks" | "finalize" | "combine") {
                gpu.finished_async().await;
            }
        }
        self.buf.result.clone()
    }

    /// Run every stage as its own submission; returns (stage, milliseconds). For profiling.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn profile(&self, gpu: &Gpu, bases: &wgpu::Buffer, scalars: &wgpu::Buffer) -> Vec<(&'static str, f64)> {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.buf.counts, 0, None);
        gpu.queue.submit([encoder.finish()]);
        let mut times: Vec<(&'static str, f64)> = Vec::new();
        for (name, pipeline, buffers, grid) in self.stages(bases, scalars) {
            gpu.wait();
            let t = std::time::Instant::now();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            gpu.encode_grid(&mut encoder, pipeline, &buffers, grid);
            gpu.queue.submit([encoder.finish()]);
            gpu.wait();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            match times.iter_mut().find(|(n, _)| *n == name) {
                Some(entry) => entry.1 += ms,
                None => times.push((name, ms)),
            }
        }
        times
    }
}

/// Convert the GPU's XYZZ result (32 words) to plain affine coordinates; `None` = infinity.
pub fn xyzz_to_affine(words: &[u32]) -> Option<([u64; 4], [u64; 4])> {
    let m = &field::FQ;
    let coord = |i: usize| host::from_montgomery(&host::from_words32(&words[i * 8..i * 8 + 8]), m);
    let (x, y, zz, zzz) = (coord(0), coord(1), coord(2), coord(3));
    if zz == [0; 4] {
        return None;
    }
    Some((host::mul_mod(&x, &host::inverse(&zz, m), m), host::mul_mod(&y, &host::inverse(&zzz, m), m)))
}

/// An Fq2 element a + b*u as two plain 256-bit numbers [a, b].
pub type Fq2Words = [host::U256; 2];

/// Convert the GPU's G2 XYZZ result (64 words) to plain affine coordinates
/// `((x.a, x.b), (y.a, y.b))` with Fq2 elements written a + b*u; `None` = infinity.
pub fn xyzz_to_affine_g2(words: &[u32]) -> Option<(Fq2Words, Fq2Words)> {
    let m = &field::FQ;
    let fq2 = |i: usize| -> Fq2Words {
        std::array::from_fn(|k| {
            host::from_montgomery(&host::from_words32(&words[(2 * i + k) * 8..(2 * i + k + 1) * 8]), m)
        })
    };
    let (x, y, zz, zzz) = (fq2(0), fq2(1), fq2(2), fq2(3));
    if zz == [[0; 4]; 2] {
        return None;
    }
    Some((fq2_mul(&x, &fq2_inverse(&zz, m), m), fq2_mul(&y, &fq2_inverse(&zzz, m), m)))
}

// Fq2 = Fq[u] / (u^2 + 1): (a + bu)(c + du) = (ac - bd) + (ad + bc)u.
fn fq2_mul(p: &Fq2Words, q: &Fq2Words, m: &host::U256) -> Fq2Words {
    let (ac, bd) = (host::mul_mod(&p[0], &q[0], m), host::mul_mod(&p[1], &q[1], m));
    let (ad, bc) = (host::mul_mod(&p[0], &q[1], m), host::mul_mod(&p[1], &q[0], m));
    [host::sub_mod(&ac, &bd, m), host::add_mod(&ad, &bc, m)]
}

// 1 / (a + bu) = (a - bu) / (a^2 + b^2)
fn fq2_inverse(p: &Fq2Words, m: &host::U256) -> Fq2Words {
    let norm = host::add_mod(&host::mul_mod(&p[0], &p[0], m), &host::mul_mod(&p[1], &p[1], m), m);
    let inv = host::inverse(&norm, m);
    [host::mul_mod(&p[0], &inv, m), host::sub_mod(&[0; 4], &host::mul_mod(&p[1], &inv, m), m)]
}
