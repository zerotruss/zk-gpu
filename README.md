# zk-gpu

Portable GPU **MSM** and **NTT** for BN254, the two operations that dominate Groth16/PLONK
proving. One Rust codebase and one set of WGSL kernels targets native GPUs (Metal, Vulkan, DX12)
and the browser (WebGPU) through `wgpu`. No CUDA, no vendor extensions.

**Tested so far:** an Apple M1 (Metal) and Chrome (WebGPU) on the same machine. The library compiles for
Linux, Windows and WebAssembly (checked in CI and from a Mac), but it has **not yet been run** on NVIDIA,
AMD or Intel GPUs, so treat portability as designed, not proven. Reports from other hardware are welcome:
run `./check.sh full` and open an issue with the output.

- **MSM** in G1 and G2: signed-digit Pippenger, fully on the GPU (sorting, bucket sums, reduction).
  One set of shaders serves both groups.
- **NTT** over Fr: forward and inverse, in place, natural order in and out.
- Verified against [arkworks](https://github.com/arkworks-rs/algebra) natively *and* inside headless Chrome.

## Results

Measured on an **Apple M1** (8-core GPU, the weakest hardware this is likely to run on), against
multi-threaded arkworks on the same machine (8 threads), with the benchmark runner described under
[Testing and benchmarking](#testing-and-benchmarking). Each figure is one call, timed from
submission until the result is ready (the time a prover that needs the answer before it can
continue actually waits), after a warm-up. MSM time includes uploading the scalars and reading the
result back. Speedup = CPU time / GPU time, from medians.

**Native** (re-measured on 2026-10-24 with `./check.sh quick` / `full` after the window re-tune):

| size | NTT | MSM |
|---|---|---|
| 2^14 | 0.5x (CPU wins) | 0.59x (CPU wins) |
| 2^16 | 0.8–1.8x (noisy) | 0.94x |
| 2^18 | 1.55–1.7x | 1.20x |
| 2^20 | 1.7x | 1.6x |
| 2^22 | 1.7-2.2x | not in the default benchmark (1.7x with an earlier method) |

For reference, 2^20: NTT 44 ms vs 74–77 ms, MSM 0.85 s vs 1.37–1.39 s.

The GPU wins from about 2^18 upward and loses below 2^16, where a fixed cost of roughly 1–2 ms per
call (submission, synchronisation) dominates. Earlier versions of this README quoted higher
small-size speedups (for example 1.8–2.2x NTT at 2^16); those were measured by queueing several calls
back to back, which hides that fixed cost. Queued throughput is real (a prover can queue
independent work), but a single call is the honest default. Short kernels are also
noisy on this laptop: the 2^14 NTT measured 1.3 ms to 3.1 ms at different times, on battery and on
AC power alike, so the small-size rows can shift by a few tenths of a speedup between runs.

**Browser** (headless Chrome, WebGPU, same M1; the baseline is single-threaded WebAssembly arkworks,
which is what a browser prover without a GPU path has to use). Re-measured on 2026-10-26 with
`./check.sh browser bench`, two runs, one call at a time after a warm-up. The GPU times were steady
between runs; the CPU time is a single unwarmed call and varied (for example 4.2 s to 7.8 s for the
2^18 MSM), so the speed-ups are rough:

| size | NTT: GPU ms (speed-up) | MSM: GPU ms (speed-up) |
|---|---|---|
| 2^14 | 2.0–2.2 ms (6–7x) | 41–46 ms (8–10x) |
| 2^16 | 4.1–4.5 ms (10–11x) | 86–87 ms (14–15x) |
| 2^18 | 10 ms (20x) | 257–283 ms (17–28x) |
| 2^20 | 43 ms (21x) | not measured |

The browser gain is large mostly because the baseline is weak (one thread, no SIMD): the GPU time is
about the same as native (about 260 ms for the 2^18 MSM in both). Against a multi-threaded browser
prover the gain is much smaller: for whole Groth16 proofs against snarkjs's own web workers it is
2.6–3.6x (see the separate `snarkjs-gpu` project).

**Which baseline matters.** Against 8-thread native arkworks the MSM is about 1.2x faster at 2^18
and 1.6x at 2^20, and slower below 2^16. A native prover built for speed on the CPU (for example
rapidsnark) can prove the same circuit faster than snarkjs plus this library on the same machine.
Use the CPU below roughly 2^16 points.

On GPUs with more integer throughput than an M1 the gap should widen; that is untested here. Only
one machine (an Apple M1) has been measured so far.

## Usage

A complete, runnable example (one MSM and one NTT, each checked against arkworks) is
`examples/quickstart.rs`: `cargo run --release --example quickstart`.

Field elements and curve coordinates are **8 little-endian `u32` words** (plain integers, fully
reduced). Scalars must be `< r`. Affine points are `x` then `y`; the point at infinity is `(0, 0)`.

```rust
use zk_gpu::{gpu::Gpu, msm::{Msm, xyzz_to_affine}, ntt::Ntt};

let gpu = Gpu::new();                       // browsers: Gpu::new_async().await

// NTT, in place, size 2^log_n (10..=27)
let ntt = Ntt::new(&gpu, log_n);
let data = gpu.buffer(&words);              // 8 u32 per element
ntt.forward(&gpu, &data);
ntt.inverse(&gpu, &data);                   // includes the 1/n factor

// MSM
let bases = Msm::upload_bases(&gpu, &plain_xy_words);   // convert to GPU form once, reuse
let scalars = gpu.buffer(&scalar_words);
let msm = Msm::new(&gpu, n, Msm::default_window(n));
let out = msm.run(&gpu, &bases, &scalars);
let affine = xyzz_to_affine(&gpu.read(out, 32));        // browsers: gpu.read_async(..).await
```

Bases are meant to be uploaded once (a proving key does not change) and reused across proofs.

**G2.** `Msm::for_group(&gpu, Group::G2, n, c, None)` builds a plan for points over Fq2. A G2 point is
32 words (`x.a, x.b, y.a, y.b`, each 8 words, Fq2 element = a + b*u with u^2 = -1); the result is
64 words, read with `xyzz_to_affine_g2`. **Bases already in Montgomery form with R = 2^256**
(the form snarkjs/ffjavascript keep in a `.zkey`) go through `Msm::upload_bases_montgomery_256`
instead of `upload_bases`.

Plans can be created for any size cheaply: the compiled shaders depend only on the window size and
the group, and are cached in `Gpu`, so a new `n` costs buffer allocation, not shader compilation.

## How it works

**Field arithmetic** (`src/field.rs`). WGSL has no 32x32→64 multiply, so a 254-bit number is 20
limbs of 13 bits, one per `u32`. A limb product is under 2^26, so about 40 can be summed without
carries and Montgomery multiplication needs only multiply-adds until one final carry pass. The
WGSL is *generated* (fully unrolled, scalar variables) because array accumulators make GPU
compilers spill to memory. The same generator produces Fq (curve) and Fr (NTT).

**NTT** (`src/ntt.rs`, `src/shaders/ntt.wgsl`). Radix-2 decimation in frequency. A pass fuses up
to 9 stages inside 16 KB of workgroup memory (the WebGPU baseline limit), and a transform takes an
even number of passes, ping-ponging with a scratch buffer so the last pass (which writes in
bit-reversed order) never races with readers. Twiddles are stored in Montgomery form, so the data
itself never needs conversion.

**MSM** (`src/msm.rs`, `src/shaders/msm_*.wgsl`). Stages:
1. `count` / `scan` / `scatter`: signed c-bit digits per scalar, counting sort of points into
   (window, bucket) lists with atomics and a prefix scan. No CPU preprocessing.
2. `accumulate`: points are added in XYZZ coordinates (8M+2S per mixed addition). Work is split
   into items of a fraction of the average bucket size, one thread each.
3. `finalize`: items of one bucket are summed.
4. `reduce chunks` / `tree sum`: each window's weighted bucket sum from many short running sums
   plus a tree sum.
5. `combine`: Horner's rule over windows.

Buckets with many work items (a nearly empty top window, many equal scalars) are reduced by a few
`merge` rounds before `finalize`, so a huge bucket is never one long serial sum in a single thread.
The two heavy stages are cut into slices of windows, each submitted separately; `run_yielding` also waits
for each slice, so a browser or display never sees the GPU busy for a second without a gap (macOS kills
such work: "Impacting Interactivity").

The shaders are written once over a field element type `Fe` (`src/group.rs`): `Fe = Fq` for G1 and
`Fe = Fq2` for G2 (Karatsuba multiplication, 3 Fq multiplications). The point count and work item
size are read from a small buffer at run time, not compiled into the shader.

## Lessons that shaped the design

These cost real time to find; they are worth knowing if you write GPU code with wgpu.

1. **Workgroup zero-initialization can cost 30x.** WebGPU requires `var<workgroup>` to start
   zeroed. On Metal, wgpu has one thread zero the whole 16 KB tile, serially, per workgroup. Our
   kernels write every slot before reading, so `Gpu::pipeline` disables it
   (`zero_initialize_workgroup_memory: false`). That single flag took the NTT from 650 ms to 44 ms
   at 2^20. Browsers do not expose the flag and zero cooperatively.
2. **One GPU thread is slow; throughput comes from many threads.** A single point addition has roughly
   65 µs of latency in one thread (estimated from the measurements). Any kernel with a long per-thread chain stalls the GPU, so the
   design avoids long chains everywhere. The first MSM took 17 s at 2^20 because the sparse top
   window put a quarter of all points into three buckets, which one thread each added serially.
   Work items fix that; real circuits with repeated scalars (many 0s and 1s) hit the same
   problem.
3. **SIMD lockstep makes uneven work expensive.** Lanes of a group wait for the slowest. Work
   items sized near a quarter of the average bucket cut the accumulate stage by 25–40%.
4. **The kernels are instruction-bound.** Removing the final modular reduction or making
   add/sub free (timing-only experiments) each saved only 7–9%. fp32 limbs were not faster than
   u32 on the M1 (383 vs 290 G ops/s). The remaining cost is the ~800 multiply-adds inside each
   field multiplication, so further gains need fewer field multiplications, not tuning.

## Testing and benchmarking

One entry point, `./check.sh`:

```sh
./check.sh quick      # lint, tests, benchmark (2 runs), compare with the baseline
./check.sh full       # the same with every size, plus the wasm build
./check.sh browser    # correctness in headless Chrome over WebGPU (add "bench" for browser timings)
./check.sh baseline   # benchmark 3 times and save the best as this device's baseline
./check.sh golden     # regenerate the golden vectors (needs node)
```

Rust needs `~/.cargo/bin` first in `PATH` (the script does this); browser mode needs `wasm-pack`
and a chromedriver matching your Chrome (wasm-pack downloads one; set `CHROMEDRIVER=/path/to/chromedriver` if its version does not match).

**What is checked**

| layer | file | what it guarantees |
|---|---|---|
| field and curve | `tests/curve.rs` | GPU point add/double equal arkworks, including P+P, P+(-P) and infinity |
| reference | `tests/ntt.rs`, `tests/msm.rs` | results equal arkworks across sizes, window sizes and work-item sizes; MSM edge cases |
| independent reference | `tests/golden.rs`, `tests/golden/` | results equal vectors made by **ffjavascript** (the math library behind snarkjs), so a mistake shared with arkworks cannot hide |
| properties | `tests/properties.rs` | no reference needed: NTT of a delta, linearity, convolution theorem, round trips; MSM linearity in scalars and bases, scaling, order independence, n = 1, window-boundary scalars, plan reuse, and a seeded random fuzz (`ZK_GPU_FUZZ_SEED=<n>` replays one failing case) |
| browser | `tests/web.rs` | the NTT/MSM against arkworks and the golden vectors on real WebGPU |
| performance | `examples/bench.rs`, `examples/compare.rs` | JSON timings; regression check against a per-device baseline |

Regenerate the golden vectors with `./check.sh golden`; they are deterministic (same seed, same files).

**The benchmark runner** (`examples/bench.rs`; its header lists the options) first checks every size against arkworks and refuses to time a
wrong answer, then warms the GPU up, then times the CPU and the GPU in separate blocks. It writes
`bench/results/*.json` with the device, power source, thread count, source hash and, for the MSM,
per-stage timings.

**The comparison** (`examples/compare.rs`) uses the *fastest* run as its statistic, because
interference only ever adds time, and takes the best of several runs. A change counts only if it
exceeds 10%, 1.5x the baseline's own noise, and 1.5 ms (short kernels moved by up to 1.2 ms between
identical runs; in 8 consecutive clean comparisons a 1 ms floor raised 3 false alarms and 1.5 ms none).
So a regression smaller than about 10% (15% for the 10 ms NTT) is not detected. The CPU side is reported but never judged: in five
identical runs the GPU timings varied by 3–6% (MSM) while one CPU run was up to 8x slower from
other load. Results from different devices are refused (exit code 2).

**Checking the checks.** `python3 tools/mutation_check.py` injects 11 deliberate bugs one at a time
(dropped carry, missing doubling case, wrong scaling, a flipped reduction...) and runs the
whole suite against each. All 11 were caught, each by between 4 and 22 independent tests.
`--perf` injects two slowdowns that keep results correct; both are flagged as regressions.

Older single-purpose examples remain for exploration: `msm_profile` (per-stage MSM timings,
`LOG=20`), `field_bench` (raw modular multiplication throughput).

## Limitations and next steps

- BN254 only (G1 and G2 MSM, Fr NTT).
- Inputs must be canonical: scalars `< r`, coordinates `< p`, NTT inputs `< r`; values out of range
  give a wrong result without an error.
- Wrong sizes panic with a clear message: a scalar, base or NTT buffer smaller than the plan needs,
  `n = 0`, a window outside 9..=20, an NTT size outside 2^10..=2^27 (buffers may be larger than needed).
- Not constant-time. Do not use for secret-dependent scalars where timing matters.
- Ideas not done: batch-affine bucket additions (the one change that cuts field multiplications,
  est. up to 1.4x), GLV, dedicated squaring (~3%), lazy reduction (~7–9% each). The Circom/snarkjs
  front end lives in the separate project `snarkjs-gpu`.
- The field-agnostic shaders and run-time sizes made the G1 MSM 3-7% slower than the saved baseline
  (inside the 10% alarm threshold); the generic packing loops are the likely cause. The default window
  is `log2(n) - 5` clamped to 11..=16 (re-tuned for G1 and G2 once the serial-bucket tail was removed).
- The NTT is only the transform; a Groth16 prover also needs coset shifts and pointwise
  products, which are simple kernels on top of this.

## Contributing and security

Issues are welcome. This project is maintained on a best-effort basis; see [CONTRIBUTING.md](CONTRIBUTING.md) and open an issue
before sending a pull request. For security problems read [SECURITY.md](SECURITY.md) first.

## Licence

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT licence](LICENSE-MIT),
at your option. Unless you state otherwise, any contribution you submit for inclusion is dual
licensed as above, without additional terms.
