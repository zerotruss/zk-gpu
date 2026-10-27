# Contributing to zk-gpu

Issues are welcome. This project is maintained on a best-effort basis, so please open an issue
before sending a pull request for a large change.

## Rules

1. **Keep the code simple and readable.** Match the naming and comment density of the code around
   it. Plain code beats clever code. Write the full form of an abbreviation the first time it
   appears in a document or comment (for example Multi-Scalar Multiplication, MSM).
2. **Test after every change, and show the tests can fail.** A new test should fail without your
   fix. Shader changes must also pass the mutation checker (below).
3. **Stay portable.** Nothing may be tuned to, or only work on, one GPU vendor or operating
   system. WebGPU's guaranteed limits are the budget, for example 16 KB of workgroup memory,
   65,535 workgroups per dimension, and no 64-bit integers in WGSL (the reason for 13-bit limbs).
   Numbers measured on one machine must say so.
4. **Report speed honestly.** Name the machine, the sizes, warm or cold, and the baseline you
   compared with (arkworks on all cores, a single-thread WebAssembly build, snarkjs...). The same
   GPU looks several times better against a weak baseline. Quote warm numbers (the first run
   includes shader compilation and worker start-up) and say when an earlier claim turns out wrong.
5. **Correctness first.** A faster kernel that is wrong for any valid input is a bug. Inputs must
   be canonical (scalars `< r`, coordinates `< p`).
6. **Keep the library independent.** `zk-gpu` must not depend on Node, Circom or snarkjs. Front
   ends live in separate projects.

## Setup

```sh
export PATH="$HOME/.cargo/bin:$PATH"      # a Homebrew rustc has no wasm target; rustup's does
rustup target add wasm32-unknown-unknown   # for the browser tests
./check.sh quick
```

The GPU tests need a real GPU; hosted CI runners have none, so maintainers run the checks below
locally before a release. For the browser tests, `CHROMEDRIVER=/path ./check.sh browser` uses a
chromedriver that matches your Chrome (a version mismatch makes the driver die on launch).

## What to run when you change something

| You changed | Run |
|---|---|
| any shader, `msm.rs`, `ntt.rs`, `field.rs` or `group.rs` | `./check.sh quick` (about 5 min), then `python3 tools/mutation_check.py` (about 20 min) if you touched shader text |
| performance-sensitive code | `./check.sh quick` (it compares with the saved baseline for your device); for a deliberate speed change re-record with `./check.sh baseline` |
| the browser path | `./check.sh browser` |
| before a release | `./check.sh full` (about 20 min, every size plus the wasm build) |
| the golden test vectors | `./check.sh golden` (regenerates them from ffjavascript; needs node and npm) |

Tests share one GPU, so they run with `--test-threads=1` (in parallel they are 6x slower and
timing-dependent). Useful variables: `ZK_GPU_QUICK=1` (smaller sizes), `ZK_GPU_FUZZ_SEED=<n>`
(replay one fuzz case), `ZK_GPU_FUZZ_ITERS=<n>`.

## How the checks work

- **Reference answers.** Results are compared with arkworks and with vectors produced by
  ffjavascript (`tests/golden/`), so a mistake shared with one reference is still caught. There are
  also property tests that need no reference (linearity, round trips, convolution theorem).
- **Mutation checker.** `tools/mutation_check.py` applies one deliberate bug at a time and requires
  the suite to fail. It matches exact source text, so after editing shaders run it and fix any
  mutant whose target text no longer occurs exactly once; otherwise it silently tests nothing.
  `--perf` applies slowdowns that keep results correct and checks that the benchmark comparison
  flags them.
- **Benchmarks.** One call is timed at a time, after a warm-up; the statistic is the fastest run,
  best of two. Baselines are per device (`bench/baselines/<device>.json`) and are **not stored in the
  repository**: run `./check.sh baseline` once on your machine (about 15 min) before `./check.sh quick`
  can compare, and before `tools/mutation_check.py --perf`. A change counts as a
  regression only above `max(10%, 1.5x the baseline's own spread, 1.5 ms)`, and the CPU side is
  reported but never judged. Run on AC power and close other GPU-heavy apps. Timings of kernels
  under about 5 ms can vary by 25% between identical runs. This also means a regression smaller than
  about 10% goes undetected.

## Lessons that shaped the code

- **Disable workgroup zero-initialisation** when creating pipelines (30x slower NTT on Metal
  otherwise). Kernels here write every slot before reading it; keep that true.
- **No long serial work in one thread.** One GPU thread is slow, and a serial bucket tail made some
  window sizes 3x slower. Split work into small items and merge in parallel rounds.
- **SIMD lanes wait for the slowest lane,** so uneven work is expensive.
- **Never hold the GPU in one long dispatch.** On macOS the watchdog can kill the device
  ("Impacting Interactivity"). Heavy stages are sliced; browser callers should use `run_yielding`,
  which awaits each heavy slice.
- **Never panic in code reached from async WebAssembly.** The JavaScript promise then hangs for
  ever. Return errors (`Gpu::try_read_async`).
- **Plans are not re-entrant.** Two overlapping runs of one MSM plan overwrite each other's
  buffers; use one plan per caller.
- **Large dispatches need 2D grids** (65,535 workgroups per dimension).
- **Pipelines are cached by shader source;** keep sizes (point count, items, window) as run-time
  values so a new size does not recompile eight shaders.

## Pull requests

- One topic per pull request, with tests.
- Say what changed and why, and paste the summary lines of `./check.sh`.
- Any speed claim names its machine and baseline.

## Licence

By contributing you agree that your work is licensed like the project, `MIT OR Apache-2.0`.
