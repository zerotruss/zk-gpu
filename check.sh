#!/bin/sh
# One entry point for correctness and performance checks.
#
#   ./check.sh quick      lint, tests (smaller sizes), benchmark, compare with the baseline
#   ./check.sh full       the same with every size, plus the wasm build
#   ./check.sh browser    correctness (and optionally timings) in headless Chrome via WebGPU
#   ./check.sh baseline   benchmark every size 3 times and save the best as this device's baseline
#   ./check.sh golden     regenerate the golden test vectors with ffjavascript (needs node + npm)
#
# Timings: close other GPU-heavy apps first. Short kernels (under ~5 ms) varied by up to ~25%
# between identical runs on the M1, which the comparison allows for; correctness is unaffected.
set -eu
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"   # Homebrew's rustc has no wasm target; rustup's does

mode="${1:-quick}"
results=bench/results
baselines=bench/baselines
mkdir -p "$results" "$baselines"

say() { printf '\n== %s\n' "$*"; }

# Benchmark sizes and number of timed runs per size for each mode.
case "$mode" in
  quick) sizes="--ntt 14,16,18 --msm 14,16,18"; runs=5 ;;
  *)     sizes="--ntt 14,16,18,20,22 --msm 14,16,18,20"; runs=7 ;;
esac

# Tests share one GPU; running them in parallel makes them slow and timing-dependent.
run_tests() {
  say "tests ($mode)"
  cargo test --release --example compare
  if [ "$mode" = quick ]; then export ZK_GPU_QUICK=1; fi
  cargo test --release -- --test-threads=1
}

# Runs the benchmark `$1` times into $results/run-N.json.
run_bench() {
  say "benchmark ($1 runs, $runs timed runs per size)"
  cargo build --release --example bench --example compare
  i=1
  while [ "$i" -le "$1" ]; do
    ./target/release/examples/bench $sizes --runs "$runs" --out "$results/run-$i.json"
    i=$((i + 1))
  done
}

case "$mode" in
  quick|full)
    say "lint"
    cargo clippy --release --all-targets -- -D warnings
    if [ "$mode" = full ]; then
      say "wasm build"
      cargo build --release --target wasm32-unknown-unknown --lib
    fi
    run_tests
    rm -f "$results"/run-*.json
    run_bench 2
    slug=$(sed -n 's/.*"device_slug": "\(.*\)".*/\1/p' "$results/run-1.json")
    baseline="$baselines/$slug.json"
    say "compare with $baseline (best of 2 runs)"
    if [ ! -f "$baseline" ]; then
      echo "no baseline for this device yet: run './check.sh baseline' once, then re-run this."
      exit 0
    fi
    ./target/release/examples/compare "$baseline" "$results"/run-*.json
    ;;

  baseline)
    rm -f "$results"/run-*.json
    run_bench 3
    slug=$(sed -n 's/.*"device_slug": "\(.*\)".*/\1/p' "$results/run-1.json")
    ./target/release/examples/compare --merge "$baselines/$slug.json" "$results"/run-*.json
    echo "saved $baselines/$slug.json"
    ;;

  browser)
    # wasm-pack downloads a chromedriver itself. If it does not match your Chrome, point
    # CHROMEDRIVER at one that does.
    driver=""
    if [ -n "${CHROMEDRIVER:-}" ]; then
      driver="--chromedriver $CHROMEDRIVER"
    fi
    say "browser correctness (WebGPU in headless Chrome)"
    wasm-pack test --headless --chrome --release $driver --test web
    if [ "${2:-}" = bench ]; then
      say "browser benchmark"
      WASM_BINDGEN_TEST_TIMEOUT=900 wasm-pack test --headless --chrome --release $driver --test web_bench -- --nocapture
    fi
    ;;

  golden)
    say "regenerate golden vectors"
    (cd tools/golden && npm install --no-audit --no-fund && node gen.mjs)
    ;;

  *)
    echo "usage: ./check.sh quick|full|browser [bench]|baseline|golden" >&2
    exit 2
    ;;
esac
say "done: $mode OK"
