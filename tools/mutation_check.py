#!/usr/bin/env python3
"""Mutation check: proves the test suite can actually catch bugs.

Each mutant is a small deliberate bug (a changed constant, a dropped special case...). It is
applied to the source, the test suite is run, and the source is restored. A mutant that makes no
test fail ("SURVIVED") points at a hole in the tests. The performance mutants keep the result
correct but make the code slower, and are checked against the saved benchmark baseline instead.

    python3 tools/mutation_check.py              # all correctness mutants (about 20 minutes: every test runs against every mutant)
    python3 tools/mutation_check.py digits-carry # only the named mutants
    python3 tools/mutation_check.py --perf       # performance mutants (slow: runs benchmarks)
    python3 tools/mutation_check.py --stale      # only check that every mutant's text still occurs exactly once (seconds)
"""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]

# (name, file, text to find, text to put instead). The text must occur exactly once.
CORRECTNESS = [
    ("digits-carry",      "src/shaders/msm_digits.wgsl",  "neg = 1u; carry = 1u;", "neg = 1u;"),
    ("digits-mask",       "src/shaders/msm_digits.wgsl",  "& ((1u << C) - 1u)", "& ((1u << C) - 2u)"),
    ("combine-shift",     "src/shaders/msm_combine.wgsl", "k < C; k++", "k < C - 1u; k++"),
    ("mixed-no-doubling", "src/shaders/curve.wgsl",       "let r = fe_sub(s2, p.y);\n  if (fe_is_zero(pp)) {\n    if (fe_is_zero(r)) { return xyzz_double(p); }", "let r = fe_sub(s2, p.y);\n  if (fe_is_zero(pp)) {\n    if (fe_is_zero(r)) { return xyzz_infinity(); }"),
    ("full-no-doubling",  "src/shaders/curve.wgsl",       "let r = fe_sub(s2, s1);\n  if (fe_is_zero(pp)) {\n    if (fe_is_zero(r)) { return xyzz_double(p); }", "let r = fe_sub(s2, s1);\n  if (fe_is_zero(pp)) {\n    if (fe_is_zero(r)) { return xyzz_infinity(); }"),
    ("mixed-no-infinity", "src/shaders/curve.wgsl",       "if (affine_is_infinity(q)) { return p; }", ""),
    ("finalize-remainder","src/shaders/msm_finalize.wgsl","(counts[slot] + sizes[2] - 1u) / sizes[2]", "counts[slot] / sizes[2]"),
    ("ntt-no-scaling",    "src/ntt.rs",                   "Direction::new(gpu, log_n, &w_inv, Some(n_inv))", "Direction::new(gpu, log_n, &w_inv, None)"),
    ("ntt-twiddle-op",    "src/shaders/ntt.wgsl",         "fr_mul(fr_sub(a, b), twiddle)", "fr_mul(fr_add(a, b), twiddle)"),
    ("field-sub-addback", "src/field.rs",                 "select(d{j}, e{j}, br == 1u)", "select(d{j}, e{j}, br == 2u)"),
    ("field-reduce",      "src/field.rs",                 "select(s{j}, {v}{j}, br == 1u)", "select(s{j}, {v}{j}, br == 2u)"),
    ("merge-skipped",     "src/shaders/msm_merge.wgsl",   "if (k % (s * G) != 0u) { return; }", "return;"),
    ("merge-stride",      "src/shaders/msm_merge.wgsl",   "load_item(base + j * s)", "load_item(base + j)"),
    ("slice-offset",      "src/shaders/msm_accumulate.wgsl", "let window = id.y + slice[0];", "let window = id.y + 0u * slice[0];"),
    ("chunk-offset",      "src/shaders/msm_reduce_chunks.wgsl", "let t = id.x + slice[0];", "let t = id.x + 0u * slice[0];"),
    ("g2-mul-cross",      "src/group.rs",                 "fq_sub(fq_sub(cross, ac), bd)", "fq_sub(cross, ac)"),
    ("g2-sqr",            "src/group.rs",                 "fq_add(ab, ab))", "ab)"),
]
# These stay correct but get slower; the benchmark comparison must flag them.
PERFORMANCE = [
    ("slow-window",       "src/msm.rs",                   ".clamp(11, 16)", ".clamp(9, 10)"),
    ("slow-items",        "src/msm.rs",                   ".clamp(8, 1024)", ".clamp(1024, 1024)"),
]


def run(cmd, timeout):
    return subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True, text=True, timeout=timeout)


def with_mutation(file, old, new, action):
    path = os.path.join(ROOT, file)
    original = open(path).read()
    if original.count(old) != 1:
        sys.exit(f"mutation text must occur exactly once in {file}: {old!r} (found {original.count(old)})")
    try:
        open(path, "w").write(original.replace(old, new))
        return action()
    finally:
        open(path, "w").write(original)


def check_correctness(selected):
    survived = []
    for name, file, old, new in CORRECTNESS:
        if selected and name not in selected:
            continue
        def action():
            r = run("ZK_GPU_QUICK=1 cargo test --release --no-fail-fast -- --test-threads=1", timeout=1800)
            return r.returncode, r.stdout + r.stderr
        code, output = with_mutation(file, old, new, action)
        if code == 0:
            survived.append(name)
            print(f"SURVIVED  {name}: no test failed")
        else:
            caught = re.findall(r"^test (\S+) \.\.\. FAILED", output, re.M)
            reason = f"{len(caught)} failing: " + ", ".join(caught) if caught else "build/other failure: " + output.strip().splitlines()[-1][:80]
            print(f"caught    {name}: {reason}")
    run("cargo build --release --tests", timeout=1800)  # leave a clean build behind
    return survived


def check_performance():
    missed = []
    for name, file, old, new in PERFORMANCE:
        def action():
            run("cargo build --release --example bench --example compare", timeout=1800)
            out = "bench/results/mutant.json"
            r = run(f"./target/release/examples/bench --ntt 14 --msm 16,18 --runs 5 --out {out}", timeout=1800)
            if r.returncode != 0:
                return 99, r.stdout + r.stderr
            slug = re.search(r'"device_slug": "(.*?)"', open(os.path.join(ROOT, out)).read()).group(1)
            c = run(f"./target/release/examples/compare bench/baselines/{slug}.json {out}", timeout=60)
            return c.returncode, c.stdout
        code, output = with_mutation(file, old, new, action)
        if code == 1:
            print(f"caught    {name}: flagged as a regression")
        else:
            missed.append(name)
            print(f"MISSED    {name}: compare exit code {code}\n{output}")
    run("cargo build --release --example bench --example compare", timeout=1800)
    return missed


def find_stale():
    """Mutants whose text no longer occurs exactly once in their file (after an edit or a reformat)."""
    stale = []
    for name, file, old, _ in CORRECTNESS + PERFORMANCE:
        found = open(os.path.join(ROOT, file)).read().count(old)
        if found != 1:
            stale.append(name)
            print(f"STALE     {name}: {file} contains the mutation text {found} times (needs exactly 1)")
    return stale


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--stale"]:
        stale = find_stale()
        print("\nno stale mutants" if not stale else f"\n{len(stale)} stale mutant(s): {', '.join(stale)}")
        sys.exit(1 if stale else 0)
    bad = check_performance() if args == ["--perf"] else check_correctness(set(args))
    print("\nall mutants caught" if not bad else f"\n{len(bad)} mutant(s) not caught: {', '.join(bad)}")
    sys.exit(1 if bad else 0)
