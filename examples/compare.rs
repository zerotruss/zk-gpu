//! Compares benchmark runs from `bench` against a saved baseline and flags slowdowns.
//!
//!   cargo run --release --example compare -- <baseline.json> <new.json> [<new2.json> ...]
//!                                            [--threshold 0.10] [--min-ms 1.5] [--force]
//!
//!   cargo run --release --example compare -- --merge <out.json> <run1.json> <run2.json> ...
//!
//! Exit code: 0 = no regression, 1 = regression, 2 = runs are not comparable.
//! `--merge` combines several runs of the same benchmark into one report holding the best
//! (fastest) value of every statistic; use it to record a baseline.
//!
//! Method. The statistic is the fastest run ("min"): slowdowns from other load, power state or
//! scheduling only ever add time, so the minimum is the most repeatable number. With several new
//! files the best of all of them is used, which filters out a single disturbed run. A change only
//! counts when it exceeds the threshold AND 1.5x the baseline's own noise (its median-vs-min gap)
//! AND an absolute floor (default 1.5 ms): the same short kernels moved by up to 1.2 ms between
//! identical runs (NTT 2^16: 3.4 to 4.7 ms). A 1 ms floor gave 3 false alarms in 8 clean rounds, 1.5 ms gave none.
use serde_json::{json, Value};
use std::process::ExitCode;

#[derive(Debug, PartialEq)]
enum Verdict {
    Ok,
    Faster,
    Regression,
}

struct Row {
    label: String,
    base: f64,
    new: f64,
    allowed: f64,
    verdict: Verdict,
}

enum Outcome {
    NotComparable(String),
    Compared { rows: Vec<Row>, notes: Vec<String> },
}

fn num(v: &Value, path: &[&str]) -> Option<f64> {
    path.iter().try_fold(v, |v, key| v.get(key))?.as_f64()
}

fn find<'a>(report: &'a Value, op: &str, log_n: u64) -> Option<&'a Value> {
    report["results"].as_array()?.iter().find(|r| r["op"] == op && r["log_n"] == log_n)
}

fn compare(baseline: &Value, news: &[Value], threshold: f64, min_ms: f64, force: bool) -> Outcome {
    for key in ["device_slug", "cpu_threads"] {
        for new in news {
            if baseline["meta"][key] != new["meta"][key] && !force {
                return Outcome::NotComparable(format!(
                    "{key} differs (baseline {}, new {}); results from different machines cannot be compared (--force to override)",
                    baseline["meta"][key], new["meta"][key]
                ));
            }
        }
    }
    let mut notes = Vec::new();
    let power = |r: &Value| r["meta"]["power"].as_str().unwrap_or("unknown").to_string();
    if news.iter().any(|n| power(n) != power(baseline)) {
        notes.push(format!("power source differs from the baseline (baseline: {})", power(baseline)));
    }

    let mut rows = Vec::new();
    for base_result in baseline["results"].as_array().unwrap() {
        let (op, log_n) = (base_result["op"].as_str().unwrap(), base_result["log_n"].as_u64().unwrap());
        let new_results: Vec<&Value> = news.iter().filter_map(|n| find(n, op, log_n)).collect();
        if new_results.is_empty() {
            notes.push(format!("{op} 2^{log_n}: in the baseline but not in the new run"));
            continue;
        }
        for metric in ["gpu_ms", "gpu_e2e_ms"] {
            let Some(base_min) = num(base_result, &[metric, "min"]) else { continue };
            let Some(new_min) = new_results.iter().filter_map(|r| num(r, &[metric, "min"])).reduce(f64::min) else {
                continue;
            };
            // Baseline noise: how far its typical run sits above its best run.
            let noise = num(base_result, &[metric, "median"]).map_or(0.0, |m| m / base_min - 1.0);
            let allowed = threshold.max(1.5 * noise).max(min_ms / base_min);
            let ratio = new_min / base_min;
            let verdict = if ratio > 1.0 + allowed {
                Verdict::Regression
            } else if ratio < 1.0 - allowed {
                Verdict::Faster
            } else {
                Verdict::Ok
            };
            rows.push(Row {
                label: format!("{op} 2^{log_n} {metric}"),
                base: base_min,
                new: new_min,
                allowed,
                verdict,
            });
        }
        // The CPU side is the noisy one (bursts of other load made it up to 8x slower in a run), so it
        // is never judged, only reported: a big drift means the speedup figures are not comparable.
        let cpu = |r: &Value| num(r, &["cpu_ms", "min"]);
        let new_cpu = new_results.iter().filter_map(|r| cpu(r)).reduce(f64::min);
        if let (Some(base_cpu), Some(new_cpu)) = (cpu(base_result), new_cpu) {
            if (new_cpu / base_cpu - 1.0).abs() > 0.25 {
                notes.push(format!(
                    "{op} 2^{log_n}: CPU time is {:.2}x the baseline's (other load?); CPU speedups are not comparable",
                    new_cpu / base_cpu
                ));
            }
        }
    }
    for new in news {
        for r in new["results"].as_array().unwrap() {
            let (op, log_n) = (r["op"].as_str().unwrap(), r["log_n"].as_u64().unwrap());
            if find(baseline, op, log_n).is_none() {
                notes.push(format!("{op} 2^{log_n}: new size, no baseline to compare with"));
            }
        }
    }
    notes.dedup();
    Outcome::Compared { rows, notes }
}

fn is_stats(v: &Value) -> bool {
    ["median", "min", "max"].iter().all(|k| v.get(k).is_some_and(Value::is_number))
}

/// Combines runs of the same benchmark: every timing statistic keeps its best (smallest) value.
fn merge(reports: &[&Value]) -> Value {
    let first = reports[0];
    match first {
        Value::Object(_) if is_stats(first) => {
            let best = |key: &str| reports.iter().filter_map(|r| r[key].as_f64()).fold(f64::INFINITY, f64::min);
            json!({ "median": best("median"), "min": best("min"), "max": best("max") })
        }
        Value::Object(map) => {
            let mut merged: serde_json::Map<String, Value> = map
                .keys()
                .map(|k| (k.clone(), merge(&reports.iter().filter_map(|r| r.get(k)).collect::<Vec<_>>())))
                .collect();
            // Recompute the derived speedups from the merged statistics.
            if let (Some(cpu), Some(gpu)) = (merged.get("cpu_ms"), merged.get("gpu_e2e_ms").or(merged.get("gpu_ms"))) {
                let ratio = |k: &str| cpu[k].as_f64().unwrap() / gpu[k].as_f64().unwrap();
                let (speedup, best) = (ratio("median"), ratio("min"));
                merged.insert("speedup".into(), json!(speedup));
                merged.insert("speedup_best".into(), json!(best));
            }
            Value::Object(merged)
        }
        Value::Array(items) => {
            assert!(
                reports.iter().all(|r| r.as_array().is_some_and(|a| a.len() == items.len())),
                "runs have different benchmark layouts and cannot be merged"
            );
            Value::Array((0..items.len()).map(|i| merge(&reports.iter().map(|r| &r[i]).collect::<Vec<_>>())).collect())
        }
        other => other.clone(),
    }
}

fn load(path: &str) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path} is not valid JSON: {e}"))
}

fn main() -> ExitCode {
    let mut files = Vec::new();
    let (mut threshold, mut min_ms, mut force) = (0.10, 1.5, false);
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().is_some_and(|a| a == "--merge") {
        args.next();
        let out = args.next().expect("--merge needs an output file");
        let runs: Vec<Value> = args.map(|f| load(&f)).collect();
        assert!(!runs.is_empty(), "--merge needs at least one input file");
        let mut merged = merge(&runs.iter().collect::<Vec<_>>());
        merged["meta"]["merged_runs"] = json!(runs.len());
        std::fs::write(&out, serde_json::to_string_pretty(&merged).unwrap()).unwrap();
        println!("merged {} runs into {out}", runs.len());
        return ExitCode::SUCCESS;
    }
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threshold" => {
                threshold = args.next().and_then(|v| v.parse().ok()).expect("--threshold needs a number, e.g. 0.10")
            }
            "--min-ms" => min_ms = args.next().and_then(|v| v.parse().ok()).expect("--min-ms needs a number, e.g. 1.5"),
            "--force" => force = true,
            _ => files.push(arg),
        }
    }
    if files.len() < 2 {
        eprintln!(
            "usage: compare <baseline.json> <new.json> [<new2.json> ...] [--threshold 0.10] [--min-ms 1.5] [--force]"
        );
        return ExitCode::from(2);
    }
    let baseline = load(&files[0]);
    let news: Vec<Value> = files[1..].iter().map(|f| load(f)).collect();

    match compare(&baseline, &news, threshold, min_ms, force) {
        Outcome::NotComparable(why) => {
            println!("NOT COMPARABLE: {why}");
            ExitCode::from(2)
        }
        Outcome::Compared { rows, notes } => {
            println!(
                "baseline source {}  vs  new source {}",
                baseline["meta"]["source_hash"], news[0]["meta"]["source_hash"]
            );
            println!(
                "{:<28} {:>11} {:>11} {:>8} {:>9}  verdict",
                "metric (fastest run)", "baseline ms", "new ms", "ratio", "allowed"
            );
            for r in &rows {
                let verdict = match r.verdict {
                    Verdict::Ok => "ok",
                    Verdict::Faster => "FASTER",
                    Verdict::Regression => "REGRESSION",
                };
                println!(
                    "{:<28} {:>11.2} {:>11.2} {:>7.2}x {:>8.0}%  {verdict}",
                    r.label,
                    r.base,
                    r.new,
                    r.new / r.base,
                    r.allowed * 100.0
                );
            }
            for note in notes {
                println!("note: {note}");
            }
            let regressions = rows.iter().filter(|r| r.verdict == Verdict::Regression).count();
            println!("{} compared, {regressions} regression(s)", rows.len());
            ExitCode::from(u8::from(regressions > 0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A report with one NTT result whose fastest GPU run is `min` and median is `median`.
    fn report(device: &str, min: f64, median: f64) -> Value {
        json!({
            "meta": { "device_slug": device, "cpu_threads": 8, "power": "ac", "source_hash": "x" },
            "results": [{ "op": "ntt", "log_n": 18, "gpu_ms": { "min": min, "median": median, "max": median } }],
        })
    }

    fn verdict(baseline: &Value, news: &[Value]) -> Verdict {
        match compare(baseline, news, 0.10, 1.5, false) {
            Outcome::Compared { mut rows, .. } => rows.remove(0).verdict,
            Outcome::NotComparable(why) => panic!("unexpectedly not comparable: {why}"),
        }
    }

    #[test]
    fn identical_runs_are_ok() {
        let r = report("m1", 10.0, 10.2);
        assert_eq!(verdict(&r, &[r.clone()]), Verdict::Ok);
    }

    #[test]
    fn a_clear_slowdown_is_a_regression() {
        assert_eq!(verdict(&report("m1", 10.0, 10.2), &[report("m1", 13.0, 13.1)]), Verdict::Regression);
    }

    #[test]
    fn a_clear_speedup_is_reported_as_faster() {
        assert_eq!(verdict(&report("m1", 10.0, 10.2), &[report("m1", 7.0, 7.1)]), Verdict::Faster);
    }

    #[test]
    fn small_changes_inside_the_threshold_are_ok() {
        assert_eq!(verdict(&report("m1", 10.0, 10.2), &[report("m1", 10.9, 11.0)]), Verdict::Ok);
    }

    #[test]
    fn differences_below_the_floor_are_latency_noise() {
        // 2.5 ms -> 3.9 ms is +56% but only 1.4 ms: not judged. 2.5 -> 4.6 ms is a real change.
        assert_eq!(verdict(&report("m1", 2.5, 2.6), &[report("m1", 3.9, 4.0)]), Verdict::Ok);
        assert_eq!(verdict(&report("m1", 2.5, 2.6), &[report("m1", 4.6, 4.7)]), Verdict::Regression);
        // The floor does not loosen big kernels: 100 ms -> 112 ms is flagged.
        assert_eq!(verdict(&report("m1", 100.0, 101.0), &[report("m1", 112.0, 113.0)]), Verdict::Regression);
    }

    #[test]
    fn a_noisy_baseline_widens_the_band() {
        // Baseline median is 40% above its best run, so a 30% change is within the noise.
        assert_eq!(verdict(&report("m1", 10.0, 14.0), &[report("m1", 13.0, 13.0)]), Verdict::Ok);
    }

    #[test]
    fn the_best_of_several_runs_hides_one_disturbed_run() {
        let base = report("m1", 10.0, 10.2);
        let disturbed = report("m1", 20.0, 21.0);
        let clean = report("m1", 10.1, 10.3);
        assert_eq!(verdict(&base, &[disturbed.clone()]), Verdict::Regression);
        assert_eq!(verdict(&base, &[disturbed, clean]), Verdict::Ok);
    }

    #[test]
    fn different_machines_are_not_comparable() {
        assert!(matches!(
            compare(&report("m1", 10.0, 10.0), &[report("rtx", 5.0, 5.0)], 0.10, 1.5, false),
            Outcome::NotComparable(_)
        ));
        assert!(matches!(
            compare(&report("m1", 10.0, 10.0), &[report("rtx", 5.0, 5.0)], 0.10, 1.5, true),
            Outcome::Compared { .. }
        ));
    }

    #[test]
    fn merge_keeps_the_best_of_every_statistic() {
        let mut a = report("m1", 10.0, 12.0);
        let mut b = report("m1", 11.0, 11.5);
        a["results"][0]["cpu_ms"] = json!({ "min": 20.0, "median": 21.0, "max": 22.0 });
        b["results"][0]["cpu_ms"] = json!({ "min": 19.0, "median": 23.0, "max": 24.0 });
        let merged = merge(&[&a, &b]);
        let r = &merged["results"][0];
        assert_eq!(r["gpu_ms"]["min"], 10.0);
        assert_eq!(r["gpu_ms"]["median"], 11.5);
        assert_eq!(r["cpu_ms"]["min"], 19.0);
        assert_eq!(r["speedup_best"], 19.0 / 10.0);
        assert_eq!(merged["meta"]["device_slug"], "m1");
    }

    #[test]
    fn cpu_drift_is_noted_but_never_a_regression() {
        let mut base = report("m1", 10.0, 10.0);
        let mut new = report("m1", 10.0, 10.0);
        base["results"][0]["cpu_ms"] = json!({ "min": 20.0, "median": 20.0, "max": 20.0 });
        new["results"][0]["cpu_ms"] = json!({ "min": 60.0, "median": 60.0, "max": 60.0 });
        match compare(&base, &[new], 0.10, 1.5, false) {
            Outcome::Compared { rows, notes } => {
                assert!(rows.iter().all(|r| r.verdict == Verdict::Ok));
                assert!(notes.iter().any(|n| n.contains("CPU time")));
            }
            Outcome::NotComparable(_) => panic!(),
        }
    }

    #[test]
    fn missing_and_new_sizes_are_noted_not_failed() {
        let base = report("m1", 10.0, 10.0);
        let mut other = report("m1", 10.0, 10.0);
        other["results"][0]["log_n"] = json!(20);
        match compare(&base, &[other], 0.10, 1.5, false) {
            Outcome::Compared { rows, notes } => {
                assert!(rows.is_empty());
                assert_eq!(notes.len(), 2);
            }
            Outcome::NotComparable(_) => panic!(),
        }
    }
}
