"""Export unbinned training-window statistics as strict JSON and Markdown.

Only the standard library is required. Input SQLite databases are opened read-only.
"""
import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from sysmon_run import iso_utc, open_readonly, resolve_inputs, resolve_window


GPU_METRICS = {"util_percent": "%", "temp_c": "degC", "power_w": "W",
               "mem_used_gb": "GiB", "mem_total_gb": "GiB"}
HOST_METRICS = {"cpu_percent": "%", "mem_percent": "%", "cpu_temp_c": "degC", "rapl_watts": "W"}


def summarize(values, unit):
    finite = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if finite:
        rank = (len(finite) - 1) * 0.95
        low, high = math.floor(rank), math.ceil(rank)
        p95 = finite[low] + (finite[high] - finite[low]) * (rank - low)
        mean, peak = statistics.fmean(finite), max(finite)
    else:
        mean = p95 = peak = None
    return {"unit": unit, "valid_samples": len(finite), "missing_samples": len(values) - len(finite),
            "mean": mean, "p95": p95, "peak": peak}


def coverage(rows, start, end, interval=None):
    times = sorted(set(float(row["ts"]) for row in rows))
    gaps = [b - a for a, b in zip(times, times[1:])]
    cadence = interval or (statistics.median(gaps) if gaps else None)
    return {"rows": len(rows), "unique_timestamps": len(times), "duplicate_timestamps": len(rows) - len(times),
            "first_utc": iso_utc(times[0]) if times else None,
            "last_utc": iso_utc(times[-1]) if times else None,
            "median_interval_seconds": statistics.median(gaps) if gaps else None,
            "expected_interval_seconds": interval,
            "max_gap_seconds": max(gaps) if gaps else None,
            "gaps_over_3_intervals": sum(g > 3 * cadence for g in gaps) if cadence else None,
            "start_lag_seconds": times[0] - start if times else None,
            "end_lag_seconds": end - times[-1] if times else None,
            "cadence_used_seconds": cadence}


def series_report(rows, metrics, start, end, interval):
    return {"coverage": coverage(rows, start, end, interval),
            "metrics": {name: summarize([row[name] for row in rows], unit) for name, unit in metrics.items()}}


def disk_report(rows, start, end, interval):
    fields = {"read_MBps": ("read_bytes", 1_000_000, "MB/s"),
              "write_MBps": ("write_bytes", 1_000_000, "MB/s"),
              "read_IOPS": ("read_count", 1, "ops/s"),
              "write_IOPS": ("write_count", 1, "ops/s"),
              "util_percent_approx": ("busy_ms", 10, "%")}
    rates = {name: [] for name in fields}
    resets = 0
    for left, right in zip(rows, rows[1:]):
        dt = right["ts"] - left["ts"]
        for name, (counter, scale, _) in fields.items():
            before, after = left[counter], right[counter]
            if dt <= 0 or before is None or after is None:
                rate = None
            elif after < before:
                resets += 1
                rate = None
            else:
                rate = (after - before) / dt / scale
                if name == "util_percent_approx":
                    rate = min(100.0, rate)
            rates[name].append(rate)
    return {"coverage": coverage(rows, start, end, interval), "counter_reset_fields": resets,
            "metrics": {name: summarize(rates[name], unit) for name, (_, _, unit) in fields.items()}}


def build_report(db, window, *, run_dir=None, expected_gpus=None, sampling_interval=None):
    start, end = window["start_epoch"], window["end_epoch"]
    with closing(open_readonly(db)) as conn:
        conn.row_factory = sqlite3.Row
        integrity = conn.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite quick_check failed: {integrity}")
        def load(table):
            return list(conn.execute(f"SELECT * FROM {table} WHERE ts >= ? AND ts <= ? ORDER BY ts", (start, end)))
        host, gpu, disk = load("host_metrics"), load("gpu_metrics"), load("disk_metrics")
    if not host and not gpu and not disk:
        raise ValueError("No metrics in the requested time window; check metadata, DB and UTC times")
    groups = defaultdict(list)
    for row in gpu:
        groups[int(row["gpu_index"])].append(row)
    gpus = {str(index): {"name": str(rows[-1]["name"]),
                        **series_report(rows, GPU_METRICS, start, end, sampling_interval)}
            for index, rows in sorted(groups.items())}
    disks = defaultdict(list)
    for row in disk:
        disks[str(row["device"])].append(row)
    disk_summaries = {device: disk_report(rows, start, end, sampling_interval) for device, rows in sorted(disks.items())}
    host_summary = series_report(host, HOST_METRICS, start, end, sampling_interval)
    totals = []
    by_time = defaultdict(list)
    for row in gpu:
        by_time[row["ts"]].append(row)
    required = expected_gpus if expected_gpus is not None else len(groups)
    for ts, rows in sorted(by_time.items()):
        # Duplicate/missing GPU rows must not create an apparently complete total.
        if len(rows) == required and len({r["gpu_index"] for r in rows}) == required:
            total = {"ts": ts}
            for field in ("power_w", "mem_used_gb"):
                vals = [row[field] for row in rows]
                total[field] = sum(vals) if all(v is not None and math.isfinite(v) for v in vals) else None
            totals.append(total)
    warnings = []
    if not expected_gpus:
        warnings.append("Expected GPU count was not supplied; missing hardware cannot be ruled out.")
    elif len(groups) != expected_gpus:
        warnings.append(f"Expected {expected_gpus} GPUs, observed {len(groups)} in the selected window.")
    for label, item in [("host", host_summary), *[(f"GPU {k}", v) for k, v in gpus.items()],
                        *[(f"disk {k}", v) for k, v in disk_summaries.items()]]:
        cov = item["coverage"]
        cadence = cov["cadence_used_seconds"]
        if not cov["rows"]:
            warnings.append(f"{label}: no samples.")
        elif cov["unique_timestamps"] < 2:
            warnings.append(f"{label}: fewer than two unique sample timestamps; coverage cannot be assessed.")
        elif cov["gaps_over_3_intervals"] or cov["start_lag_seconds"] > 3 * cadence or cov["end_lag_seconds"] > 3 * cadence:
            warnings.append(f"{label}: incomplete boundary coverage or gaps above 3 sampling intervals.")
        if cov["duplicate_timestamps"]:
            warnings.append(f"{label}: duplicate timestamps; sample-weighted statistics may be biased.")
        for name, metric in item["metrics"].items():
            if metric["missing_samples"] or not metric["valid_samples"]:
                warnings.append(f"{label} {name}: missing/invalid samples; see valid_samples and missing_samples.")
    if not disk_summaries:
        warnings.append("No disk data in the selected window.")
    if not groups:
        warnings.append("No GPU data in the selected window.")
    if any(item["counter_reset_fields"] for item in disk_summaries.values()):
        warnings.append("Disk counter resets detected; negative deltas were excluded, not reported as negative I/O.")
    if not totals:
        warnings.append("No complete simultaneous GPU samples for total power/VRAM.")
    trainer_exit = None
    if run_dir:
        exit_file = Path(run_dir) / "metadata" / "trainer-exit-code.txt"
        if exit_file.is_file():
            trainer_exit = exit_file.read_text().strip()
    if trainer_exit != "0":
        warnings.append("Training success is unconfirmed: trainer-exit-code is missing or nonzero.")
    return {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_db": str(Path(db).resolve()), "run_id": Path(run_dir).name if run_dir else None,
        "trainer_exit_code": trainer_exit, "sqlite_quick_check": integrity,
        "window": {**window, "duration_hours": (end - start) / 3600},
        "method": {"mean": "arithmetic sample mean, not time-weighted; no resampling or zero filling",
                   "p95": "95th percentile with linear interpolation at rank (n-1)*0.95",
                   "peak": "maximum recorded sample, not an instantaneous hardware maximum",
                   "missing": "null/nonfinite values excluded; no interpolation across missing samples",
                   "all_gpu_samples": "pooled per-GPU sample distribution; NOT total GPU power",
                   "gpu_totals": "sums only at exact timestamps with all required GPUs and finite values",
                   "disk": "rates between consecutive in-window samples; negative counter deltas excluded; utilization approximate",
                   "assessment": "statistics and warnings only; not an automatic hardware PASS/FAIL"},
        "expected_gpu_count": expected_gpus, "observed_gpu_count": len(groups),
        "host": host_summary, "gpus": gpus,
        "all_gpu_samples": {name: summarize([row[name] for row in gpu], unit) for name, unit in GPU_METRICS.items()},
        "gpu_totals": series_report(totals, {"power_w": "W", "mem_used_gb": "GiB"}, start, end, sampling_interval),
        "disks": disk_summaries, "warnings": warnings,
    }


def markdown_report(report):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# Sysmon QA statistics", "", f"Run: {cell(report['run_id'])}",
             f"UTC window: {report['window']['start_utc']} to {report['window']['end_utc']}",
             f"Duration: {report['window']['duration_hours']:.4f} hours",
             f"Buffer: {report['window']['buffer_minutes']} minutes per side",
             f"GPUs observed / expected: {report['observed_gpu_count']} / {report['expected_gpu_count']}",
             f"Trainer exit code: {cell(report['trainer_exit_code'])}", "",
             "Means and P95 use finite raw samples, not resampled charts or time weighting.",
             "P95 is a percentile, not 95% utilization. Peaks are sampled maxima.", "",
             "## Warnings", ""]
    lines.extend([f"- {cell(w)}" for w in report["warnings"]] or ["- No automatic warnings; this is not a PASS verdict."])
    def metric_table(title, metrics):
        lines.extend(["", f"## {cell(title)}", "", "| Metric | Unit | Mean | P95 | Peak | Valid | Missing |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for name, stat in metrics.items():
            vals = ["N/A" if stat[key] is None else f"{stat[key]:.4f}" for key in ("mean", "p95", "peak")]
            lines.append(f"| {cell(name)} | {stat['unit']} | {' | '.join(vals)} | {stat['valid_samples']} | {stat['missing_samples']} |")
    metric_table("All GPU samples pooled (not total power)", report["all_gpu_samples"])
    for index, gpu in report["gpus"].items():
        metric_table(f"GPU {index} {gpu['name']}", gpu["metrics"])
    metric_table("Simultaneous GPU totals", report["gpu_totals"]["metrics"])
    metric_table("Host", report["host"]["metrics"])
    for device, disk in report["disks"].items():
        metric_table(f"Disk {device}", disk["metrics"])
    lines.extend(["", "## Coverage", "", "| Series | Rows | First UTC | Last UTC | Max gap s | Gaps over 3 intervals |",
                  "| --- | ---: | --- | --- | ---: | ---: |"])
    for label, item in [("host", report["host"]), *[(f"GPU {k}", v) for k, v in report["gpus"].items()]]:
        c = item["coverage"]
        lines.append(f"| {label} | {c['rows']} | {c['first_utc']} | {c['last_utc']} | {c['max_gap_seconds']} | {c['gaps_over_3_intervals']} |")
    lines.extend(["", "## Method", ""] + [f"- {key}: {value}" for key, value in report["method"].items()])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--db")
    parser.add_argument("--from", dest="start", help="ISO 8601 or epoch; naive CLI times mean UTC")
    parser.add_argument("--to", dest="end")
    parser.add_argument("--buffer-minutes", type=float, default=0, help="Statistics buffer (default: 0; exclude idle context)")
    parser.add_argument("--expected-gpus", type=int)
    parser.add_argument("--sampling-interval", type=float, help="Expected sampling seconds; otherwise infer median cadence")
    parser.add_argument("--out-dir", required=True, help="Write summary.json and summary.md here")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_dir, db = resolve_inputs(args.run_dir, args.db)
        window = resolve_window(run_dir, args.start, args.end, args.buffer_minutes)
        if window is None:
            raise ValueError("Statistics require run metadata or both --from and --to")
        if args.expected_gpus is not None and args.expected_gpus <= 0:
            raise ValueError("--expected-gpus must be positive")
        if args.sampling_interval is not None and (not math.isfinite(args.sampling_interval) or args.sampling_interval <= 0):
            raise ValueError("--sampling-interval must be finite and positive")
        output = Path(args.out_dir).expanduser().resolve()
        targets = [output / "summary.json", output / "summary.md"]
        if db in targets:
            raise ValueError("Output must not replace the input database")
        if not args.overwrite and any(p.exists() for p in targets):
            raise ValueError("Output exists; choose another --out-dir or explicitly use --overwrite")
        report = build_report(db, window, run_dir=run_dir, expected_gpus=args.expected_gpus,
                              sampling_interval=args.sampling_interval)
        report["statistics_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        document = markdown_report(report)
        output.mkdir(parents=True, exist_ok=True)
        targets[0].write_text(encoded, encoding="utf-8")
        targets[1].write_text(document, encoding="utf-8")
        print(f"Saved {targets[0]} and {targets[1]}")
        print(f"Warnings: {len(report['warnings'])}; review before making a QA verdict.")
        return 0
    except (OSError, ValueError, sqlite3.Error, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
