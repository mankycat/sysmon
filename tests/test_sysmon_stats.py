import contextlib
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sysmon_run import resolve_window
from sysmon_stats import build_report, disk_report, main, summarize


class SummaryMathTests(unittest.TestCase):
    def test_mean_linear_p95_peak_and_missing(self):
        s = summarize([0, 10, 20, 30, None, float("nan"), float("inf")], "%")
        self.assertEqual((s["mean"], s["peak"]), (15, 30))
        self.assertAlmostEqual(s["p95"], 28.5)
        self.assertEqual((s["valid_samples"], s["missing_samples"]), (4, 3))

    def test_all_missing_is_null_not_zero(self):
        s = summarize([None, float("nan")], "W")
        self.assertIsNone(s["mean"])
        self.assertIsNone(s["peak"])
        self.assertIsNone(s["p95"])
        json.dumps(s, allow_nan=False)

    def test_disk_counter_reset_is_excluded(self):
        rows = [{"ts": t, "read_bytes": value, "write_bytes": value,
                 "read_count": value, "write_count": value, "busy_ms": value}
                for t, value in ((0, 100), (10, 200), (20, 0))]
        result = disk_report(rows, 0, 20, 10)
        self.assertEqual(result["metrics"]["read_IOPS"]["mean"], 10)
        self.assertEqual(result["metrics"]["read_IOPS"]["missing_samples"], 1)
        self.assertEqual(result["counter_reset_fields"], 5)


class DatabaseSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.db = self.run / "metrics/sysmon/metrics.db"
        self.db.parent.mkdir(parents=True)
        (self.run / "metadata").mkdir()
        (self.run / "metadata/start-time.txt").write_text("2026-11-06T13:00:00+00:00")
        (self.run / "metadata/end-time.txt").write_text("2026-11-06T13:02:00+00:00")
        (self.run / "metadata/trainer-exit-code.txt").write_text("0")
        self.window = resolve_window(self.run)
        self.start = self.window["start_epoch"]
        with sqlite3.connect(self.db) as conn:
            conn.executescript("""
                CREATE TABLE host_metrics(ts REAL, cpu_percent REAL, mem_percent REAL, cpu_temp_c REAL, rapl_watts REAL);
                CREATE TABLE gpu_metrics(ts REAL, gpu_index INTEGER, name TEXT, util_percent REAL,
                  mem_used_gb REAL, mem_total_gb REAL, power_w REAL, temp_c REAL);
                CREATE TABLE disk_metrics(ts REAL, device TEXT, read_bytes INTEGER, write_bytes INTEGER,
                  read_count INTEGER, write_count INTEGER, busy_ms INTEGER);
            """)
            for i, offset in enumerate((-60, 0, 60, 120, 180)):
                value = 9999 if offset in (-60, 180) else offset / 6 + 10
                ts = self.start + offset
                conn.execute("INSERT INTO host_metrics VALUES(?, ?, 20, 40, 50)", (ts, value))
                for gpu in range(2):
                    conn.execute("INSERT INTO gpu_metrics VALUES(?, ?, 'test GPU', ?, 10, 96, ?, 60)",
                                 (ts, gpu, value, value * 10 + gpu * 100))
                conn.execute("INSERT INTO disk_metrics VALUES(?, '__total__', ?, ?, ?, ?, ?)",
                             (ts, i * 60000000, i * 120000000, i * 600, i * 1200, i * 6000))

    def report(self, **kwargs):
        return build_report(self.db, self.window, run_dir=self.run, expected_gpus=2, sampling_interval=60, **kwargs)

    def test_exact_training_window_excludes_idle_context(self):
        result = self.report()
        metric = result["gpus"]["0"]["metrics"]["util_percent"]
        self.assertEqual((metric["mean"], metric["p95"], metric["peak"]), (20, 29, 30))
        self.assertEqual(result["host"]["coverage"]["rows"], 3)
        self.assertEqual(result["gpu_totals"]["metrics"]["power_w"]["mean"], 500)
        self.assertEqual(result["all_gpu_samples"]["power_w"]["mean"], 250)
        self.assertFalse(result["warnings"])

    def test_missing_gpu_and_gap_are_reported(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM gpu_metrics WHERE gpu_index=1")
            conn.execute("DELETE FROM host_metrics WHERE ts > ?", (self.start,))
        result = self.report()
        self.assertTrue(any("Expected 2 GPUs" in warning for warning in result["warnings"]))
        self.assertTrue(any("host:" in warning for warning in result["warnings"]))
        self.assertIsNone(result["gpu_totals"]["metrics"]["power_w"]["mean"])

    def test_cli_exports_strict_json_and_markdown_without_changing_db(self):
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        output = self.run / "qa-summary"
        args = ["--run-dir", str(self.run), "--expected-gpus", "2", "--out-dir", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        result = json.loads((output / "summary.json").read_text())
        self.assertEqual(result["observed_gpu_count"], 2)
        self.assertIn("GPU 0", (output / "summary.md").read_text())
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), before)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(args)
        self.assertEqual(raised.exception.code, 2)

    def test_empty_window_fails_instead_of_reporting_zero_load(self):
        with self.assertRaises(ValueError):
            build_report(self.db, resolve_window(start="0", end="100"))

    def test_total_peak_is_not_sum_of_individual_peaks(self):
        with sqlite3.connect(self.db) as conn:
            for offset, power in ((0, 300), (60, 200), (120, 100)):
                conn.execute("UPDATE gpu_metrics SET power_w=? WHERE gpu_index=1 AND ts=?", (power, self.start + offset))
        result = self.report()
        self.assertEqual(result["gpu_totals"]["metrics"]["power_w"]["peak"], 400)

    def test_missing_total_metric_is_not_treated_as_zero(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE gpu_metrics SET power_w=NULL WHERE gpu_index=1")
        result = self.report()
        self.assertIsNone(result["gpu_totals"]["metrics"]["power_w"]["mean"])
        self.assertEqual(result["gpu_totals"]["metrics"]["power_w"]["missing_samples"], 3)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
