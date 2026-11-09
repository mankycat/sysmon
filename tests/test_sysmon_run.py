import sqlite3
import tempfile
import unittest
from pathlib import Path

from sysmon_run import open_readonly, parse_timestamp, resolve_inputs, resolve_window


class RunWindowTests(unittest.TestCase):
    def test_offset_and_naive_cli_times_mean_the_same_utc_instant(self):
        self.assertEqual(parse_timestamp("2026-11-06 13:00"), parse_timestamp("2026-11-06T21:00:00+08:00"))
        self.assertEqual(parse_timestamp("2026-11-06T13:00:00Z"), parse_timestamp("2026-11-06 13:00"))

    def test_metadata_requires_timezone(self):
        with self.assertRaises(ValueError):
            parse_timestamp("2026-11-06 13:00", require_timezone=True)

    def test_bad_windows(self):
        for kwargs in ({"start": "0"}, {"start": "2", "end": "1"}, {"buffer_minutes": -1},
                       {"start": "nan", "end": "100"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_window(**kwargs)

    def test_run_metadata_buffer_and_db_path_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "metadata").mkdir()
            (run / "metadata/start-time.txt").write_text("2026-11-06T13:00:00+00:00")
            (run / "metadata/end-time.txt").write_text("2026-11-08T13:00:00+00:00")
            db = run / "metrics/sysmon/metrics.db"
            db.parent.mkdir(parents=True)
            sqlite3.connect(db).close()
            self.assertEqual(resolve_inputs(db_path=db), (run.resolve(), db.resolve()))
            window = resolve_window(run, buffer_minutes=60)
            self.assertEqual(window["start_utc"], "2026-11-06T12:00:00+00:00")
            self.assertEqual(window["end_utc"], "2026-11-08T14:00:00+00:00")
            self.assertEqual(resolve_window(run)["buffer_minutes"], 0)

    def test_explicit_window_overrides_metadata(self):
        result = resolve_window("/nonexistent/run", "100", "200", 1)
        self.assertEqual((result["start_epoch"], result["end_epoch"]), (40, 260))

    def test_missing_db_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "absent.db"
            with self.assertRaises(FileNotFoundError):
                open_readonly(db)
            self.assertFalse(db.exists())

    def test_readonly_connection_rejects_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "test.db"
            sqlite3.connect(db).close()
            conn = open_readonly(db)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    conn.execute("CREATE TABLE forbidden(x)")
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
