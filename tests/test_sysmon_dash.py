import math
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from sysmon_dash import bin_series, deriv_per_sec


class BinSeriesTests(unittest.TestCase):
    def setUp(self):
        self.start = int(datetime(2026, 11, 6, 13, 0, tzinfo=timezone.utc).timestamp())
        self.ts = [self.start + 1.25, self.start + 3.5, self.start + 61.25]
        self.vals = [10.0, 20.0, 30.0]

    def test_sixty_second_bins_keep_correct_epoch_and_means(self):
        ts, vals = bin_series(self.ts, self.vals, 60)
        self.assertEqual(ts, [self.start, self.start + 60])
        self.assertEqual(vals, [15.0, 30.0])

    def test_epoch_seconds_do_not_depend_on_datetime_index_resolution(self):
        index = pd.to_datetime(self.ts, unit="s", utc=True)
        for unit in ("ms", "us", "ns"):
            with self.subTest(unit=unit):
                with patch("sysmon_dash.pd.to_datetime", return_value=index.as_unit(unit)):
                    ts, vals = bin_series(self.ts, self.vals, 60)
                self.assertEqual(ts, [self.start, self.start + 60])
                self.assertEqual(vals, [15.0, 30.0])

    def test_five_minute_bins(self):
        ts, vals = bin_series(self.ts, self.vals, 300)
        self.assertEqual(ts, [self.start])
        self.assertEqual(vals, [20.0])

    def test_disabled_binning_returns_original_samples(self):
        for bin_sec in (None, 0, -1):
            with self.subTest(bin_sec=bin_sec):
                ts, vals = bin_series(self.ts, self.vals, bin_sec)
                self.assertEqual(ts, self.ts)
                self.assertEqual(vals, self.vals)

    def test_empty_samples(self):
        self.assertEqual(bin_series([], [], 60), ([], []))

    def test_missing_bins_are_not_silently_removed(self):
        ts, vals = bin_series([self.start + 1, self.start + 121], [10.0, 30.0], 60)
        self.assertEqual(ts, [self.start, self.start + 60, self.start + 120])
        self.assertEqual(vals[0], 10.0)
        self.assertTrue(math.isnan(vals[1]))
        self.assertEqual(vals[2], 30.0)


class DerivPerSecondTests(unittest.TestCase):
    def test_rates_work_with_pandas_copy_on_write(self):
        ts, rates = deriv_per_sec([0.0, 2.0, 4.0], [10.0, 30.0, 50.0])
        self.assertEqual(ts, [2.0, 4.0])
        self.assertEqual(rates, [10.0, 10.0])

    def test_duplicate_timestamps_remain_supported(self):
        ts, rates = deriv_per_sec([0.0, 0.0, 2.0], [10.0, 12.0, 32.0])
        self.assertEqual(ts, [0.0, 2.0])
        self.assertEqual(rates, [2_000_000.0, 10.0])

    def test_insufficient_samples(self):
        self.assertEqual(deriv_per_sec([], []), ([], []))
        self.assertEqual(deriv_per_sec([1.0], [10.0]), ([], []))


if __name__ == "__main__":
    unittest.main()
