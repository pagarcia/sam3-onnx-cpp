import unittest
from parity_thresholds import check_thresholds


class ParityThresholdTests(unittest.TestCase):
    def test_valid_min_and_max(self):
        self.assertTrue(check_thresholds({"dice": .95, "error": .01},
            {"min": {"dice": .9}, "max": {"error": .02}})[0])

    def test_missing_max_metric_cannot_pass_as_zero(self):
        self.assertFalse(check_thresholds({}, {"max": {"error": .1}})[0])

    def test_nonfinite_or_nonnumeric_metrics_and_thresholds_fail(self):
        for bad in (float("nan"), float("inf"), -float("inf"), None, "0.1", True, 10**1000):
            for kind in ("min", "max"):
                with self.subTest(bad=bad, kind=kind):
                    self.assertFalse(check_thresholds({"dice": bad}, {kind: {"dice": .9}})[0])
                    self.assertFalse(check_thresholds({"dice": .95}, {kind: {"dice": bad}})[0])

    def test_no_or_malformed_gates_fail(self):
        for gates in ({}, {"min": {}}, {"max": []}, {"minimum": {"dice": .9}}, None):
            self.assertFalse(check_thresholds({"dice": 1}, gates)[0])

    def test_threshold_violations(self):
        passed, failures = check_thresholds({"dice": .8, "error": .1},
            {"min": {"dice": .9}, "max": {"error": .02}})
        self.assertFalse(passed)
        self.assertEqual(len(failures), 2)


if __name__ == "__main__":
    unittest.main()
