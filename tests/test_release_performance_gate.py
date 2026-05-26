import json
import runpy
import tempfile
import unittest
from pathlib import Path

gate = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "reproduce" / "check_release_performance.py")
)
ROOT = gate["ROOT"]
evaluate_suite = gate["evaluate_suite"]
resolve_path = gate["resolve_path"]


class ReleasePerformanceGateTest(unittest.TestCase):
    def test_evaluate_suite_passes_lower_and_higher_metrics(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            root = Path(tmp_dir)
            report = root / "report.json"
            baseline = root / "baseline.json"
            report.write_text(json.dumps({"metrics": {"mae": 1.1, "r2": 0.91}}), encoding="utf-8")
            baseline.write_text(
                json.dumps(
                    {
                        "suite": "test",
                        "version": 1,
                        "checks": [
                            {
                                "id": "toy",
                                "task": "regression",
                                "report": str(report.relative_to(ROOT)),
                                "metrics": [
                                    {
                                        "name": "mae",
                                        "path": ["metrics", "mae"],
                                        "baseline": 1.0,
                                        "direction": "lower",
                                        "max_delta": 0.2,
                                    },
                                    {
                                        "name": "r2",
                                        "path": ["metrics", "r2"],
                                        "baseline": 0.9,
                                        "direction": "higher",
                                        "max_drop": 0.02,
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate_suite(baseline.relative_to(ROOT))

        self.assertTrue(result["ok"])
        self.assertEqual(result["metric_count"], 2)

    def test_evaluate_suite_fails_when_metric_regresses(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            root = Path(tmp_dir)
            report = root / "report.json"
            baseline = root / "baseline.json"
            report.write_text(json.dumps({"mae": 1.25}), encoding="utf-8")
            baseline.write_text(
                json.dumps(
                    {
                        "suite": "test",
                        "version": 1,
                        "checks": [
                            {
                                "id": "toy",
                                "task": "regression",
                                "report": str(report.relative_to(ROOT)),
                                "metrics": [
                                    {
                                        "name": "mae",
                                        "path": ["mae"],
                                        "baseline": 1.0,
                                        "direction": "lower",
                                        "max_delta": 0.1,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate_suite(baseline.relative_to(ROOT))

        self.assertFalse(result["ok"])
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["results"][0]["status"], "fail")

    def test_selector_path_extracts_matching_list_item(self):
        data = {
            "results": [
                {"name": "A", "label": "old", "metric": 1.0},
                {"name": "A", "label": "new", "metric": 0.5},
            ]
        }

        value = resolve_path(data, ["results", {"name": "A", "label": "new"}, "metric"])

        self.assertEqual(value, 0.5)

    def test_allow_missing_marks_missing_report_without_failing(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            root = Path(tmp_dir)
            baseline = root / "baseline.json"
            baseline.write_text(
                json.dumps(
                    {
                        "suite": "test",
                        "version": 1,
                        "checks": [
                            {
                                "id": "toy",
                                "task": "regression",
                                "report": str((root / "missing.json").relative_to(ROOT)),
                                "metrics": [
                                    {
                                        "name": "mae",
                                        "path": ["mae"],
                                        "baseline": 1.0,
                                        "direction": "lower",
                                        "max_delta": 0.1,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate_suite(baseline.relative_to(ROOT), allow_missing=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["results"][0]["status"], "missing_allowed")


if __name__ == "__main__":
    unittest.main()
