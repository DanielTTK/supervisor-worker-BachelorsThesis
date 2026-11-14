"""Focused tests for readable benchmark result reports."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from report_results import render_report


class ReportResultsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name) / "run_under_test"
        self.run_dir.mkdir()
        manifest = {
            "backend": "ollama",
            "cases": 24,
            "inference": {"worker": "worker_model", "supervisor": "supervisor_model"},
            "conditions": {
                "workers_only": {"worker": {}, "synthesis": {}},
                "generic": {"worker": {}, "review": {}, "synthesis": {}},
                "custom": {"worker": {}, "review": {}, "synthesis": {}},
            },
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_completed_report_separates_end_to_end_and_replay(self):
        columns = (
            "arm", "scenario", "case_kind", "scope", "metric", "numerator",
            "denominator", "percent", "ci_low", "ci_high", "comparator",
            "comparator_numerator", "comparator_denominator",
        )
        rows = [
            {
                "arm": "workers_only", "scenario": "natural", "case_kind": "all",
                "scope": "all", "metric": "final_exact_accuracy", "numerator": "12",
                "denominator": "24", "percent": "50", "ci_low": "40", "ci_high": "60",
                "comparator": "", "comparator_numerator": "", "comparator_denominator": "",
            },
            {
                "arm": "custom", "scenario": "replay", "case_kind": "all",
                "scope": "all", "metric": "review_miss_rate", "numerator": "2",
                "denominator": "10", "percent": "20", "ci_low": "", "ci_high": "",
                "comparator": "", "comparator_numerator": "", "comparator_denominator": "",
            },
            {
                "arm": "custom", "scenario": "replay", "case_kind": "all",
                "scope": "all", "metric": "paired_miss_difference_pp", "numerator": "2",
                "denominator": "10", "percent": "-10", "ci_low": "-20", "ci_high": "0",
                "comparator": "generic", "comparator_numerator": "3", "comparator_denominator": "10",
            },
        ]
        with (self.run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (self.run_dir / "complete.json").write_text("{}", encoding="utf-8")

        markdown, latex = render_report(self.run_dir)

        self.assertIn("End-to-end results", markdown)
        self.assertIn("Reviewer replay (same saved inputs)", markdown)
        self.assertIn("50.0% (12/24); 95% CI 40.0–60.0", markdown)
        self.assertIn("-10.0 pp", markdown)
        self.assertIn("Custom review - Generic review", markdown)
        self.assertIn(r"\begin{tabularx}", latex)
        self.assertIn(r"\%", latex)
        self.assertEqual(latex.count(r"\end{itemize}"), 1)

    def test_incomplete_report_shows_progress_without_scored_metrics(self):
        import sqlite3

        connection = sqlite3.connect(self.run_dir / "runs.sqlite3")
        connection.execute("CREATE TABLE events (key TEXT PRIMARY KEY, event TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO events VALUES (?, ?)",
            ("event-1", json.dumps({
                "scenario": "natural", "arm": "generic", "stage": "worker", "status": "ok",
            })),
        )
        connection.commit()
        connection.close()

        markdown, latex = render_report(self.run_dir)

        self.assertIn("no scored summary is available", markdown)
        self.assertIn("1 stored events", markdown)
        self.assertIn("operational progress, not performance metrics", markdown)
        self.assertNotIn("Exact final answers", markdown)
        self.assertIn("Stored events", latex)


if __name__ == "__main__":
    unittest.main()
