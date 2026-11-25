import unittest

from prompt_language_comparison.report import build_report


def planned(tasks):
    return [
        {"stage": "pilot", "task_id": task, "model": "local", "prompt_version": lang, "repeat": attempt}
        for task in tasks for lang in ("en", "zh") for attempt in (1, 2, 3)
    ]


def recorded(row, passed):
    return {
        **row, "raw_reply": "def solve(): pass",
        "scores": {suite: {"status": "passed" if passed else "failed"} for suite in ("humaneval", "humaneval_plus")},
    }


class ReportTest(unittest.TestCase):
    def test_paired_scores_exclude_missing_and_duplicate_records(self):
        manifest = planned(["a", "b", "c"])
        records = []
        for row in manifest:
            if row["task_id"] == "b" and row["prompt_version"] == "zh" and row["repeat"] == 3:
                records.append({**row, "raw_reply": None, "scores": {}})
            else:
                passes = row["repeat"] <= (2 if row["prompt_version"] == "en" else 1)
                records.append(recorded(row, passes))
        records.append(records[-1].copy())
        report = build_report(manifest, records, bootstrap_samples=50)
        plus = [x for x in report["comparison"] if x["suite"] == "humaneval_plus"][0]
        self.assertEqual(plus["paired_tasks"], 1)
        self.assertAlmostEqual(plus["english_pass_at_1_pct"], 200 / 3)
        self.assertAlmostEqual(plus["chinese_pass_at_1_pct"], 100 / 3)
        self.assertAlmostEqual(plus["delta_pp"], -100 / 3)
        self.assertEqual(plus["ci95_pp"], [plus["delta_pp"], plus["delta_pp"]])
        self.assertEqual(report["coverage"]["duplicate_keys"], [["pilot", "c", "local", "zh", 3]])
        self.assertEqual(report["coverage"]["missing_replies"], [["pilot", "b", "local", "zh", 3]])

    def test_missing_pair_has_no_score_and_no_interval(self):
        report = build_report(planned(["a"]), [], bootstrap_samples=50)
        self.assertEqual(len(report["coverage"]["not_run"]), 6)
        self.assertTrue(all(x["delta_pp"] is None and x["ci95_pp"] is None for x in report["comparison"]))

    def test_paired_delta_can_be_positive_or_zero_on_different_suites(self):
        manifest = planned(["a"])
        records = []
        for row in manifest:
            record = recorded(row, True)
            if row["prompt_version"] == "en":
                record["scores"]["humaneval_plus"] = {"status": "failed"}
            records.append(record)
        result = {row["suite"]: row for row in build_report(manifest, records)["comparison"]}
        self.assertEqual(result["humaneval"]["delta_pp"], 0)
        self.assertEqual(result["humaneval_plus"]["delta_pp"], 100)
