import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from prompt_language_comparison.experiment import make_plan, run_experiment


class MissingBackend:
    def __init__(self):
        self.seeds = []

    def generate(self, prompt, seed):
        self.seeds.append(seed)
        return {"raw_reply": None, "error": "offline in test", "attempts": 3}


class ExperimentTest(unittest.TestCase):
    def test_only_enabled_models_contribute_slots(self):
        config = {"models": [{"id": "qwen2.5:3b", "enabled": True},
                             {"id": "api-slot", "enabled": False}]}
        pilot, formal = (make_plan(stage, config) for stage in ("pilot", "formal"))
        self.assertEqual((len(pilot), len(formal)), (48, 984))
        self.assertEqual({row["model"] for row in formal}, {"qwen2.5:3b"})
        self.assertTrue({row["task_id"] for row in pilot} <= {row["task_id"] for row in formal})

    def test_resume_keeps_existing_reply_attempts_once(self):
        contract = {"stage": "pilot", "seed": 42, "rows": [
            {"stage": "pilot", "model": "local", "task_id": "a", "prompt_version": lang, "repeat": rep}
            for lang in ("en", "zh") for rep in (1, 2, 3)
        ]}
        backend = MissingBackend()
        with TemporaryDirectory() as scratch:
            directory = Path(scratch)
            for _ in range(2):
                report = run_experiment(directory, contract, {"a": {"en": "English", "zh": "中文"}},
                                        {"a": {}}, {"local": backend})
            self.assertEqual(len((directory / "records.jsonl").read_text().splitlines()), 6)
            self.assertEqual(len(json.loads((directory / "report.json").read_text())["coverage"]["missing_replies"]), 6)
        self.assertEqual(backend.seeds, [42, 43, 44, 42, 43, 44])
        self.assertEqual(report["coverage"]["duplicate_keys"], [])

    def test_resume_repairs_only_an_interrupted_tail(self):
        rows = [
            {"stage": "pilot", "task_id": "a", "model": "local", "prompt_version": lang, "repeat": rep}
            for lang in ("en", "zh") for rep in (1, 2, 3)
        ]
        manifest = {"stage": "pilot", "seed": 42, "rows": rows}
        backend = MissingBackend()
        with TemporaryDirectory() as scratch:
            directory = Path(scratch)
            saved = {**rows[0], "raw_reply": None}
            replies = directory / "replies.jsonl"
            original = json.dumps(saved).encode() + b'\n'
            replies.write_bytes(original + b'{"raw_reply": "interrupted')
            report = run_experiment(
                directory, manifest, {"a": {"en": "a", "zh": "甲"}},
                {"a": {}}, {"local": backend}, generation_workers=2,
            )
            self.assertEqual(len(backend.seeds), 5)
            self.assertNotIn(b'interrupted', replies.read_bytes())
            self.assertEqual(report["coverage"]["unique_recorded"], 6)
            with replies.open("ab") as stream:
                stream.write(b"{malformed complete line}\n")
            with self.assertRaisesRegex(ValueError, "corrupt complete journal"):
                run_experiment(directory, manifest, {}, {}, {"local": backend})

    def test_regenerated_report_recovers_replies_and_durable_judge_errors(self):
        from prompt_language_comparison.experiment import regenerate_report

        rows = [
            {"stage": "pilot", "task_id": "a", "model": "local", "prompt_version": lang, "repeat": rep}
            for lang in ("en", "zh") for rep in (1, 2, 3)
        ]
        manifest = {"stage": "pilot", "seed": 42, "rows": rows}
        slot = ["pilot", "a", "local", "en", 1]
        with TemporaryDirectory() as scratch:
            directory = Path(scratch)
            (directory / "manifest.json").write_text(json.dumps(manifest))
            (directory / "replies.jsonl").write_text(json.dumps({**rows[0], "raw_reply": "valid code"}) + "\n")
            (directory / "judge-errors.jsonl").write_text(json.dumps({"key": slot, "reason": "container unavailable"}) + "\n")
            output = regenerate_report(directory)
            self.assertEqual(output["coverage"]["received"], 1)
            self.assertEqual(len(output["coverage"]["not_run"]), 5)
            self.assertEqual(output["coverage"]["ungraded"]["humaneval"], [slot])
            self.assertEqual(output["infrastructure_errors"][0]["reason"], "container unavailable")

    def test_formal_gate_requires_matching_complete_pilot_and_ceiling_decision(self):
        from prompt_language_comparison.experiment import digest, verify_pilots
        from prompt_language_comparison.tasks import plan

        prompts = {f"HumanEval/{i}": {"en": f"Task {i}", "zh": f"第 {i} 题"} for i in range(164)}
        models = {"local": {"backend": "ollama", "model_digest": "x", "base_url": "http://127.0.0.1:11439"}}
        contract = {"seed": 42, "models": models, "datasets": {"test": "fixture"},
                    "judge": {"image": "fixture", "worker_sha256": "1234", "container_sha256": "5678"}}

        def write_pilot(directory, phase, passed, *, incomplete=False):
            directory.mkdir(exist_ok=True)
            rows = plan(phase, models, seed=42)
            frozen = {**contract, "stage": phase, "rows": rows, "models": {
                "local": {**models["local"], "base_url": "http://127.0.0.1:11434"},
            }, "prompt_sha256": digest({row["task_id"]: prompts[row["task_id"]] for row in rows})}
            (directory / "manifest.json").write_text(json.dumps(frozen))
            scores = {suite: {"status": "passed" if passed else "failed"}
                      for suite in ("humaneval", "humaneval_plus")}
            replies = [{**row, "raw_reply": "code",
                        "prompt": prompts[row["task_id"]][row["prompt_version"]]} for row in rows]
            grades = [{**row, "scores": scores} for row in replies]
            for name, journal in (("replies.jsonl", replies), ("records.jsonl", grades[:-1] if incomplete else grades)):
                (directory / name).write_text("".join(json.dumps(row) + "\n" for row in journal))

        with TemporaryDirectory() as scratch:
            pilot, extra = (Path(scratch) / phase for phase in ("pilot", "extra-pilot"))
            write_pilot(pilot, "pilot", False, incomplete=True)
            with self.assertRaisesRegex(ValueError, "both grades"):
                verify_pilots(contract, prompts, pilot, extra)
            write_pilot(pilot, "pilot", False)
            self.assertEqual(verify_pilots(contract, prompts, pilot, extra)["evidence"][0]["groups"][0]["graded"], 24)
            write_pilot(pilot, "pilot", True)
            write_pilot(extra, "extra-pilot", True)
            with self.assertRaisesRegex(ValueError, "both pilots reached the ceiling"):
                verify_pilots(contract, prompts, pilot, extra)
            value = verify_pilots(contract, prompts, pilot, extra, "continue with scope")
            self.assertEqual([run["phase"] for run in value["evidence"]], ["pilot", "extra-pilot"])
            self.assertEqual(value["decision"], "continue with scope")

    def test_fixed_replies_reach_both_judges_and_the_paired_table(self):
        from prompt_language_comparison.__main__ import FIXED_REPLIES
        from prompt_language_comparison.tasks import build_prompts, load_tasks

        tasks = load_tasks()
        tid = "HumanEval/0"
        task_prompts = build_prompts(tasks[tid])
        replies = dict(FIXED_REPLIES)

        class FixedBackend:
            def generate(self, prompt, seed):
                key = ["correct", "incorrect", "base_only"][seed - 42]
                if prompt == task_prompts["zh"] and key == "base_only":
                    key = "correct"
                return {"raw_reply": replies[key]}

        rows = [
            {"stage": "fixture", "task_id": tid, "model": "fixed", "prompt_version": lang, "repeat": rep}
            for lang in ("en", "zh") for rep in (1, 2, 3)
        ]
        with TemporaryDirectory() as scratch:
            directory = Path(scratch)
            report = run_experiment(directory, {"stage": "fixture", "seed": 42, "rows": rows},
                                    {tid: task_prompts}, tasks, {"fixed": FixedBackend()}, generation_workers=2)
            self.assertTrue((directory / "report.md").exists())
            self.assertEqual(len((directory / "records.jsonl").read_text().splitlines()), 6)
            results = {row["suite"]: row for row in report["comparison"]}
            self.assertAlmostEqual(results["humaneval"]["delta_pp"], 0.0)
            self.assertAlmostEqual(results["humaneval_plus"]["delta_pp"], 100 / 3)
            self.assertEqual(results["humaneval_plus"]["paired_tasks"], 1)
