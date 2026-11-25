import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class SingleTaskDemoTest(unittest.TestCase):
    def test_both_languages_record_the_three_expected_judgments(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "records.jsonl"
            run = subprocess.run(
                [sys.executable, "-m", "prompt_language_comparison", "--output", str(output)],
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(len(records), 6)
        pairs = {(r["prompt_version"], r["repeat"]): r for r in records}
        self.assertEqual(set(pairs), {(lang, n) for lang in ("en", "zh") for n in (1, 2, 3)})

        for repeat, expected in enumerate(
            (("passed", "passed"), ("failed", "failed"), ("passed", "failed")), start=1,
        ):
            en, zh = pairs["en", repeat], pairs["zh", repeat]
            for record in (en, zh):
                with self.subTest(language=record["prompt_version"], repeat=repeat):
                    self.assertEqual(record["task_id"], "HumanEval/0")
                    self.assertEqual(record["stage"], "demo")
                    self.assertEqual(record["model"], "fixed-replies")
                    self.assertIn("def has_close_elements(", record["code"])
                    self.assertNotIn("```", record["code"])
                    self.assertIn(record["code"].strip(), record["raw_reply"])
                    self.assertNotIn("def check(candidate)", record["prompt"])
                    self.assertNotIn("[1.0, 2.0, 3.9, 4.0, 5.0, 2.2]", record["prompt"])
                    for suite, status in zip(("humaneval", "humaneval_plus"), expected):
                        result = record["scores"][suite]
                        self.assertEqual(result["status"], status)
                        if status == "failed":
                            self.assertTrue(result["reason"])
            self.assertEqual(en["code"], zh["code"])
            self.assertEqual(en["prompt"].split('"""', 1)[0], zh["prompt"].split('"""', 1)[0])
            self.assertEqual(en["prompt"].split(">>>", 1)[1], zh["prompt"].split(">>>", 1)[1])
            self.assertIn("Check if in given list", en["prompt"])
            self.assertIn("严格小于", zh["prompt"])


if __name__ == "__main__":
    unittest.main()
