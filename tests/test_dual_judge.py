import gzip
import json
from pathlib import Path
import unittest

from prompt_language_comparison.container import judge


DATA = Path(__file__).resolve().parents[1] / 'prompt_language_comparison' / 'data'


def task_from_archives(task_id):
    def read_row(name):
        with gzip.open(DATA / name, 'rt', encoding='utf-8') as archive:
            return next(row for row in map(json.loads, archive) if row['task_id'] == task_id)

    base = read_row('HumanEval.jsonl.gz')
    plus = read_row('HumanEvalPlus-v0.1.10.jsonl.gz')
    return {
        **base,
        'plus_prompt': plus['prompt'],
        'plus_solution': plus['canonical_solution'],
        'base_input': plus['base_input'],
        'plus_input': plus['plus_input'],
        'atol': plus['atol'],
    }


class DualJudgeTest(unittest.TestCase):
    def test_humaneval_plus_accepts_float_roundoff_within_reference_tolerance(self):
        task = task_from_archives('HumanEval/4')
        code = (
            'def mean_absolute_deviation(numbers):\n'
            '    center = sum(numbers) / len(numbers)\n'
            '    deviation = sum(abs(x - center) for x in numbers) / len(numbers)\n'
            '    return round(deviation, 8)\n'
        )
        for suite in ('humaneval', 'humaneval_plus'):
            with self.subTest(suite=suite):
                result = judge(task, code, suite)
                self.assertEqual(result['status'], 'passed', result['reason'])

    def test_original_checks_can_call_the_official_prompt_helper(self):
        task = task_from_archives('HumanEval/32')
        candidate = task['prompt'] + task['canonical_solution']
        result = judge(task, candidate, 'humaneval')
        self.assertEqual(result['status'], 'passed', result['reason'])

    def test_humaneval_plus_accepts_another_valid_polynomial_root(self):
        task = task_from_archives('HumanEval/32')
        task['base_input'] = []
        task['plus_input'] = [[[-6, 11, -6, 1]]]
        result = judge(task, 'def find_zero(xs):\n    return 3.0\n', 'humaneval_plus')
        self.assertEqual(result['status'], 'passed', result['reason'])
