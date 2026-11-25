import io
import tokenize
import unittest

from prompt_language_comparison.tasks import build_prompts, load_tasks, plan


class TaskPlanTest(unittest.TestCase):
    def test_reproducible_cohorts_and_complete_run_matrix(self):
        first = plan('pilot', ['local'])
        self.assertEqual(first, plan('pilot', ['local']))
        self.assertEqual(len(first), 48)
        self.assertEqual(len({tuple(row.values()) for row in first}), 48)
        self.assertEqual(
            list(dict.fromkeys(row['task_id'] for row in first)),
            [f'HumanEval/{n}' for n in (163, 28, 6, 70, 62, 57, 35, 26)],
        )
        second = plan('extra-pilot', ['local'])
        self.assertEqual(len(second), 48)
        self.assertEqual({row['task_id'] for row in first} & {row['task_id'] for row in second}, set())
        formal = plan('formal', ['local'])
        self.assertEqual(len(formal), 984)
        self.assertEqual({row['task_id'] for row in formal}, {f'HumanEval/{i}' for i in range(164)})
        self.assertTrue(all(row['stage'] == 'formal' for row in formal))

    def test_bilingual_requests_preserve_original_examples_without_tests(self):
        task = load_tasks()['HumanEval/6']
        prompts = build_prompts(task)
        self.assertTrue(prompts['en'].endswith(task['prompt']))
        self.assertEqual(prompts['en'].split('"""')[0], prompts['zh'].split('"""')[0])
        self.assertEqual(prompts['en'].split('>>>')[1], prompts['zh'].split('>>>')[1])
        self.assertIn('最深的嵌套', prompts['zh'])
        self.assertNotIn('assert candidate', prompts['zh'])
        self.assertNotIn('assert candidate', prompts['en'])


    def test_every_task_has_a_translation_that_only_changes_its_docstring(self):
        for task_id, task in load_tasks().items():
            with self.subTest(task_id=task_id):
                prompts = build_prompts(task)
                en, zh = prompts['en'].split('\n\n', 1)[1], prompts['zh'].split('\n\n', 1)[1]
                english_tokens = self._tokens(en)
                chinese_tokens = self._tokens(zh)
                target = max(i for i, token in enumerate(english_tokens) if token[0] == tokenize.STRING)
                original_docstring, translated_docstring = english_tokens[target][1], chinese_tokens[target][1]
                self.assertEqual(english_tokens[:target], chinese_tokens[:target])
                self.assertEqual(english_tokens[target + 1:], chinese_tokens[target + 1:])
                self.assertRegex(translated_docstring, '[一-鿿]')
                for example in (line for line in original_docstring.splitlines() if line.lstrip().startswith('>>>')):
                    self.assertIn(example, translated_docstring.splitlines())

    @staticmethod
    def _tokens(source):
        return [(token.type, token.string) for token in tokenize.generate_tokens(io.StringIO(source).readline)]
