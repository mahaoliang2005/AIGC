from argparse import ArgumentParser
import ast
import json
from pathlib import Path
import re

from .container import JudgeError, judge


ANSWER_INSTRUCTIONS = (
    "Complete the Python function below. Preserve its signature and any provided "
    "helper code. Return only the complete Python code with all required imports."
)
ENGLISH_DESCRIPTION = (
    "Check if in given list of numbers, are any two numbers closer to each other than\n"
    "    given threshold."
)
CHINESE_DESCRIPTION = "判断给定的数字列表中是否存在两个数字，其距离严格小于给定阈值。"
TASK_FILE = Path(__file__).with_name("data") / "humaneval_0.json"
FIXED_HEADER = "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
ADJACENT_GAPS = (
    "    ordered = sorted(numbers)\n"
    "    return any(b - a {comparison} threshold for a, b in zip(ordered, ordered[1:]))\n"
)
FIXED_REPLIES = (
    ("correct", "```python\n" + FIXED_HEADER + ADJACENT_GAPS.format(comparison="<") + "```\n"),
    ("incorrect", FIXED_HEADER + "    return False\n"),
    ("base_only", "```python\n" + FIXED_HEADER + ADJACENT_GAPS.format(comparison="<=") + "```\n"),
)
PYTHON_FENCE = re.compile(r"^[ \t]*```(?:python|py)?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*\r?$", re.M | re.S | re.I)


def build_prompts(original):
    if original.count(ENGLISH_DESCRIPTION) != 1:
        raise ValueError("HumanEval/0 no longer matches the reviewed translation")
    chinese = original.replace(ENGLISH_DESCRIPTION, CHINESE_DESCRIPTION, 1)
    return {
        "en": f"{ANSWER_INSTRUCTIONS}\n\n{original}",
        "zh": f"{ANSWER_INSTRUCTIONS}\n\n{chinese}",
    }


def extract_code(raw_reply, entry_point):
    blocks = PYTHON_FENCE.findall(raw_reply)
    if len(blocks) > 1:
        raise ValueError("reply contains more than one Python code block")
    code = (blocks[0] if blocks else raw_reply).strip() + "\n"
    tree = ast.parse(code)
    if not any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body):
        raise ValueError(f"reply must define {entry_point} at module scope")
    return code


def score_reply(task, raw_reply):
    try:
        code = extract_code(raw_reply, task["entry_point"])
    except (SyntaxError, ValueError) as error:
        result = {"status": "failed", "reason": f"code extraction: {error}"}
        return None, {suite: result for suite in ("humaneval", "humaneval_plus")}
    return code, {suite: judge(task, code, suite) for suite in ("humaneval", "humaneval_plus")}


def run_demo():
    task = json.loads(TASK_FILE.read_text(encoding="utf-8"))
    for version, prompt in build_prompts(task["prompt"]).items():
        for repeat, (fixture, raw_reply) in enumerate(FIXED_REPLIES, start=1):
            code, scores = score_reply(task, raw_reply)
            yield {
                "stage": "demo",
                "task_id": task["task_id"],
                "model": "fixed-replies",
                "prompt_version": version,
                "repeat": repeat,
                "fixture": fixture,
                "prompt": prompt,
                "raw_reply": raw_reply,
                "code": code,
                "scores": scores,
            }


def main():
    parser = ArgumentParser(description="Evaluate fixed replies to one HumanEval task in English and Chinese.")
    parser.add_argument("--output", required=True, type=Path, help="JSONL file for the six generation records")
    args = parser.parse_args()
    try:
        records = list(run_demo())
    except JudgeError as error:
        parser.exit(1, f"Evaluation infrastructure error: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    for record in records:
        original = record["scores"]["humaneval"]["status"]
        extended = record["scores"]["humaneval_plus"]["status"]
        print(f"{record['prompt_version']} {record['fixture']:9} HumanEval={original} HumanEval+={extended}")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
