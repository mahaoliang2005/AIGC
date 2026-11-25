"""Pinned benchmark tasks, bilingual prompts, and the experiment matrix."""

import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import random
import tokenize


ANSWER_INSTRUCTIONS = (
    "Complete the Python function below. Preserve its signature and any provided "
    "helper code. Return only the complete Python code with all required imports."
)
DEFAULT_SEED = 42
LANGUAGES = ("en", "zh")
REPEATS = 3
DATA_DIR = Path(__file__).with_name("data")
TASK_IDS = tuple(f"HumanEval/{index}" for index in range(164))
TRANSLATIONS_FILE = DATA_DIR / "zh_descriptions.json"
ARCHIVES = {
    "humaneval": (
        "HumanEval.jsonl.gz",
        "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
    ),
    "humaneval_plus": (
        "HumanEvalPlus-v0.1.10.jsonl.gz",
        "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
    ),
}


def dataset_versions():
    return {key: {"archive": name, "sha256": digest} for key, (name, digest) in ARCHIVES.items()}


def _load_archive(name, digest):
    archive = (DATA_DIR / name).read_bytes()
    if sha256(archive).hexdigest() != digest:
        raise ValueError(f"{name}: unexpected SHA-256")
    records = [json.loads(line) for line in gzip.decompress(archive).splitlines() if line.strip()]
    by_id = {record["task_id"]: record for record in records}
    if len(records) != len(TASK_IDS) or set(by_id) != set(TASK_IDS):
        raise ValueError(f"{name}: expected each HumanEval task exactly once")
    return by_id


def load_tasks():
    """Join the two pinned sources by task ID for requests and judges."""
    base = _load_archive(*ARCHIVES["humaneval"])
    plus = _load_archive(*ARCHIVES["humaneval_plus"])
    catalog = {}
    for task_id in TASK_IDS:
        original, extended = base[task_id], plus[task_id]
        if original["entry_point"] != extended["entry_point"]:
            raise ValueError(f"{task_id}: suite entry points do not match")
        catalog[task_id] = {
            "task_id": task_id,
            "entry_point": original["entry_point"],
            "prompt": original["prompt"],
            "test": original["test"],
            "plus_prompt": extended["prompt"],
            "plus_solution": extended["canonical_solution"],
            "base_input": extended["base_input"],
            "plus_input": extended["plus_input"],
            "atol": extended["atol"],
        }
    return catalog


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in Chinese descriptions: {key}")
        result[key] = value
    return result


def load_translations():
    with TRANSLATIONS_FILE.open(encoding="utf-8") as file:
        translations = json.load(file, object_pairs_hook=_unique_json_object)
    if set(translations) - set(TASK_IDS):
        raise ValueError("Chinese descriptions contain unknown task IDs")
    return translations


def _translate_docstring(original, task_id, replacements):
    # The last triple-quoted string is the target function docstring.
    long_strings = [
        token for token in tokenize.generate_tokens(io.StringIO(original).readline)
        if token.type == tokenize.STRING and token.string.startswith(('"""', "'''"))
    ]
    if not long_strings:
        raise ValueError(f"{task_id}: no translatable docstring")
    docstring = long_strings[-1]
    first, opening = docstring.start
    last, closing = docstring.end
    lines = original.splitlines(keepends=True)
    for key, translated in replacements.items():
        number = int(key)
        if not first <= number <= last or not isinstance(translated, str) or "\n" in translated:
            raise ValueError(f"{task_id}: invalid description patch on line {number}")
        original_line = lines[number - 1]
        end = closing - 3 if number == last else len(original_line.rstrip("\r\n"))
        start = opening + 3 if number == first else 0
        content = original_line[start:end]
        if not content.strip() or content.lstrip().startswith(">>>"):
            raise ValueError(f"{task_id}: description patch on blank or example line {number}")
        indent = len(content) - len(content.lstrip())
        lines[number - 1] = original_line[:start + indent] + translated + original_line[end:]
    return "".join(lines)


def build_prompts(task, translations=None):
    """Construct requests without exposing either suite's grading data."""
    if translations is None:
        translations = load_translations()
    task_id = task["task_id"]
    if not translations.get(task_id):
        raise ValueError(f"{task_id}: Chinese description is missing")
    english = task["prompt"]
    chinese = _translate_docstring(english, task_id, translations[task_id])
    return {
        "en": f"{ANSWER_INSTRUCTIONS}\n\n{english}",
        "zh": f"{ANSWER_INSTRUCTIONS}\n\n{chinese}",
    }


def plan(phase, model_ids, seed=DEFAULT_SEED):
    """Return the intended generation rows for a selected model roster."""
    models = tuple(model_ids)
    if not models or len(models) != len(set(models)):
        raise ValueError("A nonempty roster of distinct model IDs is required")
    if phase == "formal":
        selected = TASK_IDS
    elif phase in ("pilot", "extra-pilot"):
        sample = random.Random(seed).sample(TASK_IDS, 16)
        selected = sample[:8] if phase == "pilot" else sample[8:]
    else:
        raise ValueError(f"Unknown experiment phase: {phase}")
    return [
        {"stage": phase, "task_id": task, "model": model, "prompt_version": version, "repeat": repeat}
        for task in selected
        for model in models
        for version in LANGUAGES
        for repeat in range(1, REPEATS + 1)
    ]
