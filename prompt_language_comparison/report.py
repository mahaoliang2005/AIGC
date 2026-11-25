"""Summarize complete per-task results and estimate paired language differences."""

from collections import Counter
import json
from pathlib import Path
from random import Random
from statistics import mean

from .run_storage import atomic_text

SUITES = ("humaneval", "humaneval_plus")
KEY_FIELDS = ("stage", "task_id", "model", "prompt_version", "repeat")
REPEATS = (1, 2, 3)


def record_key(record):
    return tuple(record[field] for field in KEY_FIELDS)


def _percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _paired_interval(deltas, draws, seed):
    if not deltas:
        return None
    rng = Random(seed)
    boot = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(draws))
    return [_percentile(boot, 0.025), _percentile(boot, 0.975)]


def build_report(manifest, records, *, replies=None, bootstrap_samples=5000, seed=42):
    """Report coverage and scores for the exact run slots in one phase."""
    planned = {record_key(row) for row in manifest}
    if not planned or len(planned) != len(manifest) or len({key[0] for key in planned}) != 1:
        raise ValueError("expected one nonempty phase with unique planned keys")
    groups = sorted({key[:-1] for key in planned})
    if planned != {(*group, repeat) for group in groups for repeat in REPEATS}:
        raise ValueError("the plan must include three repetitions for every group")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    def index(rows):
        counts = Counter(record_key(row) for row in rows)
        duplicates = {key for key, count in counts.items() if count > 1}
        keyed = {record_key(row): row for row in rows if record_key(row) in planned}
        return counts.keys(), duplicates, keyed

    recorded, duplicate_records, verdicts = index(records)
    sent, duplicate_replies, originals = index(replies if replies is not None else records)
    duplicates = duplicate_records | duplicate_replies
    indexed = {}
    inconsistent = set()
    for key, raw in originals.items():
        if key in duplicates:
            continue
        verdict = verdicts.get(key)
        if verdict is not None and verdict.get("raw_reply") != raw.get("raw_reply"):
            inconsistent.add(key)
            verdict = None
        indexed[key] = {**raw, "scores": verdict.get("scores", {}) if verdict else {}}

    coverage = {
        "planned": len(planned),
        "received": sum(row.get("raw_reply") is not None for row in indexed.values()),
        "unique_recorded": len((indexed.keys() & verdicts.keys()) - inconsistent),
        "unique_replies": len(indexed),
        "not_run": [list(key) for key in sorted(planned - sent - recorded)],
        "unexpected_keys": [list(key) for key in sorted((sent | recorded) - planned)],
        "duplicate_keys": [list(key) for key in sorted(duplicates)],
        "orphan_records": [list(key) for key in sorted((recorded - sent) & planned)],
        "inconsistent_records": [list(key) for key in sorted(inconsistent)],
        "missing_replies": [list(key) for key, row in sorted(indexed.items()) if row.get("raw_reply") is None],
        "ungraded": {
            suite: [list(key) for key, row in sorted(indexed.items())
                    if row.get("raw_reply") is not None and not _is_graded(row, suite)]
            for suite in SUITES
        },
    }
    per_task = []
    for stage, task_id, model, language in groups:
        group = (stage, task_id, model, language)
        received = [indexed[key] for i in REPEATS
                    if (key := (*group, i)) in indexed and indexed[key].get("raw_reply") is not None]
        per_task.append({
            "stage": stage, "task_id": task_id, "model": model, "language": language,
            "received": len(received), "missing": len(REPEATS) - len(received),
            "suites": {
                suite: {
                    "graded": sum(_is_graded(row, suite) for row in received),
                    "passed": sum(_verdict(row, suite) == "passed" for row in received),
                }
                for suite in SUITES
            },
        })

    overall, comparison = [], []
    models = sorted({row["model"] for row in per_task})
    expected_count = len(REPEATS)
    for suite in SUITES:
        for model in models:
            by_language = {
                lang: {row["task_id"]: row["suites"][suite]["passed"]
                       for row in per_task if row["model"] == model and row["language"] == lang
                       and row["suites"][suite]["graded"] == expected_count}
                for lang in ("en", "zh")
            }
            for lang, scores in by_language.items():
                overall.append({"suite": suite, "model": model, "language": lang,
                                "complete_tasks": len(scores),
                                "pass_at_1_pct": 100 * mean(scores.values()) / expected_count if scores else None})
            en, zh = by_language["en"], by_language["zh"]
            paired = sorted(en.keys() & zh.keys())
            differences = [(zh[tid] - en[tid]) * 100 / expected_count for tid in paired]
            comparison.append({
                "suite": suite, "model": model, "paired_tasks": len(paired),
                "english_pass_at_1_pct": 100 * mean(en[tid] for tid in paired) / expected_count if paired else None,
                "chinese_pass_at_1_pct": 100 * mean(zh[tid] for tid in paired) / expected_count if paired else None,
                "delta_pp": mean(differences) if paired else None,
                "ci95_pp": _paired_interval(differences, bootstrap_samples, seed),
            })
    return {"stage": groups[0][0], "coverage": coverage, "per_task": per_task,
            "overall": overall, "comparison": comparison,
            "bootstrap": {"resamples": bootstrap_samples, "seed": seed}}


def _verdict(row, suite):
    return row.get("scores", {}).get(suite, {}).get("status")


def _is_graded(row, suite):
    return _verdict(row, suite) in ("passed", "failed")


def _display(value):
    return "—" if value is None else f"{value:.2f}"


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_text(directory / "report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    names = {"humaneval": "HumanEval", "humaneval_plus": "HumanEval+"}
    lines = [f"# {report['stage']} 中英文提示词评测", "",
             f"计划 {report['coverage']['planned']} 次生成，收到 {report['coverage']['received']} 份回复。"]
    for suite in ("humaneval_plus", "humaneval"):
        lines.extend(["", f"## {names[suite]}", "",
                      "| 模型 | 题面 | 题数 | pass@1 (%) |", "| --- | --- | ---: | ---: |"])
        for row in report["overall"]:
            if row["suite"] == suite:
                lines.append(f"| {row['model']} | {row['language']} | {row['complete_tasks']} | {_display(row['pass_at_1_pct'])} |")
        lines.extend(["", "| 模型 | 配对题数 | 中文 − 英文（pp） | 95% 区间（pp） |", "| --- | ---: | ---: | --- |"])
        for row in report["comparison"]:
            if row["suite"] == suite:
                interval = "—" if row["ci95_pp"] is None else " 到 ".join(map(_display, row["ci95_pp"]))
                lines.append(f"| {row['model']} | {row['paired_tasks']} | {_display(row['delta_pp'])} | {interval} |")
    review = report.get("translation_review")
    if review:
        lines.extend(["", f"题面审校方式为{review['method']}。"])
    errors = report.get("infrastructure_errors", [])
    if errors:
        still_ungraded = {tuple(key) for keys in report["coverage"]["ungraded"].values() for key in keys}
        outstanding = still_ungraded.intersection(tuple(item["key"]) for item in errors)
        lines.extend(["", f"记录了 {len(errors)} 次评测设施错误，其中 {len(outstanding)} 个位置仍待判分。"])
    gate = report.get("pilot_gate")
    if gate:
        lines.extend(["", "正式运行前核对的试跑：", "",
                      "| 批次 | 模型 | 题面 | HumanEval+ 通过 / 判分 |", "| --- | --- | --- | ---: |"])
        for evidence in gate["evidence"]:
            for group in evidence["groups"]:
                lines.append(
                    f"| {evidence['phase']} | {group['model']} | {group['language']} | "
                    f"{group['passed']} / {group['graded']} |"
                )
        if gate["decision"]:
            lines.extend(["", f"两批达到满分阈值后的决定：{gate['decision']}"])
    lines.extend([
        "", "评测使用项目自己的固定时间和内存限制。HumanEval 套件限时 3 秒，",
        "HumanEval+ 套件限时 12 秒。官方 EvalPlus 的超时算法不同，性能相关失败可能有所差异。",
        "", "HumanEval+ v0.1.10 的 HumanEval/32 自带参考实现有一条新增用例",
        "未达到官方多项式残差阈值。本次评测保留该用例与原阈值。",
    ])
    atomic_text(directory / "report.md", "\n".join(lines) + "\n")
