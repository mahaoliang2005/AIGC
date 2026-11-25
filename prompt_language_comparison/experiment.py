"""Plan, resume and summarize a bilingual run."""

from argparse import ArgumentParser
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlsplit

from . import __main__ as scorer
from . import container
from .__main__ import score_reply
from .backends import BackendError, OllamaBackend, RemoteBackend, _is_loopback
from .container import IMAGE, WORKER, JudgeError
from .report import SUITES, build_report, record_key, write_report
from .run_storage import append_record, atomic_text, load_jsonl, run_lock
from .tasks import build_prompts, dataset_versions, load_tasks, load_translations, plan

DEFAULT_CONFIG = Path(__file__).with_name("data") / "experiment_config.json"
DEFAULT_PILOT_DIRECTORY = Path("results/pilot-local")
DEFAULT_EXTRA_PILOT_DIRECTORY = Path("results/extra-pilot-local")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(serialized).hexdigest()


def make_backends(config):
    enabled = {}
    for item in config["models"]:
        if not item.get("enabled"):
            continue
        model_id = item["id"]
        if model_id in enabled:
            raise ValueError(f"duplicate model ID: {model_id}")
        if item["backend"] == "ollama":
            enabled[model_id] = OllamaBackend(item["model"], **({"base_url": item["base_url"]} if item.get("base_url") else {}))
        elif item["backend"] == "remote":
            enabled[model_id] = RemoteBackend(
                item["model"], item["base_url"], api_key_env=item.get("api_key_env"),
                model_version=item.get("model_version"), options=item.get("options"),
                min_request_interval_seconds=item.get("min_request_interval_seconds", 0),
            )
        else:
            raise ValueError(f"unknown backend: {item['backend']}")
    if not enabled:
        raise ValueError("at least one model must be enabled")
    return enabled


def make_plan(phase, config):
    roster = [item["id"] for item in config["models"] if item.get("enabled")]
    return plan(phase, roster, seed=config.get("seed", 42))


def freeze_manifest(phase, config, backends, tasks):
    rows = make_plan(phase, config)
    task_ids = sorted({row["task_id"] for row in rows})
    translations = load_translations()
    prompts = {tid: build_prompts(tasks[tid], translations) for tid in task_ids}
    manifest = {
        "stage": phase, "seed": config.get("seed", 42), "datasets": dataset_versions(),
        "prompt_sha256": digest(prompts),
        "models": {model_id: backend.describe() for model_id, backend in backends.items()},
        "judge": {
            "image": IMAGE,
            "worker_sha256": sha256(WORKER.read_bytes()).hexdigest(),
            "container_sha256": sha256(Path(container.__file__).read_bytes()).hexdigest(),
            "code_extraction_sha256": sha256(Path(scorer.__file__).read_bytes()).hexdigest(),
        },
        "rows": rows,
    }
    if phase == "formal":
        audit = Path(__file__).with_name("data") / "TRANSLATION_REVIEW.md"
        manifest["translation_review"] = {
            "method": "一位代理撰写，独立代理对照英文逐题复核",
            "audit_sha256": sha256(audit.read_bytes()).hexdigest(),
        }
    return manifest, prompts


def validate_existing(records, allowed):
    keyed = {}
    for row in records:
        key = record_key(row)
        if key in keyed or key not in allowed:
            raise ValueError(f"duplicate or unplanned journal key: {key}")
        keyed[key] = row
    return keyed


def grade(task, reply):
    if reply["raw_reply"] is None:
        return {**reply, "code": None, "scores": {}}
    code, scores = score_reply(task, reply["raw_reply"])
    return {**reply, "code": code, "scores": scores}


def run_experiment(directory, manifest, prompts, tasks, backends, *, workers=3, generation_workers=1):
    if workers < 1 or generation_workers < 1:
        raise ValueError("worker counts must be positive")
    with run_lock(directory):
        return _run_experiment(directory, manifest, prompts, tasks, backends, workers, generation_workers)


def _run_experiment(directory, manifest, prompts, tasks, backends, workers, generation_workers):
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if load_json(manifest_path) != manifest:
            raise ValueError("the frozen manifest differs; use a new run directory")
    else:
        atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    reply_path = directory / "replies.jsonl"
    record_path = directory / "records.jsonl"
    error_path = directory / "judge-errors.jsonl"
    allowed = {record_key(row) for row in manifest["rows"]}
    replies = validate_existing(load_jsonl(reply_path, repair_tail=True), allowed)
    records = validate_existing(load_jsonl(record_path, repair_tail=True), allowed)
    errors = load_jsonl(error_path, repair_tail=True)
    if records.keys() - replies.keys():
        raise ValueError("found graded records without their generation journal")

    pending = {}
    infrastructure_errors = []

    def collect(done):
        for future in done:
            key = pending.pop(future)
            try:
                record = future.result()
            except JudgeError as error:
                detail = {"key": list(key), "reason": str(error)}
                infrastructure_errors.append(detail)
                errors.append(detail)
                append_record(error_path, detail)
                print(f"Judge error for {key}: {error}", flush=True)
                continue
            append_record(record_path, record)
            records[key] = record
            if len(records) % 12 == 0 or len(records) == len(allowed):
                print(f"{manifest['stage']}: {len(records)}/{len(allowed)} records", flush=True)

    def generate(row):
        prompt = prompts[row["task_id"]][row["prompt_version"]]
        seed = manifest["seed"] + row["repeat"] - 1
        response = backends[row["model"]].generate(prompt, seed=seed)
        return {**row, **response, "prompt": prompt}

    rows = (row for row in manifest["rows"] if record_key(row) not in records)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for reply in _scheduled_replies(rows, replies, generate, generation_workers):
            key = record_key(reply)
            if key not in replies:
                append_record(reply_path, reply)
                replies[key] = reply
            pending[pool.submit(grade, tasks[reply["task_id"]], reply)] = key
            if len(pending) >= 2 * workers:
                collect(wait(pending, return_when=FIRST_COMPLETED).done)
        if pending:
            collect(wait(pending).done)

    report = _report_from_journals(manifest, list(replies.values()), list(records.values()), errors)
    report["infrastructure_errors_this_run"] = infrastructure_errors
    write_report(report, directory)
    return report


def _scheduled_replies(rows, saved, generate, workers):
    if workers == 1:
        for row in rows:
            key = record_key(row)
            yield saved[key] if key in saved else generate(row)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        for row in rows:
            key = record_key(row)
            if key in saved:
                yield saved[key]
                continue
            pending.add(pool.submit(generate, row))
            if len(pending) >= workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
        if pending:
            for future in wait(pending).done:
                yield future.result()


def _report_from_journals(manifest, replies, records, errors):
    report = build_report(manifest["rows"], records, replies=replies, seed=manifest["seed"])
    report["translation_review"] = manifest.get("translation_review")
    report["pilot_gate"] = manifest.get("pilot_gate")
    report["judge"] = manifest.get("judge")
    report["infrastructure_errors"] = errors
    return report


def regenerate_report(directory):
    with run_lock(directory):
        manifest = load_json(directory / "manifest.json")
        report = _report_from_journals(
            manifest, load_jsonl(directory / "replies.jsonl"), load_jsonl(directory / "records.jsonl"),
            load_jsonl(directory / "judge-errors.jsonl"),
        )
        write_report(report, directory)
        return report


def pilot_groups(report):
    groups = {}
    for row in report["per_task"]:
        key = (row["model"], row["language"])
        group = groups.setdefault(key, {"model": row["model"], "language": row["language"], "graded": 0, "passed": 0})
        suite = row["suites"]["humaneval_plus"]
        group["graded"] += suite["graded"]
        group["passed"] += suite["passed"]
    return [groups[key] for key in sorted(groups)]


def ceiling_reached(report):
    groups = pilot_groups(report)
    return bool(groups) and all(group["graded"] == 24 and group["passed"] >= 23 for group in groups)


def _comparison_models(models):
    # Two local servers can serve the same installed model and runtime.
    result = {}
    for model_id, settings in models.items():
        settings = dict(settings)
        if settings.get("backend") == "ollama" and _is_loopback(urlsplit(settings.get("base_url", "")).hostname):
            settings.pop("base_url", None)
        result[model_id] = settings
    return result


def _pilot_evidence(directory, phase, formal, prompts):
    with run_lock(directory):
        frozen = load_json(directory / "manifest.json")
        expected = plan(phase, formal["models"], seed=formal["seed"])
        expected_keys = {record_key(row) for row in expected}
        expected_prompts = {row["task_id"]: prompts[row["task_id"]] for row in expected}
        judge = formal["judge"]
        frozen_judge = frozen["judge"]
        incompatible_judge = (
            not {"image", "worker_sha256"} <= frozen_judge.keys()
            or any(judge.get(key) != value for key, value in frozen_judge.items())
        )
        if (
            frozen["stage"] != phase or frozen["seed"] != formal["seed"]
            or _comparison_models(frozen["models"]) != _comparison_models(formal["models"])
            or frozen["datasets"] != formal["datasets"]
            or len(frozen["rows"]) != len(expected)
            or {record_key(row) for row in frozen["rows"]} != expected_keys
            or frozen["prompt_sha256"] != digest(expected_prompts)
            or incompatible_judge
        ):
            raise ValueError(f"{phase} must use the same seed, cohort, prompts, models and judge")
        replies = load_jsonl(directory / "replies.jsonl")
        records = load_jsonl(directory / "records.jsonl")
        known_replies = validate_existing(replies, expected_keys)
        verdicts = validate_existing(records, expected_keys)
        if (
            known_replies.keys() != expected_keys or verdicts.keys() != expected_keys
            or any(
                row.get("raw_reply") is None or row.get("prompt") != prompts[row["task_id"]][row["prompt_version"]]
                for row in known_replies.values()
            )
            or any(
                row.get("raw_reply") != known_replies[key].get("raw_reply")
                or any(row.get("scores", {}).get(suite, {}).get("status") not in ("passed", "failed")
                       for suite in SUITES)
                for key, row in verdicts.items()
            )
        ):
            raise ValueError(f"{phase} must have a reply and both grades for every slot")
        report = build_report(expected, records, replies=replies, seed=formal["seed"])
        groups = pilot_groups(report)
        if len(groups) != 2 * len(formal["models"]) or any(group["graded"] != 24 for group in groups):
            raise ValueError(f"{phase} must have 24 HumanEval+ grades for every model-language group")
        artifacts = {
            name: sha256((directory / name).read_bytes()).hexdigest()
            for name in ("manifest.json", "replies.jsonl", "records.jsonl")
        }
        return {
            "phase": phase, "groups": groups, "ceiling_reached": ceiling_reached(report),
            "sha256": artifacts,
        }


def verify_pilots(formal, prompts, pilot_directory, extra_pilot_directory, decision=None):
    first = _pilot_evidence(pilot_directory, "pilot", formal, prompts)
    evidence = [first]
    if first["ceiling_reached"]:
        second = _pilot_evidence(extra_pilot_directory, "extra-pilot", formal, prompts)
        evidence.append(second)
        if second["ceiling_reached"]:
            if not decision or not decision.strip():
                raise ValueError("both pilots reached the ceiling; record the team's decision with --ceiling-decision")
            return {"evidence": evidence, "decision": decision.strip()}
    if decision is not None:
        raise ValueError("--ceiling-decision applies only when both pilots reached the ceiling")
    return {"evidence": evidence, "decision": None}


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--phase", required=True, choices=("pilot", "extra-pilot", "formal"))
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        sub.add_argument("--directory", type=Path, required=True)
        if name == "run":
            sub.add_argument("--workers", type=int, default=3)
            sub.add_argument("--generation-workers", type=int, default=1)
            sub.add_argument("--pilot-directory", type=Path, default=DEFAULT_PILOT_DIRECTORY)
            sub.add_argument("--extra-pilot-directory", type=Path, default=DEFAULT_EXTRA_PILOT_DIRECTORY)
            sub.add_argument("--ceiling-decision")
    result = commands.add_parser("report")
    result.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "report":
            report = regenerate_report(args.directory)
        else:
            config = load_json(args.config)
            if args.command == "plan":
                rows = make_plan(args.phase, config)
                args.directory.mkdir(parents=True, exist_ok=True)
                atomic_text(args.directory / "plan.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
                print(f"{args.phase}: {len(rows)} planned replies")
                return
            if args.workers < 1 or args.generation_workers < 1:
                raise ValueError("--workers and --generation-workers must be positive")
            if args.phase != "formal" and args.ceiling_decision is not None:
                raise ValueError("--ceiling-decision applies to the formal phase")
            tasks = load_tasks()
            backends = make_backends(config)
            manifest, prompts = freeze_manifest(args.phase, config, backends, tasks)
            if args.phase == "formal":
                manifest["pilot_gate"] = verify_pilots(
                    manifest, prompts, args.pilot_directory, args.extra_pilot_directory, args.ceiling_decision,
                )
            report = run_experiment(
                args.directory, manifest, prompts, tasks, backends,
                workers=args.workers, generation_workers=args.generation_workers,
            )
            if report["infrastructure_errors_this_run"]:
                parser.exit(1, f"Judge errors remain; rerun to reuse stored replies. See {args.directory / 'judge-errors.jsonl'}.\n")
        if report["stage"] in ("pilot", "extra-pilot"):
            print("Ceiling condition triggered." if ceiling_reached(report) else "Ceiling condition not triggered.")
        print(f"Report: {args.directory / 'report.md'}")
    except (BackendError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
