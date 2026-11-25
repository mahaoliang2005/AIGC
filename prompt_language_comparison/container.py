"""Docker boundary for running generated code and its tests."""

import json
from pathlib import Path
import subprocess
from uuid import uuid4


IMAGE = "python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c"
WORKER = Path(__file__).with_name("judge_worker.py")
MEMORY = "512m"


class JudgeError(RuntimeError):
    pass


def judge(task, code, suite):
    if suite not in ("humaneval", "humaneval_plus"):
        raise ValueError(f"unknown test suite: {suite}")
    payload = {"suite": suite, "code": code, "entry_point": task["entry_point"], "test": task["test"]}
    if suite == "humaneval_plus":
        payload["reference"] = task.get("plus_prompt", task["prompt"]) + task["plus_solution"]
        payload["additional_inputs"] = task["plus_input"]
        payload["atol"] = task.get("atol", 0)
        if "base_input" in task:
            payload["base_inputs"] = task["base_input"]

    name = "language-judge-" + uuid4().hex
    command = [
        "docker", "run", "--name", name, "--pull", "never", "--network", "none",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534", "--memory", MEMORY, "--memory-swap", MEMORY,
        "--cpus", "1", "--pids-limit", "32", "-i", IMAGE,
        "python", "-I", "-B", "-c", WORKER.read_text(encoding="utf-8"),
    ]
    try:
        try:
            process = subprocess.run(command, input=json.dumps(payload), capture_output=True, text=True, timeout=30)
        except FileNotFoundError as error:
            raise JudgeError("Docker is required to grade replies") from error
        except subprocess.TimeoutExpired as error:
            raise JudgeError(f"judge container {name} did not report a result") from error
        if process.returncode:
            if process.returncode == 137 and _out_of_memory(name):
                return {"status": "failed", "reason": f"judge container exceeded its {MEMORY} memory limit"}
            raise JudgeError(process.stderr.strip() or f"judge container exited with code {process.returncode}")
    finally:
        _remove_container(name)
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise JudgeError("judge container did not return a JSON verdict") from error
    if not isinstance(result, dict) or result.get("status") not in ("passed", "failed"):
        raise JudgeError("judge container returned an invalid verdict")
    return result


def _out_of_memory(name):
    try:
        state = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.OOMKilled}}", name],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return state.returncode == 0 and state.stdout.strip() == "true"


def _remove_container(name):
    try:
        removed = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        return
    except subprocess.TimeoutExpired as error:
        raise JudgeError(f"judge container {name} cleanup did not finish") from error
    if removed.returncode and "No such container" not in removed.stderr:
        raise JudgeError(removed.stderr.strip() or f"could not remove judge container {name}")
