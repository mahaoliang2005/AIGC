"""Subprocess entry point. This program runs inside the judge container."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import json
import math
import os
from reprlib import repr as short_repr
import signal
import sys


class CandidateTimeout(Exception):
    pass


def time_limit_reached(signum, frame):
    raise CandidateTimeout("code exceeded the suite time limit")


def load_function(code, entry_point, label):
    namespace = {}
    exec(compile(code, label, "exec"), namespace)
    return namespace[entry_point]


def returns_floats(value):
    if isinstance(value, (list, tuple)):
        return bool(value) and all(isinstance(item, float) for item in value)
    return isinstance(value, float)


def within_tolerance(actual, expected, atol):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            within_tolerance(left, right, atol) for left, right in zip(actual, expected)
        )
    return actual == expected or (
        math.isfinite(actual)
        and math.isfinite(expected)
        and abs(actual - expected) <= atol + 1e-7 * abs(expected)
    )


def matches_expected(actual, expected, entry_point, arguments, atol):
    # EvalPlus accepts any root with a sufficiently small residual for HumanEval/32.
    if entry_point == "find_zero":
        coefficients, = arguments
        residual = sum(value * math.pow(actual, i) for i, value in enumerate(coefficients))
        return abs(residual) <= atol
    if actual == expected:
        return True
    return bool(atol) and within_tolerance(actual, expected, atol)


def reference_cases(payload):
    if payload["suite"] == "humaneval":
        return []
    entry_point = payload["entry_point"]
    reference = load_function(payload["reference"], entry_point, "<HumanEval+ reference>")
    case_groups = (
        ("base", payload.get("base_inputs", [])),
        ("additional", payload["additional_inputs"]),
    )
    cases = []
    for group, arguments in case_groups:
        for index, case in enumerate(arguments, start=1):
            expected = None if entry_point == "find_zero" else reference(*deepcopy(case))
            cases.append((f"HumanEval+ {group} case {index}", case, expected))
    return cases


def grade(payload):
    entry_point = payload["entry_point"]
    checks = None
    # The original single-task demo predates the release input fields and uses
    # the equivalent OpenAI checks for its base cases.
    if payload["suite"] == "humaneval" or "base_inputs" not in payload:
        checks = compile(payload["test"], "<HumanEval checks>", "exec")
    cases = reference_cases(payload)
    atol = payload.get("atol", 0)
    signal.signal(signal.SIGALRM, time_limit_reached)
    signal.alarm(3 if payload["suite"] == "humaneval" else 12)
    phase = "loading candidate"
    try:
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            candidate = {}
            exec(compile(payload["code"], "<candidate>", "exec"), candidate)
            solution = candidate[entry_point]
            if checks is not None:
                phase = "HumanEval"
                exec(checks, candidate)
                candidate["check"](solution)
            for phase, case, expected in cases:
                if atol == 0 and returns_floats(expected):
                    atol = 1e-6
                actual = solution(*deepcopy(case))
                if not matches_expected(actual, expected, entry_point, case, atol):
                    if entry_point == "find_zero":
                        reason = f"{phase}: polynomial residual exceeds tolerance"
                    else:
                        reason = f"{phase}: expected {short_repr(expected)}, got {short_repr(actual)}"
                    return {"status": "failed", "reason": reason}
    except AssertionError as error:
        return {"status": "failed", "reason": f"{phase}: {str(error) or 'assertion failed'}"}
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        return {"status": "failed", "reason": f"{phase}: {type(error).__name__}: {str(error)[:240]}"}
    finally:
        signal.alarm(0)
    return {"status": "passed", "reason": None}


if __name__ == "__main__":
    print(json.dumps(grade(json.load(sys.stdin))))
