"""Durable storage for one experiment directory."""

from contextlib import contextmanager
import fcntl
import json
import os
from tempfile import NamedTemporaryFile


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def atomic_text(path, text):
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}-", delete=False,
    ) as output:
        staged = output.name
        try:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
            os.replace(staged, path)
        finally:
            if os.path.exists(staged):
                os.unlink(staged)


def load_jsonl(path, *, repair_tail=False):
    if not path.exists():
        return []
    mode = "r+b" if repair_tail else "rb"
    rows = []
    with path.open(mode) as stream:
        line_number = 0
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                return rows
            line_number += 1
            terminated = line.endswith(b"\n")
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if terminated or not repair_tail:
                    kind = "complete" if terminated else "interrupted"
                    raise ValueError(f"corrupt {kind} journal line {line_number} in {path.name}") from error
                stream.truncate(start)
                sync(stream)
                return rows
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON journal object on line {line_number} in {path.name}")
            rows.append(row)
            if not terminated:
                if repair_tail:
                    stream.write(b"\n")
                    sync(stream)
                return rows


def sync(stream):
    stream.flush()
    os.fsync(stream.fileno())


def append_record(path, record):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        sync(output)
