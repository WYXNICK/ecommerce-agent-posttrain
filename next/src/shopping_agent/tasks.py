"""Frozen split membership and proportional collection quotas, never gold actor input."""

from collections import Counter
from fractions import Fraction
import gzip
import json

from shopping_agent.protocol import ROOT, REFERENCE_ROOT, sha256


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


class TaskPool:
    def __init__(self):
        self.directory = ROOT / "data/task_pool"
        self.metadata = json.loads((self.directory / "metadata.json").read_text())

    def tasks(self, split):
        if split not in self.metadata["splits"]:
            raise ValueError(f"unknown task split: {split}")
        rows = self.artifact(f"{split}.jsonl")
        ids = [row["task_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate tasks in {split}")
        return rows

    def artifact(self, name):
        path = self.directory / name
        if sha256(path) != self.metadata["files"][name]["sha256"]:
            raise ValueError(f"frozen task artifact changed: {name}")
        return read_jsonl(path)

    def protected_ids(self, role):
        name = "held_out.jsonl" if role == "train" else "evaluation_holdout.jsonl"
        return ({row["task_id"] for row in self.artifact(name)}
                | {row["task_id"] for row in read_jsonl(REFERENCE_ROOT / "data/evaluation/tasks.jsonl")})

    def domains(self):
        path = self.directory / "private/audit.jsonl.gz"
        if sha256(path) != self.metadata["files"]["private/audit.jsonl.gz"]["sha256"]:
            raise ValueError("private task audit changed")
        with gzip.open(path, "rt") as stream:
            return {row["task_id"]: row["domain"] for row in map(json.loads, stream)}

    def validate_source(self):
        for key, hash_key in (("source", "source_sha256"), ("search_index", "search_index_sha256")):
            if sha256(REFERENCE_ROOT / self.metadata["config"][key]) != self.metadata[hash_key]:
                raise ValueError(f"environment {key} does not match frozen task pool")


def domain_quotas(tasks, domains, target):
    counts = Counter(domains[row["task_id"]] for row in tasks)
    total = sum(counts.values())
    if target <= 0 or target > total:
        raise ValueError("target must be positive and cannot exceed unique candidate tasks")
    exact = {name: Fraction(target * count, total) for name, count in counts.items()}
    quotas = {name: value.numerator // value.denominator for name, value in exact.items()}
    order = sorted(exact, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:target - sum(quotas.values())]:
        quotas[name] += 1
    return quotas
