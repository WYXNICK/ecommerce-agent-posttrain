#!/usr/bin/env python3
"""Evaluate a frozen API teacher on a complete frozen development or test split."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

NEXT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NEXT / "src"))

from shopping_agent.clients import ModelHubClient
from shopping_agent.protocol import StudentTokenizer, load_config, sha256, validate_execution
from shopping_agent.runs import RunStore, evaluate
from shopping_agent.runtime import run_episode
from shopping_agent.tasks import TaskPool


def teacher_config(name):
    return load_config(teacher_profile="sol" if name == "sol" else None)


def entry_files():
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def specification(config, tasks, fingerprint, environment_url, workers, split="test"):
    return {
        "kind": "teacher_evaluate", "split": split, "formal": split == "test",
        "execution_fingerprint": fingerprint, "entry_files": entry_files(),
        "teacher": config["teacher"], "student_budget_tokenizer": config["student"],
        "workers": workers, "environment_url": environment_url,
        "candidate_task_ids": [task["task_id"] for task in tasks],
        "behavior_attempts_per_task": 1,
        "sampling": "gateway default sampling; high reasoning; no behavioral resampling",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("sol", "astra"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--environment-url", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and the eight environment slots")
    fingerprint = validate_execution()
    config, pool = teacher_config(args.teacher), TaskPool()
    pool.validate_source()
    tasks = pool.tasks(args.split)
    tokenizer = StudentTokenizer.load(config)
    client = ModelHubClient(config["teacher"])
    spec = specification(config, tasks, fingerprint, args.environment_url, args.workers, args.split)
    episode = partial(run_episode, client=client, config=config, tokenizer=tokenizer,
                      base_url=args.environment_url, fingerprint=fingerprint)
    with RunStore(args.output, spec) as store:
        summary = evaluate(store, tasks=tasks, episode=episode, workers=args.workers)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
