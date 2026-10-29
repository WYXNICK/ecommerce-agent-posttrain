"""Explicit collection/evaluation entry points; no implicit training or test runs."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
from urllib.request import urlopen

from shopping_agent.clients import LocalChatClient, ModelHubClient
from shopping_agent.protocol import ROOT, StudentTokenizer, freeze_execution, load_config, sha256, teacher_profiles, validate_execution
from shopping_agent.runs import RunStore, collect, evaluate
from shopping_agent.runtime import run_episode
from shopping_agent.tasks import TaskPool, domain_quotas, read_jsonl


def selected_tasks(pool, split, subset):
    tasks = pool.tasks(split)
    if subset:
        requested = [row["task_id"] for row in read_jsonl(Path(subset))]
        if len(requested) != len(set(requested)) or not set(requested) <= {row["task_id"] for row in tasks}:
            raise ValueError("subset must contain unique tasks from the selected frozen split")
        tasks = [{"task_id": task_id} for task_id in requested]
    if not tasks:
        raise ValueError("task selection is empty")
    return tasks


def checkpoint_manifest(directory):
    directory = Path(directory).resolve()
    files = sorted(path for path in directory.rglob("*") if path.is_file()
                   and (path.suffix in {".json", ".safetensors", ".jinja"} or path.name == "merges.txt"))
    if not (directory / "config.json").is_file() or not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("checkpoint must contain config.json and exported safetensors weights")
    return {str(path.relative_to(directory)): sha256(path) for path in files}


def verify_served_checkpoint(base_url, served_name, checkpoint):
    with urlopen(base_url.rstrip("/") + "/models", timeout=30) as response:
        cards = json.load(response)["data"]
    card = next((card for card in cards if card["id"] == served_name), None)
    if card is None or Path(card.get("root", "")).resolve() != Path(checkpoint).resolve():
        raise ValueError("served model root does not match --checkpoint; restart serving with this exported checkpoint")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze", help="review then freeze source/configuration hashes")
    commands.add_parser("check", help="verify frozen code/data and student tokenizer; no model API calls")
    for name in ("collect", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--environment-url", default="http://127.0.0.1:5701")
        command.add_argument("--workers", type=int, help="override the frozen default worker count")
        command.add_argument("--subset", type=Path, help="explicit subset; evaluation then cannot be formal")
        if name == "collect":
            command.add_argument("--teacher-profile", choices=teacher_profiles())
            command.add_argument("--seed-from", type=Path, help="reuse immutable prior evidence from the same teacher and tasks")
            command.add_argument("--role", choices=("train", "validation"), required=True)
            command.add_argument("--target", type=int)
            command.add_argument("--build-only", action="store_true", help="rebuild artifacts from raw evidence without API calls")
        else:
            command.add_argument("--split", choices=("dev_smoke", "dev", "test"), required=True)
            command.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
            command.add_argument("--served-model", default="shopping-agent")
            command.add_argument("--checkpoint", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "freeze":
        freeze_execution()
        print("Frozen: next/configs/execution_manifest.json")
        return 0
    fingerprint = validate_execution()
    config, pool = load_config(teacher_profile=getattr(args, "teacher_profile", None)), TaskPool()
    pool.validate_source()
    tokenizer = StudentTokenizer.load(config)
    if args.command == "check":
        for split in pool.metadata["splits"]:
            pool.tasks(split)
        pool.artifact("held_out.jsonl")
        pool.artifact("evaluation_holdout.jsonl")
        pool.domains()
        print(json.dumps({"execution_fingerprint": fingerprint, "student": config["student"],
                          "tokenizer_check": tokenizer.text_tokens("工具执行检查"), "status": "passed"}))
        return 0
    if args.workers is None:
        args.workers = config["collection"]["workers"]
    if args.workers < 1:
        raise ValueError("workers must be positive")
    specification = {"execution_fingerprint": fingerprint, "workers": args.workers,
                     "environment_url": args.environment_url, "kind": args.command}
    if args.command == "collect":
        split = "teacher_candidates" if args.role == "train" else "sft_validation_candidates"
        tasks = selected_tasks(pool, split, args.subset)
        if {row["task_id"] for row in tasks} & pool.protected_ids(args.role):
            raise ValueError("collection contains protected held-out tasks")
        default = "training_target" if args.role == "train" else "validation_target"
        target = args.target if args.target is not None else config["collection"][default]
        domains = pool.domains()
        quotas = domain_quotas(tasks, domains, target)
        specification.update(role=args.role, split=split, target=target, domain_quotas=quotas,
                             teacher=config["teacher"])
        client = None if args.build_only else ModelHubClient(config["teacher"])
    else:
        tasks = selected_tasks(pool, args.split, args.subset)
        weights = checkpoint_manifest(args.checkpoint)
        verify_served_checkpoint(args.model_url, args.served_model, args.checkpoint)
        specification.update(split=args.split, formal=args.split == "test" and args.subset is None,
                             checkpoint_files=weights, checkpoint=str(args.checkpoint.resolve()),
                             model_url=args.model_url, served_model=args.served_model)
        client = LocalChatClient(args.model_url, args.served_model, api_key=os.environ.get("LOCAL_MODEL_API_KEY", "EMPTY"))
    specification["candidate_task_ids"] = [task["task_id"] for task in tasks]
    episode = partial(run_episode, client=client, config=config, tokenizer=tokenizer,
                      base_url=args.environment_url, fingerprint=fingerprint)
    with RunStore(args.output, specification, seed_from=getattr(args, "seed_from", None)) as store:
        if args.command == "collect":
            summary = collect(store, tasks=tasks, domains=domains, quotas=quotas, tokenizer=tokenizer,
                              config=config, episode=episode, workers=args.workers, build_only=args.build_only)
        else:
            summary = evaluate(store, tasks=tasks, episode=episode, workers=args.workers)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 2
