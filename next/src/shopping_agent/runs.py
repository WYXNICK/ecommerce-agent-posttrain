"""Resumable runs: append-only evidence, reproducible derived artifacts, bounded workers."""

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal

from shopping_agent.collection.sft import acceptance_reasons, build_sft_row, write_jsonl
from shopping_agent.evaluation.summary import summarize_trajectories
from shopping_agent.protocol import sha256
from shopping_agent.tasks import read_jsonl
from shopping_agent.training.sft.dataset import build_supervised_example


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class RunStore:
    def __init__(self, directory, specification, *, seed_from=None):
        self.directory = Path(directory)
        self.specification = specification
        self.seed_from = Path(seed_from).resolve() if seed_from is not None else None

    def seed_rows(self):
        receipt_path = self.directory / "source.json"
        if self.seed_from is not None and not receipt_path.exists():
            if (self.directory / "raw.jsonl").exists():
                raise ValueError("cannot add a source after collecting new evidence")
            source = self.seed_from
            if (source / "source.json").exists():
                raise ValueError("source already inherits evidence; resume that output directly")
        elif receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            source = Path(receipt["directory"])
            if self.seed_from is not None and self.seed_from != source:
                raise ValueError("inherited source differs from the frozen receipt")
        else:
            return []
        with (source / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            original = json.loads((source / "run.json").read_text())
            def identity(specification):
                ignored = {"execution_fingerprint", "workers", "environment_url"}
                transport = {"rate_limit_attempts", "rate_limit_initial_delay_seconds", "rate_limit_max_delay_seconds"}
                return {**{key: value for key, value in specification.items() if key not in ignored},
                        "teacher": {key: value for key, value in specification["teacher"].items() if key not in transport}}
            if (self.specification["kind"] != "collect" or original["kind"] != "collect"
                    or identity(original) != identity(self.specification)):
                raise ValueError("source must preserve teacher identity, tasks, role, target and quotas")
            evidence = {"directory": str(source), "run_sha256": sha256(source / "run.json"),
                        "raw_sha256": sha256(source / "raw.jsonl")}
            if receipt_path.exists():
                if json.loads(receipt_path.read_text()) != evidence:
                    raise ValueError("inherited evidence changed")
            else:
                write_json(receipt_path, evidence)
            return read_jsonl(source / "raw.jsonl")

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.directory / "run.json"
            source_prefix_rows = 0
            source_fingerprint = None
            if path.exists():
                if json.loads(path.read_text()) != self.specification:
                    # A reviewed transport repair may resume the same experiment.
                    # Original run/rows stay immutable; new rows carry the new source hash.
                    receipt_path = self.directory.parent / "transport-update.json"
                    if not receipt_path.exists():
                        raise ValueError("run configuration differs; choose a new output directory")
                    receipt = json.loads(receipt_path.read_text())
                    original = json.loads(path.read_text())
                    expected = {**self.specification, "execution_fingerprint": original["execution_fingerprint"]}
                    evidence = receipt["runs"][self.directory.name]
                    if (original != expected or receipt["scope"] != "transport-only"
                            or receipt["from_execution_fingerprint"] != original["execution_fingerprint"]
                            or receipt["to_execution_fingerprint"] != self.specification["execution_fingerprint"]
                            or sha256(path) != evidence["run_sha256"]):
                        raise ValueError("transport repair does not preserve this run's frozen configuration")
                    prefix = evidence["raw_prefix"]
                    with (self.directory / "raw.jsonl").open("rb") as stream:
                        digest = hashlib.sha256(stream.read(prefix["bytes"])).hexdigest()
                    if digest != prefix["sha256"]:
                        raise ValueError("pre-repair raw evidence changed")
                    source_prefix_rows = prefix["rows"]
                    source_fingerprint = original["execution_fingerprint"]
            else:
                if (self.directory / "raw.jsonl").exists():
                    raise ValueError("raw evidence has no run specification")
                write_json(path, self.specification)
            self.rows = read_jsonl(self.directory / "raw.jsonl") if (self.directory / "raw.jsonl").exists() else []
            ids = set(self.specification["candidate_task_ids"])
            if len(self.rows) < source_prefix_rows or any(
                    row["task_id"] not in ids or row["execution_fingerprint"] != (
                        source_fingerprint if index < source_prefix_rows else self.specification["execution_fingerprint"])
                    for index, row in enumerate(self.rows)):
                raise ValueError("raw evidence does not belong to this run")
            inherited = self.seed_rows()
            if any(row["task_id"] not in ids for row in inherited):
                raise ValueError("inherited evidence contains tasks outside this run")
            self.rows = inherited + self.rows
            return self
        except BaseException:
            self.lock.close()
            raise

    def append(self, row):
        with (self.directory / "raw.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)

    def __exit__(self, *_):
        self.lock.close()


@contextmanager
def stop_on_signal():
    """Finish leased episodes and flush evidence after SIGTERM/SIGINT."""
    stop = {"requested": False}
    previous = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, lambda *_: stop.update(requested=True))
    try:
        yield stop
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def usage_summary(rows):
    totals = Counter()
    models = Counter()
    calls_without_usage = 0
    transport_retries = 0
    for row in rows:
        for call in row.get("model_calls", []):
            transport_retries += max(0, call.get("transport_attempts", 1) - 1)
            models[call.get("response_model") or "unknown"] += 1
            usage = call.get("usage")
            if not usage:
                calls_without_usage += 1
                continue
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                totals[key] += usage.get(key, 0)
            totals["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return {"reported_tokens": dict(totals), "calls_without_usage": calls_without_usage,
            "transport_retries_without_usage": transport_retries,
            "response_models": dict(models), "billing_note": "reported usage only; transport retries may incur unreported costs"}


class CollectionProgress:
    def __init__(self, store, *, domains, quotas, tokenizer, max_length):
        self.store, self.domains, self.quotas = store, domains, quotas
        self.tokenizer, self.max_length = tokenizer, max_length
        self.accepted = {}
        self.rejected = []
        for row in store.rows:
            self.consider(row)

    def consider(self, trajectory):
        task_id = trajectory["task_id"]
        _, reasons = acceptance_reasons(trajectory)
        domain = self.domains[task_id]
        if not reasons:
            row = build_sft_row(trajectory)
            example = build_supervised_example(row["messages"], row["tools"], self.tokenizer,
                                               max_length=self.max_length)
            if example is None:
                reasons.append("not_trainable_with_frozen_student_template_or_length")
            elif task_id in self.accepted:
                reasons.append("duplicate_accepted_task")
            elif self.counts()[domain] >= self.quotas[domain]:
                reasons.append("domain_quota_full")
            else:
                row["supervised_tokens"] = sum(token != -100 for token in example["labels"])
                row["sequence_tokens"] = len(example["input_ids"])
                self.accepted[task_id] = row
        if reasons:
            self.rejected.append({"trajectory_id": trajectory["trajectory_id"], "task_id": task_id,
                                  "reasons": reasons})

    def counts(self):
        return Counter(self.domains[task_id] for task_id in self.accepted)

    def publish(self, status):
        counts = self.counts()
        summary = {
            "status": status, "attempts": len(self.store.rows),
            "accepted_unique_tasks": len(self.accepted), "target": sum(self.quotas.values()),
            "domain_quotas": self.quotas, "accepted_by_domain": dict(counts),
            "shortfall_by_domain": {name: quota - counts[name] for name, quota in self.quotas.items()},
            "rejection_counts": dict(Counter(reason for row in self.rejected for reason in row["reasons"])),
            "failure_counts": dict(Counter(row.get("failure_kind") or "strict_success" for row in self.store.rows)),
            "usage": usage_summary(self.store.rows),
        }
        name = "train.jsonl" if self.store.specification["role"] == "train" else "validation.jsonl"
        write_jsonl(self.store.directory / name, self.accepted.values())
        write_jsonl(self.store.directory / "rejected.jsonl", self.rejected)
        write_json(self.store.directory / "summary.json", summary)
        return summary


def collect(store, *, tasks, domains, quotas, tokenizer, config, episode, workers, build_only=False):
    progress = CollectionProgress(store, domains=domains, quotas=quotas, tokenizer=tokenizer,
                                  max_length=config["budgets"]["context_tokens"])
    attempts = Counter(row["task_id"] for row in store.rows if row.get("failure_kind") != "infrastructure")
    maximum = config["collection"]["attempts_per_task"]
    halted = False
    progress.publish("ready")
    with stop_on_signal() as stop, ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        while not build_only:
            counts = progress.counts()
            pending_ids = {task["task_id"] for task in pending.values()}
            inflight = Counter(domains[task_id] for task_id in pending_ids)
            # Try unseen tasks before retrying failed model behavior. One lease per task.
            ordered = sorted(enumerate(tasks), key=lambda item: (attempts[item[1]["task_id"]], item[0]))
            for _, task in ordered:
                if len(pending) >= workers or stop["requested"] or halted:
                    break
                task_id = task["task_id"]
                domain = domains[task_id]
                if (task_id in progress.accepted or task_id in pending_ids or attempts[task_id] >= maximum
                        or counts[domain] + inflight[domain] >= quotas[domain]):
                    continue
                future = executor.submit(episode, task, attempt_index=attempts[task_id])
                pending[future] = task
                pending_ids.add(task_id)
                inflight[domain] += 1
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                task = pending.pop(future)
                row = future.result()
                store.append(row)
                progress.consider(row)
                if row.get("failure_kind") == "infrastructure":
                    halted = True
                else:
                    attempts[task["task_id"]] += 1
                summary = progress.publish("running")
                print(json.dumps({"task_id": row["task_id"], "episode_status": row["status"],
                                  "accepted": summary["accepted_unique_tasks"], "target": summary["target"]}), flush=True)
        complete = len(progress.accepted) == sum(quotas.values())
        status = "complete" if complete else "infrastructure_error" if halted else "interrupted" if stop["requested"] else "incomplete"
        return progress.publish(status)


def evaluate(store, *, tasks, episode, workers):
    latest = {row["task_id"]: row for row in store.rows}
    remaining = iter(task for task in tasks if task["task_id"] not in latest
                     or latest[task["task_id"]].get("failure_kind") == "infrastructure")
    halted = False

    def publish(status):
        summary = summarize_trajectories([task["task_id"] for task in tasks], store.rows)
        unresolved = [task_id for task_id, row in latest.items() if row.get("failure_kind") == "infrastructure"]
        summary.update(status=status, unresolved_infrastructure_tasks=sorted(unresolved),
                       split=store.specification["split"], formal=store.specification["formal"],
                       usage=usage_summary(store.rows))
        write_json(store.directory / "summary.json", summary)
        return summary

    with stop_on_signal() as stop, ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        while True:
            while len(pending) < workers and not halted and not stop["requested"]:
                task = next(remaining, None)
                if task is None:
                    break
                attempt = sum(row["task_id"] == task["task_id"] for row in store.rows)
                pending[executor.submit(episode, task, attempt_index=attempt)] = task
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                pending.pop(future)
                row = future.result()
                store.append(row)
                latest[row["task_id"]] = row
                halted |= row.get("failure_kind") == "infrastructure"
                summary = publish("running")
                print(json.dumps({"task_id": row["task_id"], "episode_status": row["status"],
                                  "completed": summary["completed_tasks"], "expected": len(tasks)}), flush=True)
        complete = len(latest) == len(tasks) and not any(row.get("failure_kind") == "infrastructure" for row in latest.values())
        return publish("complete" if complete else "infrastructure_error" if halted else "interrupted")
