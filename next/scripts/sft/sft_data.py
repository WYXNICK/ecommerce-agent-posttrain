"""Prepare the selected frozen teacher data with the collection-time student template."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

NEXT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NEXT / "src"))
from shopping_agent.protocol import StudentTokenizer, load_config, sha256, validate_execution
from shopping_agent.runs import write_json
from shopping_agent.tasks import TaskPool
from shopping_agent.training.sft.dataset import build_supervised_example


def selected_data():
    selection = json.loads((NEXT / "configs/data_selection.json").read_text())
    path = NEXT / selection["primary_dataset_manifest"]
    manifest = json.loads(path.read_text())
    config = load_config()
    if manifest["status"] != "complete" or manifest["teacher"] != config["teacher"]["expected_response_model"]:
        raise ValueError("selected data must be complete and match the official teacher")
    rows = {}
    pool = TaskPool()
    for role in ("train", "validation"):
        name = f"{role}.jsonl.gz"
        source, expected = path.parent / name, manifest["files"][name]
        if sha256(source) != expected["sha256"]:
            raise ValueError(f"compressed teacher data changed: {role}")
        data = gzip.decompress(source.read_bytes())
        if hashlib.sha256(data).hexdigest() != expected["uncompressed_sha256"]:
            raise ValueError(f"teacher data content changed: {role}")
        values = [json.loads(line) for line in data.splitlines()]
        ids = {row["task_id"] for row in values}
        if len(values) != expected["rows"] or len(ids) != len(values) or ids & pool.protected_ids(role):
            raise ValueError(f"wrong count, duplicate or held-out task in {role}")
        rows[role] = values
    if {r["task_id"] for r in rows["train"]} & {r["task_id"] for r in rows["validation"]}:
        raise ValueError("training and validation tasks overlap")
    return rows, path, manifest


def prepare(model_path, output):
    import torch

    fingerprint = validate_execution()
    config = load_config()
    rows, manifest_path, manifest = selected_data()
    tokenizer = StudentTokenizer.load(config, path=model_path)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "complete", "execution_fingerprint": fingerprint,
              "dataset_manifest_sha256": sha256(manifest_path), "student": config["student"],
              "teacher": manifest["teacher"], "roles": {}}
    for role, values in rows.items():
        examples = []
        for index, row in enumerate(values):
            example = build_supervised_example(row["messages"], row["tools"], tokenizer,
                                               max_length=config["budgets"]["context_tokens"])
            if (example is None or len(example["input_ids"]) != row["sequence_tokens"]
                    or sum(t != -100 for t in example["labels"]) != row["supervised_tokens"]
                    or example["labels"][0] != -100):
                raise ValueError(f"collection-time template/supervision changed: task {row['task_id']}")
            examples.append({key: torch.tensor(value, dtype=torch.long) for key, value in example.items()})
            if (index + 1) % 100 == 0:
                print(f"prepare {role}: {index + 1}/{len(values)}", flush=True)
        path = output / f"{role}.pt"
        torch.save(examples, path)
        report["roles"][role] = {"examples": len(examples), "sha256": sha256(path),
            "task_ids": [r["task_id"] for r in values],
            "max_sequence_tokens": max(r["sequence_tokens"] for r in values),
            "supervised_tokens": sum(r["supervised_tokens"] for r in values)}
    write_json(output / "prepared.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.output)


if __name__ == "__main__":
    main()
