#!/usr/bin/env python3
"""Audit unused reserve tasks; produce candidates, never a certified test split."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import sys

NEXT = Path(__file__).resolve().parents[2]
ROOT = NEXT.parent
sys.path.insert(0, str(NEXT / "scripts"))
from prepare_task_pool import NON_PRODUCT_OPTION, inspect_task, read_jsonl, sha256, write_jsonl
from web_agent_site.engine.comparators import compare_category
from web_agent_site.engine.engine import load_products
from web_agent_site.engine.goal import get_goals
from web_agent_site.engine.search import MultiFieldBM25Searcher
from web_agent_site.engine.variant_price import (
    candidate_options_for_evaluation, matching_candidate_options,
    option_axes, resolve_variant_price,
)

# These entries do not represent ordinary, completely specified retail variants.
ODD_OPTION = re.compile(r"赠品专拍|赠品勿拍|链接勿拍|随机发|随机款|随机色|颜色随机|款式随机|\b0\s*kwh\b", re.I)
APPROXIMATE_BUDGET = re.compile(r"左右|上下|大概|大约|差不多|出头|多一点")


def real_option(entry: dict) -> bool:
    return not (NON_PRODUCT_OPTION.search(entry["value"]) or ODD_OPTION.search(entry["value"]))


def observed_reserve(reserve: set[int], original: dict) -> tuple[set[int], list[dict]]:
    """Read observed episodes and accepted training rows, not planned run ID lists."""
    used, sources = set(), []
    representative = {original[i]["group_id"]: i for i in reserve}
    observed_to_reserve = {i: representative[row["group_id"]] for i, row in original.items()
                           if row["group_id"] in representative}
    names = {"raw.jsonl", "train.jsonl", "validation.jsonl", "accepted.jsonl", "rejected.jsonl"}
    paths = sorted(p for p in (NEXT / "outputs").rglob("*.jsonl") if p.name in names)
    for path in paths:
        ids, observed_ids = set(), set()
        # All of these runtime artifacts carry task_id at the top level.
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("task_id") in observed_to_reserve:
                    observed_ids.add(row["task_id"])
                    ids.add(observed_to_reserve[row["task_id"]])
        used.update(ids)
        sources.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                        "reserve_task_ids": sorted(ids), "observed_task_ids": sorted(observed_ids)})
    return used, sources


def structural_signals(goal: dict, product: dict, config: dict) -> dict:
    axes = option_axes(product)["axes"]
    selected, _ = candidate_options_for_evaluation(product, goal["required_options_by_key"])
    choices = {}
    budget_crossings = []
    upper = goal["price_upper"]
    explicit_budget = upper is not None and not APPROXIMATE_BUDGET.search(goal["instruction_text"])
    for name in goal["required_options_by_key"]:
        axis = axes[name]
        entries = [entry for entry in axis["values"].values() if real_option(entry)]
        choices[name] = len(entries)
        if explicit_budget:
            for entry in entries:
                alternative = {**selected, axis["source_axis"]: entry["value"]}
                price = resolve_variant_price(product, alternative)
                if price["status"] == "pass" and price["price"] > upper:
                    budget_crossings.append({"axis": name, "value": entry["value"],
                                             "price": price["price"]})
    varying = sum(n >= config["minimum_choices_per_axis"] for n in choices.values())
    return {
        "A_multi_axis": varying >= config["minimum_varying_required_axes"],
        "B_budget_coupling": bool(budget_crossings),
        "required_axis_choices": choices, "budget_upper": upper,
        "budget_crossings": budget_crossings,
    }


def retrieval_signal(goal: dict, by_asin: dict, searcher, config: dict) -> dict:
    # Query is exactly the public instruction; no hidden target title/ASIN is added.
    hits = searcher.search(goal["instruction_text"], k=config["search_top_k"])
    gold_rank = next((hit.rank for hit in hits if hit.asin == goal["asin"]), None)
    distractors = []
    for hit in hits:
        if gold_rank is None or hit.rank >= gold_rank:
            continue
        product = by_asin[hit.asin]
        if compare_category(goal["category"], product)["status"] != "pass":
            continue
        _, resolution = matching_candidate_options(product, goal["required_options_by_key"])
        # Missing metadata is UNVERIFIABLE, not evidence of a conflicting specification.
        if resolution["status"] == "fail":
            distractors.append({"asin": hit.asin, "rank": hit.rank, "mismatch": resolution})
    return {"C_retrieval_discrimination": len(distractors) >= config["minimum_preceding_spec_distractors"],
            "gold_rank": gold_rank, "preceding_spec_distractors": distractors}


def audit(config: dict, output: Path) -> dict:
    metadata_path = NEXT / "data/task_pool/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    pool_path = ROOT / config["pool"]
    source = ROOT / metadata["config"]["source"]
    index = ROOT / metadata["config"]["search_index"]
    assert sha256(source) == metadata["source_sha256"], "source changed"
    assert sha256(index) == metadata["search_index_sha256"], "index changed"
    reserve = {r["task_id"] for r in read_jsonl(pool_path)}
    with gzip.open(NEXT / "data/task_pool/private/audit.jsonl.gz", "rt") as stream:
        original = {r["task_id"]: r for r in map(json.loads, stream)}
    assert all(original[i]["assignment"] == "train_reserve" and
               not original[i]["quality_reasons"] and
               not original[i]["historical_group"] for i in reserve)
    assert len({original[i]["group_id"] for i in reserve}) == len(reserve)
    reserve_groups = {original[i]["group_id"] for i in reserve}
    assert not any(r["group_id"] in reserve_groups for r in original.values()
                   if r["assignment"] in {"teacher_candidates", "grpo_train", "sft_validation_candidates", "dev", "test"})
    used, exposure = observed_reserve(reserve, original)
    print(f"Exposure audit: {len(reserve)} reserve, {len(used)} observed", flush=True)
    products, by_asin, prices, _ = load_products(str(source))
    goals = get_goals(products, prices)
    searcher = MultiFieldBM25Searcher(index, expected_product_sha256=metadata["source_sha256"])
    rows = []
    for task_id in sorted(reserve - used):
        goal = goals[task_id]
        product = by_asin[goal["asin"]]
        assert goal["asin"] == original[task_id]["asin"], "runtime task mapping changed"
        quality = inspect_task(task_id, goal, product, searcher.contains_asin(goal["asin"]))
        reasons = list(quality["quality_reasons"])
        gold = quality["gold_check"]
        if gold["price"].get("price", 0) <= 0:
            reasons.append("nonpositive_gold_price")
        if ODD_OPTION.search(product["title"]) or any(
            ODD_OPTION.search(str(v)) for v in gold["selected_options"].values()
        ):
            reasons.append("nonstandard_or_random_target")
        features = structural_signals(goal, product, config) if not reasons else {}
        retrieval = retrieval_signal(goal, by_asin, searcher, config) if not reasons else {}
        signals = [name for name in ("A_multi_axis", "B_budget_coupling", "C_retrieval_discrimination")
                   if features.get(name) or retrieval.get(name)]
        eligible = not reasons and retrieval["gold_rank"] is not None
        candidate = eligible and len(signals) >= config["minimum_difficulty_signals"]
        rows.append({"task_id": task_id, "group_id": original[task_id]["group_id"],
                     "asin": goal["asin"], "domain": quality["domain"],
                     "instruction": goal["instruction_text"], "title": product["title"],
                     "quality_reasons": reasons, "signals": signals, "candidate": candidate,
                     "gold_check": gold, **features, **retrieval})
        if len(rows) % 500 == 0:
            print(f"Audited {len(rows)}/{len(reserve - used)}", flush=True)
    candidates = [row for row in rows if row["candidate"]]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "audit.jsonl", rows)
    (output / "exposure.json").write_text(json.dumps(exposure, ensure_ascii=False, indent=2) + "\n")
    write_jsonl(output / "candidates.jsonl", [{"task_id": row["task_id"], "signals": row["signals"]} for row in candidates])
    summary = {
        "schema": config["version"], "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "status": "offline_candidates_not_certified_test_set",
        "source_sha256": sha256(source), "search_index_sha256": sha256(index),
        "pool_sha256": sha256(pool_path), "script_sha256": sha256(Path(__file__)),
        "reserve_total": len(reserve), "observed_reserve": len(used),
        "observed_reserve_ids": sorted(used), "unobserved_reserve": len(rows),
        "exposure_files_scanned": len(exposure),
        "quality_excluded": sum(bool(row["quality_reasons"]) for row in rows),
        "quality_reason_counts": dict(Counter(r for row in rows for r in row["quality_reasons"])),
        "target_outside_top_k": sum(not row["quality_reasons"] and row["gold_rank"] is None for row in rows),
        "signal_counts": dict(Counter(s for row in rows for s in row["signals"])),
        "candidate_count": len(candidates),
        "candidate_combinations": dict(Counter("+".join(row["signals"]) for row in candidates)),
        "candidate_domains": dict(Counter(row["domain"] for row in candidates)),
        "semantically_and_interactively_certified": 0,
        "files": {name: {"sha256": sha256(output / name), "path": str((output / name).relative_to(ROOT))}
                  for name in ("audit.jsonl", "exposure.json", "candidates.jsonl")},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=NEXT / "configs/hard_candidates.json")
    parser.add_argument("--output", type=Path, default=NEXT / "outputs/evaluation/hard-candidates")
    args = parser.parse_args()
    result = audit(json.loads(args.config.read_text()), args.output)
    print(json.dumps({k: result[k] for k in ("reserve_total", "observed_reserve", "unobserved_reserve", "candidate_count", "candidate_combinations")}, ensure_ascii=False, indent=2))
