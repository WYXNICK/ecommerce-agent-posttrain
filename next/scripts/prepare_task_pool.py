#!/usr/bin/env python3
"""Freeze fresh ShopSimulator task splits using offline Reward v3 validation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata

IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
# Original environment and large inputs remain read-only in the repository root.
ROOT = IMPLEMENTATION_ROOT.parent
SHOP_ENV = ROOT / "environments/ShopSimulator/shop_env"
sys.path.insert(0, str(SHOP_ENV))

from web_agent_site.engine.reward import evaluate_purchase  # noqa: E402
from web_agent_site.engine.reward_features import canonicalize_option_axis  # noqa: E402
from web_agent_site.engine.variant_price import (  # noqa: E402
    candidate_options_for_evaluation,
    resolve_variant_price,
)

SCHEMA = "shopping-task-pool-v1"
PRICE_HINT = re.compile(
    r"(?:预算|价格|售价|价钱|价位|总价|费用|成本|花费|多少钱|"
    r"[零一二三四五六七八九十百千万两\d]+(?:\.\d+)?\s*(?:元|块|钱)|"
    r"\d+(?:\.\d+)?\s*(?:左右|上下|出头))"
)
COLOR = re.compile(r"([黑白灰银金红蓝绿棕粉紫黄橙米咖褐](?:[、/和及与]?[黑白灰银金红蓝绿棕粉紫黄橙米咖褐])*)(?:色|拼色|撞色)")
NON_PRODUCT_OPTION = re.compile(r"咨询客服|联系客服|咨询.*报价|定金|订金|补差价|补差专用|运费专拍|勿拍|测试选项|盲盒随机")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in text if char.isalnum())


def color_families(text: str) -> set[str]:
    # Relative/negative descriptions need semantic review rather than a literal gate.
    if re.search(r"相反|除了|(?:不要|不选|不喜欢|不是|非)[^，。；,;]{0,3}[黑白灰银金红蓝绿棕粉紫黄橙米咖褐]", text):
        return set()
    text = text.replace("粉红色", "粉色").replace("咖啡色", "咖色")
    colors = {char for match in COLOR.finditer(text) for char in match.group(1)
              if char not in "、/和及与"}
    return {"棕" if color in "咖褐" else color for color in colors}


def trigrams(text: str) -> set[str]:
    text = normalized_text(text)
    return {text[i : i + 3] for i in range(max(len(text) - 2, 1))} if text else set()


def similar_pairs(texts: list[str], threshold: float):
    """Exact Jaccard join using rare-first prefix filtering, not an LSH estimate."""
    ratio = Fraction(str(threshold))
    if not 0 < ratio <= 1:
        raise ValueError("similarity threshold must be in (0, 1]")
    numerator, denominator = ratio.numerator, ratio.denominator
    tokens = [trigrams(text) for text in texts]
    frequency = Counter(token for row in tokens for token in row)
    postings = defaultdict(list)
    for current in sorted(range(len(tokens)), key=lambda i: (len(tokens[i]), i)):
        row = tokens[current]
        if not row:
            continue
        prefix_size = len(row) - (numerator * len(row) + denominator - 1) // denominator + 1
        prefix = sorted(row, key=lambda token: (frequency[token], token))[:prefix_size]
        candidates = {other for token in prefix for other in postings[token]}
        for other in sorted(candidates):
            previous = tokens[other]
            if len(previous) * denominator < len(row) * numerator:
                continue
            intersection = len(row & previous)
            union = len(row) + len(previous) - intersection
            if intersection * denominator >= union * numerator:
                yield other, current
        for token in prefix:
            postings[token].append(current)


def duplicate_groups(records: list[dict], config: dict) -> tuple[list[int], dict]:
    """Connect same products, instruction templates and near duplicates transitively."""
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)

    seen = {}
    counts = Counter()
    for i, record in enumerate(records):
        keys = (
            ("asin", record["asin"]),
            ("title", normalized_text(record["title"])),
            ("instruction_template", re.sub(r"\d+", "#", normalized_text(record["instruction"]))),
        )
        for kind, value in keys:
            if not value:
                continue
            if (kind, value) in seen:
                join(i, seen[kind, value])
                counts[kind] += 1
            else:
                seen[kind, value] = i
    for field, threshold in (
        ("instruction", config["instruction_similarity"]),
        ("title", config["title_similarity"]),
    ):
        for left, right in similar_pairs([r[field] for r in records], threshold):
            join(left, right)
            counts[f"near_{field}_pairs"] += 1
    return [find(i) for i in range(len(records))], dict(counts)


def task_ids_in(value) -> set[int]:
    """Read historical IDs from task lists, nested trajectories and result metadata."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "task_id" and isinstance(item, int) and not isinstance(item, bool):
                found.add(item)
            elif (key == "task_ids" or key.endswith("_task_ids")) and isinstance(item, list):
                found.update(i for i in item if isinstance(i, int) and not isinstance(i, bool))
            elif isinstance(item, (dict, list)):
                found.update(task_ids_in(item))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                found.update(task_ids_in(item))
    return found


def historical_inventory() -> tuple[list[dict], dict]:
    origins = defaultdict(list)
    sources = {}
    paths = set()
    for directory in ("data", "experiments", "outputs", "docs/reproduction", "examples"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix not in {".json", ".jsonl"} or not path.is_file():
                continue
            if any(part in {"task_pool", "models", "checkpoints"} for part in path.parts):
                continue
            paths.add(path)
    for path in sorted(paths):
        ids = set()
        with path.open(encoding="utf-8") as stream:
            if path.suffix == ".jsonl":
                for line in stream:
                    if line.strip():
                        ids.update(task_ids_in(json.loads(line)))
            else:
                ids = task_ids_in(json.load(stream))
        if ids:
            name = str(path.relative_to(ROOT))
            sources[name] = {"sha256": sha256(path), "task_count": len(ids)}
            for task_id in ids:
                origins[task_id].append(name)
    return [
        {"task_id": task_id, "sources": paths} for task_id, paths in sorted(origins.items())
    ], sources


def inspect_task(task_id: int, goal: dict, product: dict, indexed: bool) -> dict:
    selected, option_resolution = candidate_options_for_evaluation(
        product, goal["required_options_by_key"]
    )
    price = resolve_variant_price(product, selected)
    reward = evaluate_purchase(product, goal, selected_options=selected, price_resolution=price)
    reasons = []
    if not goal["instruction_text"].strip():
        reasons.append("empty_instruction")
    if not indexed:
        reasons.append("target_not_indexed")
    if goal["unresolved_option_requirements"] or option_resolution["status"] != "pass":
        reasons.append("unresolved_options")
    if price["status"] != "pass":
        reasons.append("price_unverifiable")
    if PRICE_HINT.search(goal["instruction_text"]) and goal["price_upper"] is None:
        reasons.append("unparsed_price_hint")
    if reward.reward_type != "gold_purchase" or not reward.reward_valid:
        reasons.append("target_not_valid_gold")
    query_colors = color_families(goal["instruction_text"])
    for axis, requirement in goal["required_options_by_key"].items():
        if canonicalize_option_axis(axis) == "color":
            option_colors = color_families(requirement["value"])
            if query_colors and option_colors and not query_colors & option_colors:
                reasons.append("query_option_color_conflict")
    if any(NON_PRODUCT_OPTION.search(str(value)) for value in selected.values()):
        reasons.append("non_product_variant")
    if any(re.search(r"^0\s*(?:kwh|wh|mah|ah)$", str(value).casefold())
           for value in selected.values()):
        reasons.append("placeholder_capacity_variant")
    if "??" in goal["instruction_text"] or "\ufffd" in goal["instruction_text"]:
        reasons.append("corrupted_instruction_text")
    constraints = (
        len(goal["expected_core_functions"]) + len(goal["required_options_by_key"])
        + len(goal["expected_brand"]) + len(goal["expected_model"])
        + int(goal["price_upper"] is not None)
    )
    return {
        "task_id": task_id,
        "asin": goal["asin"],
        "title": product["title"],
        "instruction": goal["instruction_text"],
        "domain": product["domain_en_short"],
        "category": goal["category"],
        "constraint_count": constraints,
        "constraint_band": "low" if constraints <= 4 else "medium" if constraints <= 8 else "high",
        "budget_required": goal["price_upper"] is not None,
        "required_option_axes": len(goal["required_options_by_key"]),
        "option_axes": len(product.get("customization_options") or {}),
        "quality_reasons": reasons,
        "gold_check": {"selected_options": selected, "price": price,
                       "reward_type": reward.reward_type, "reward_valid": reward.reward_valid},
    }


def stratum(record: dict) -> tuple:
    return (record["domain"], record["constraint_band"], record["budget_required"],
            bool(record["required_option_axes"]))


def stratified_sample(records: list[dict], count: int, seed: str) -> list[dict]:
    """Use proportional largest-remainder quotas and a stable seeded hash order."""
    if not 0 <= count <= len(records):
        raise ValueError(f"requested {count} tasks from only {len(records)} candidates")
    if not records:
        return []
    buckets = defaultdict(list)
    for record in records:
        buckets[stratum(record)].append(record)
    quotas = {key: count * len(rows) // len(records) for key, rows in buckets.items()}
    remaining = count - sum(quotas.values())
    order = sorted(buckets, key=lambda k: (-(count * len(buckets[k]) % len(records)), k))
    for key in order[:remaining]:
        quotas[key] += 1
    selected = []
    for key, rows in sorted(buckets.items()):
        ranked = sorted(rows, key=lambda r: hashlib.sha256(
            f"{seed}:{r['task_id']}".encode()
        ).digest())
        selected.extend(ranked[:quotas[key]])
    return sorted(selected, key=lambda r: hashlib.sha256(
        f"{seed}:order:{r['task_id']}".encode()
    ).digest())


def build_splits(records: list[dict], groups: list[int], history_ids: set[int], config: dict):
    contaminated = {groups[i] for i, r in enumerate(records) if r["task_id"] in history_ids}
    eligible = [
        {**r, "group_id": groups[i]} for i, r in enumerate(records)
        if not r["quality_reasons"] and groups[i] not in contaminated
    ]
    representatives = {}
    for record in sorted(eligible, key=lambda r: hashlib.sha256(
        f"{config['seed']}:representative:{r['task_id']}".encode()
    ).digest()):
        representatives.setdefault(record["group_id"], record)
    available = list(representatives.values())
    splits = {}
    for name, count in config["split_sizes"].items():
        selected = stratified_sample(available, count, f"{config['seed']}:{name}")
        splits[name] = selected
        ids = {r["task_id"] for r in selected}
        available = [r for r in available if r["task_id"] not in ids]
    splits["train_reserve"] = available
    for child, parent in (("teacher_pilot", "teacher_candidates"), ("dev_smoke", "dev")):
        splits[child] = stratified_sample(
            splits[parent], config["subset_sizes"][child], f"{config['seed']}:{child}"
        )
    return splits, contaminated, len(eligible), len(representatives)


def distribution(records: list[dict]) -> dict:
    return {field: dict(sorted(Counter(str(r[field]) for r in records).items()))
            for field in ("domain", "constraint_band", "budget_required", "required_option_axes")}


def validate_splits(splits: dict, history_ids: set[int], config: dict) -> None:
    seen_ids, seen_groups = set(), set()
    for name in (*config["split_sizes"], "train_reserve"):
        rows = splits[name]
        ids = {r["task_id"] for r in rows}
        groups = {r["group_id"] for r in rows}
        if len(ids) != len(rows) or len(groups) != len(rows):
            raise ValueError(f"duplicate task or group within {name}")
        if ids & (seen_ids | history_ids) or groups & seen_groups:
            raise ValueError(f"historical or cross-split overlap in {name}")
        if any(r["quality_reasons"] for r in rows):
            raise ValueError(f"unqualified task in {name}")
        if name in config["split_sizes"] and len(rows) != config["split_sizes"][name]:
            raise ValueError(f"wrong count in {name}")
        seen_ids.update(ids)
        seen_groups.update(groups)
    for child, parent in (("teacher_pilot", "teacher_candidates"), ("dev_smoke", "dev")):
        ids = {r["task_id"] for r in splits[child]}
        if len(ids) != config["subset_sizes"][child] or not ids <= {r["task_id"] for r in splits[parent]}:
            raise ValueError(f"invalid subset {child}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=IMPLEMENTATION_ROOT / "configs/task_pool.json")
    parser.add_argument("--output-dir", type=Path, default=IMPLEMENTATION_ROOT / "data/task_pool")
    parser.add_argument("--refresh-history", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = ROOT / config["source"]
    contract = json.loads((ROOT / "data/environment.json").read_text())
    if sha256(source) != contract["product_data_sha256"]:
        raise ValueError("source does not match frozen environment product hash")

    # These are the runtime loader and goal builder; never assume task_id == file row.
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals
    from web_agent_site.engine.search import MultiFieldBM25Searcher

    products, by_asin, prices, _ = load_products(str(source))
    goals = get_goals(products, prices)
    searcher = MultiFieldBM25Searcher(
        ROOT / config["search_index"], expected_product_sha256=sha256(source)
    )
    searcher.close()
    connection = sqlite3.connect(f"file:{ROOT / config['search_index']}?mode=ro", uri=True)
    try:
        indexed_asins = {row[0] for row in connection.execute("SELECT asin FROM products")}
    finally:
        connection.close()
    records = [inspect_task(i, goal, by_asin[goal["asin"]], goal["asin"] in indexed_asins)
               for i, goal in enumerate(goals)]
    print(f"Static validation complete: {len(records)} tasks", flush=True)
    output = args.output_dir
    history_path = output / "history.jsonl"
    history_sources = {}
    if history_path.exists() and not args.refresh_history:
        history = read_jsonl(history_path)
        history_sources = json.loads((output / "metadata.json").read_text())["history_sources"]
    else:
        history, history_sources = historical_inventory()
    history_ids = {r["task_id"] for r in history if 0 <= r["task_id"] < len(goals)}
    non_runtime_ids = sorted(r["task_id"] for r in history if r["task_id"] not in history_ids)
    groups, group_stats = duplicate_groups(records, config)
    print(f"Duplicate grouping complete: {len(set(groups))} groups", flush=True)
    splits, contaminated, eligible_count, representative_count = build_splits(
        records, groups, history_ids, config
    )
    validate_splits(splits, history_ids, config)

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(history_path, history)
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", ({"task_id": r["task_id"]} for r in rows))
    role = {r["task_id"]: name for name, rows in splits.items()
            if name not in {"teacher_pilot", "dev_smoke"} for r in rows}
    evaluation_groups = contaminated | {r["group_id"] for name in ("test", "dev")
                                         for r in splits[name]}
    protected_groups = evaluation_groups | {r["group_id"] for r in splits["sft_validation_candidates"]}
    held_out = {r["task_id"] for i, r in enumerate(records) if groups[i] in protected_groups}
    evaluation_only = {r["task_id"] for i, r in enumerate(records) if groups[i] in evaluation_groups}
    write_jsonl(output / "held_out.jsonl", ({"task_id": i} for i in sorted(held_out)))
    write_jsonl(output / "evaluation_holdout.jsonl", ({"task_id": i} for i in sorted(evaluation_only)))
    audit = []
    for i, record in enumerate(records):
        row = {**record, "group_id": groups[i], "historical": record["task_id"] in history_ids,
               "historical_group": groups[i] in contaminated}
        row["assignment"] = role.get(record["task_id"], "excluded")
        if row["assignment"] == "excluded" and not row["quality_reasons"] and not row["historical_group"]:
            row["assignment"] = "duplicate_not_selected"
        audit.append(row)
    private = output / "private"
    private.mkdir(exist_ok=True)
    audit_path = private / "audit.jsonl.gz"
    with audit_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in audit:
                compressed.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())
    (private / "audit.jsonl").unlink(missing_ok=True)

    runtime_paths = [SHOP_ENV / "web_agent_site/engine" / f"{name}.py" for name in (
        "engine", "goal", "constraints", "reward", "reward_features", "variant_price",
        "comparators", "normalize", "search",
    )]
    runtime_paths += [SHOP_ENV / "configs/brand_aliases.json", ROOT / "data/environment.json",
                      Path(__file__), args.config]
    artifact_paths = sorted(output.glob("*.jsonl")) + [audit_path]
    metadata = {
        "schema_version": SCHEMA,
        "config": config,
        "source_sha256": sha256(source),
        "search_index_sha256": sha256(ROOT / config["search_index"]),
        "contract": {key: contract[key] for key in ("environment_version", "observation_version", "tool_version")},
        "reward_version": contract["reward"]["version"],
        "source_task_count": len(records),
        "history_task_count": len(history_ids),
        "non_runtime_history_ids": non_runtime_ids,
        "history_sources": history_sources,
        "quality_pass_count": sum(not r["quality_reasons"] for r in records),
        "quality_rejection_reasons": dict(Counter(reason for r in records for reason in r["quality_reasons"])),
        "historically_contaminated_groups": len(contaminated),
        "fresh_quality_pass_count": eligible_count,
        "fresh_representative_count": representative_count,
        "duplicate_grouping": {"groups": len(set(groups)), **group_stats},
        "splits": {name: {"tasks": len(rows), "distribution": distribution(rows)}
                   for name, rows in splits.items()},
        "quality_pass_distribution": distribution([r for r in audit if not r["quality_reasons"]]),
        "stratification_reference": distribution([
            r for r in audit if r["assignment"] not in {"excluded", "duplicate_not_selected"}
        ]),
        "subsets": {"teacher_pilot": "teacher_candidates", "dev_smoke": "dev"},
        "selection": "Proportional domain/constraint-band/budget/required-option strata; one seeded representative per duplicate group; test first.",
        "validation_scope": "Offline target variant and reward checks plus index membership; no model rollout, API judge or semantic certification.",
        "runtime_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in runtime_paths},
        "files": {str(p.relative_to(output)): {"sha256": sha256(p)} for p in artifact_paths},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": {k: len(v) for k, v in splits.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
