import itertools
import hashlib
import gzip
import json
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_task_pool import (
    build_splits,
    color_families,
    duplicate_groups,
    inspect_task,
    similar_pairs,
    stratified_sample,
    task_ids_in,
    trigrams,
    validate_splits,
)
from web_agent_site.engine.constraints import explicit_budget_from_instruction
from web_agent_site.engine.reward_features import compile_reward_features


def record(task_id, *, instruction=None, title=None):
    return {
        "task_id": task_id, "asin": str(task_id),
        "title": title or f"商品{chr(65 + task_id)}",
        "instruction": instruction or f"需求{chr(65 + task_id)}",
        "domain": "Home" if task_id % 2 else "Kids",
        "constraint_band": "medium", "budget_required": bool(task_id % 3),
        "required_option_axes": task_id % 2, "quality_reasons": [],
    }


class PrepareTaskPoolTests(unittest.TestCase):
    def test_frozen_artifacts_have_valid_hashes_and_no_split_leakage(self):
        directory = ROOT / "data/task_pool"
        if not (directory / "metadata.json").exists():
            self.skipTest("run prepare_task_pool.py first")
        metadata = json.loads((directory / "metadata.json").read_text())
        config = json.loads((ROOT / "configs/task_pool.json").read_text())
        self.assertEqual(metadata["config"], config)
        for name, digest in metadata["runtime_sha256"].items():
            self.assertEqual(hashlib.sha256((REPOSITORY_ROOT / name).read_bytes()).hexdigest(), digest, name)
        for name, entry in metadata["files"].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                             entry["sha256"], name)
        with gzip.open(directory / "private/audit.jsonl.gz", "rt", encoding="utf-8") as stream:
            audit = {r["task_id"]: r for r in (json.loads(line) for line in stream)}
        splits = {}
        for name in metadata["splits"]:
            ids = [json.loads(line)["task_id"] for line in (directory / f"{name}.jsonl").read_text().splitlines()]
            splits[name] = [audit[i] for i in ids]
            self.assertEqual(len(ids), metadata["splits"][name]["tasks"])
        history = {json.loads(line)["task_id"]
                   for line in (directory / "history.jsonl").read_text().splitlines()}
        validate_splits(splits, history, config)
        held_out = {json.loads(line)["task_id"]
                    for line in (directory / "held_out.jsonl").read_text().splitlines()}
        for name in ("teacher_candidates", "grpo_train", "train_reserve"):
            self.assertFalse({r["task_id"] for r in splits[name]} & held_out)
        for name in ("test", "dev", "sft_validation_candidates"):
            self.assertTrue({r["task_id"] for r in splits[name]} <= held_out)
        canonical = {json.loads(line)["task_id"]
                     for line in (REPOSITORY_ROOT / "data/evaluation/tasks.jsonl").read_text().splitlines()}
        self.assertTrue(canonical <= held_out)
        for rows in splits.values():
            for r in rows:
                self.assertFalse(r["historical_group"])
                self.assertFalse(r["quality_reasons"])
                self.assertTrue(r["gold_check"]["reward_valid"])
                self.assertEqual(r["gold_check"]["reward_type"], "gold_purchase")
        group_roles = {}
        for name in (*config["split_sizes"], "train_reserve"):
            for r in splits[name]:
                group_roles[r["group_id"]] = name
        # Also cover rejected historical relatives, not just selected task IDs.
        historical_groups = {r["group_id"] for r in audit.values() if r["historical"]}
        self.assertFalse(set(group_roles) & historical_groups)
        protected_groups = historical_groups | {
            r["group_id"] for name in ("test", "dev", "sft_validation_candidates")
            for r in splits[name]
        }
        self.assertEqual(held_out, {r["task_id"] for r in audit.values()
                                    if r["group_id"] in protected_groups})

    def test_exact_similarity_join_matches_brute_force(self):
        rng = random.Random(29)
        texts = ["".join(rng.choices("abcdefghijklmnop", k=40)) for _ in range(25)]
        texts += [text[:30] + "XYZ" + text[33:] for text in texts]
        texts += ["", "a", "a", "ａｂｃ", "ABC"]
        for threshold in (0.5, 0.85, 0.95, 1.0):
            expected = set()
            for a, b in itertools.combinations(range(len(texts)), 2):
                x, y = trigrams(texts[a]), trigrams(texts[b])
                if x and y and len(x & y) / len(x | y) >= threshold:
                    expected.add((a, b))
            actual = {tuple(sorted(pair)) for pair in similar_pairs(texts, threshold)}
            self.assertEqual(actual, expected, threshold)

    def test_historical_group_is_excluded_transitively(self):
        rows = [record(i) for i in range(12)]
        rows[1]["asin"] = rows[0]["asin"]
        rows[2]["title"] = rows[1]["title"]
        rows[3]["instruction"] = "购买电热杯，预算100元"
        rows[4]["instruction"] = "购买电热杯，预算200元"
        config = {
            "seed": 1, "instruction_similarity": 0.85, "title_similarity": 0.95,
            "split_sizes": {"test": 1, "dev": 1, "sft_validation_candidates": 1,
                            "teacher_candidates": 2, "grpo_train": 1},
            "subset_sizes": {"teacher_pilot": 1, "dev_smoke": 1},
        }
        groups, _ = duplicate_groups(rows, config)
        self.assertEqual(groups[0], groups[2])
        self.assertEqual(groups[3], groups[4])
        splits, _, _, _ = build_splits(rows, groups, {0}, config)
        validate_splits(splits, {0}, config)
        selected = {r["task_id"] for values in splits.values() for r in values}
        self.assertFalse(selected & {0, 1, 2})
        self.assertEqual(len(selected & {3, 4}), 1)

    def test_stratification_preserves_counts_and_is_order_independent(self):
        rows = [record(i) for i in range(100)]
        a = stratified_sample(rows, 30, "seed")
        b = stratified_sample(list(reversed(rows)), 30, "seed")
        self.assertEqual(a, b)
        self.assertEqual(len({r["task_id"] for r in a}), 30)
        self.assertEqual(sum(r["domain"] == "Home" for r in a), 15)
        with self.assertRaises(ValueError):
            stratified_sample(rows, 101, "seed")

    def test_history_reads_nested_rollouts_and_excluded_old_tests(self):
        data = {"rollouts": [{"task_id": 3}], "curation": {"excluded_source_task_ids": [9]},
                "success_task_ids": [7], "tasks": 200}
        self.assertEqual(task_ids_in(data), {3, 7, 9})

    def test_quality_uses_real_reward_and_rejects_unparsed_price(self):
        product = {
            "asin": "cup", "title": "保温杯", "Title": "保温杯", "category": "家居›杯子",
            "domain_en_short": "Home", "attribute": ["保温"], "Attributes": ["保温"],
            "pricing": [50.0], "customization_options": {},
        }

        def inspect(query):
            instruction = {"instruction": query, "attributes": ["保温"], "instruction_options": []}
            goal = {"asin": "cup", "category": product["category"], "instruction_text": query,
                    "price_upper": explicit_budget_from_instruction(query)}
            goal.update(compile_reward_features(instruction, product))
            return inspect_task(0, goal, product, True)

        self.assertEqual(inspect("保温杯预算100元以内")["quality_reasons"], [])
        self.assertIn("target_not_valid_gold", inspect("保温杯预算20元以内")["quality_reasons"])
        self.assertIn("unparsed_price_hint", inspect("保温杯价格便宜一些")["quality_reasons"])

    def test_gold_score_does_not_override_conflicting_color_annotation(self):
        product = {
            "asin": "shoe", "title": "平底鞋", "Title": "平底鞋", "category": "服装›鞋",
            "domain_en_short": "Clothing", "attribute": ["平底"], "Attributes": ["平底"],
            "pricing": [50.0],
            "customization_options": {"颜色分类": [{"value": "银色", "price": 50.0}]},
        }
        instruction = {"instruction": "棕色平底鞋，预算100元以内", "attributes": ["平底"],
                       "instruction_options": ["银色"]}
        goal = {"asin": "shoe", "category": product["category"],
                "instruction_text": instruction["instruction"], "price_upper": 100.0}
        goal.update(compile_reward_features(instruction, product))
        result = inspect_task(0, goal, product, True)
        self.assertEqual(result["gold_check"]["reward_type"], "gold_purchase")
        self.assertIn("query_option_color_conflict", result["quality_reasons"])

    def test_color_gate_respects_mixed_colors_aliases_and_relative_phrases(self):
        self.assertTrue(color_families("白、黑、红色拼色") & color_families("标准白/黑色/荧光焰红"))
        self.assertTrue(color_families("灰色") & color_families("灰蓝色"))
        self.assertEqual(color_families("粉红色"), color_families("粉色"))
        self.assertEqual(color_families("咖啡色"), color_families("棕色"))
        self.assertFalse(color_families("与白色正相反的颜色"))
        self.assertFalse(color_families("不要黑色"))


if __name__ == "__main__":
    unittest.main()
