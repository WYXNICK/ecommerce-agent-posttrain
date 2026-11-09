import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

NEXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEXT / "scripts/evaluation"))
from audit_hard import observed_reserve, retrieval_signal, structural_signals


class HardCandidateTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((NEXT / "configs/hard_candidates.json").read_text())
        self.product = {"category": "服装›上衣", "customization_options": {
            "颜色": [{"value": "红色", "price": 100}, {"value": "蓝色", "price": 100}],
            "尺码": [{"value": "M", "price": 100}, {"value": "L", "price": 200}],
        }}
        self.goal = {"asin": "gold", "category": "服装›上衣", "price_upper": 150,
                     "instruction_text": "买红色M码上衣，预算不超过150元",
                     "required_options_by_key": {"color": {"value": "红色"}, "size": {"value": "M"}}}

    def test_budget_crossing_requires_resolved_real_variant(self):
        result = structural_signals(self.goal, self.product, self.config)
        self.assertTrue(result["A_multi_axis"])
        self.assertTrue(result["B_budget_coupling"])
        self.product["customization_options"]["尺码"][1]["value"] = "联系客服报价"
        result = structural_signals(self.goal, self.product, self.config)
        self.assertFalse(result["A_multi_axis"])
        self.assertFalse(result["B_budget_coupling"])

    def test_approximate_budget_not_counted_as_exact_conflict(self):
        self.goal["instruction_text"] = "买红色M码上衣，150元左右"
        self.assertFalse(structural_signals(self.goal, self.product, self.config)["B_budget_coupling"])

    def test_missing_metadata_is_not_a_proven_spec_conflict(self):
        hits = [SimpleNamespace(asin=str(i), rank=i + 1) for i in range(3)]
        hits.append(SimpleNamespace(asin="gold", rank=4))
        queries = []
        searcher = SimpleNamespace(search=lambda query, k: queries.append(query) or hits)
        products = {str(i): copy.deepcopy(self.product) for i in range(3)}
        for product in products.values():
            product["customization_options"]["颜色"] = [{"value": "蓝色", "price": 100}]
        self.assertTrue(retrieval_signal(self.goal, products, searcher, self.config)["C_retrieval_discrimination"])
        del products["0"]["customization_options"]["颜色"]
        self.assertFalse(retrieval_signal(self.goal, products, searcher, self.config)["C_retrieval_discrimination"])
        self.assertEqual(queries, [self.goal["instruction_text"]] * 2)

    def test_observed_duplicate_group_excluded_but_planned_ids_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs/run"
            output.mkdir(parents=True)
            (output / "raw.jsonl").write_text('{"task_id": 6}\n')
            (output / "run.json").write_text('{"candidate_task_ids": [5, 7]}')
            original = {5: {"group_id": 1}, 6: {"group_id": 1}, 7: {"group_id": 2}}
            with patch("audit_hard.NEXT", root), patch("audit_hard.ROOT", root):
                used, sources = observed_reserve({5, 7}, original)
            self.assertEqual(used, {5})
            self.assertEqual(sources[0]["observed_task_ids"], [6])


if __name__ == "__main__":
    unittest.main()
