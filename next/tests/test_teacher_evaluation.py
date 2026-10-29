"""Teacher benchmark completion gate and full-test specification checks."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/teacher_evaluation"))
from evaluate import specification, teacher_config
from run import collection_complete


class TeacherEvaluationTests(unittest.TestCase):
    def dependency(self, directory, config):
        (directory / "status.json").write_text(json.dumps({
            "status": "complete", "teacher": config["teacher"], "execution_fingerprint": "frozen"}))
        for role, count in (("train", 2000), ("validation", 200)):
            (directory / role).mkdir()
            (directory / role / "summary.json").write_text(json.dumps({
                "status": "complete", "target": count, "accepted_unique_tasks": count}))

    def test_gate_waits_for_collection_and_accepts_only_complete_datasets(self):
        config = teacher_config("astra")
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            self.assertFalse(collection_complete(directory, config, "frozen"))
            self.dependency(directory, config)
            self.assertTrue(collection_complete(directory, config, "frozen"))
            state = json.loads((directory / "status.json").read_text())
            state["status"] = "failed"
            (directory / "status.json").write_text(json.dumps(state))
            self.assertFalse(collection_complete(directory, config, "frozen"))

    def test_gate_rejects_wrong_teacher_contract_and_incomplete_artifacts(self):
        config = teacher_config("astra")
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            self.dependency(directory, config)
            with self.assertRaises(ValueError):
                collection_complete(directory, teacher_config("sol"), "frozen")
            with self.assertRaises(ValueError):
                collection_complete(directory, config, "changed")
            (directory / "train/summary.json").write_text(json.dumps({
                "status": "complete", "target": 2000, "accepted_unique_tasks": 1999}))
            with self.assertRaises(ValueError):
                collection_complete(directory, config, "frozen")

    def test_teacher_spec_preserves_entire_test_and_does_not_change_protocol(self):
        config = teacher_config("sol")
        original = deepcopy(config)
        tasks = [{"task_id": i} for i in range(1000)]
        spec = specification(config, tasks, "frozen", "http://localhost:5703", 6)
        self.assertEqual(config, original)
        self.assertEqual(spec["candidate_task_ids"], list(range(1000)))
        self.assertTrue(spec["formal"])
        self.assertEqual(spec["split"], "test")
        self.assertEqual(spec["behavior_attempts_per_task"], 1)
        self.assertEqual(spec["teacher"], config["teacher"])
        self.assertEqual(set(spec["entry_files"]), {"evaluate.py", "run.py"})


if __name__ == "__main__":
    unittest.main()
