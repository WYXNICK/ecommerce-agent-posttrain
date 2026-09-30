import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.export_grpo import export_actor


class GrpoExportTest(unittest.TestCase):
    def test_merges_exported_adapter_before_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "actor"
            checkpoint.mkdir()
            output = root / "final"

            def execute(command, **kwargs):
                self.assertTrue(kwargs["check"])
                if "verl.model_merger" in command:
                    stage = Path(command[command.index("--target_dir") + 1])
                    adapter = stage / "lora_adapter"
                    adapter.mkdir(parents=True)
                    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
                    (adapter / "adapter_model.safetensors").write_bytes(b"trained adapter")
                    (stage / "model.safetensors").write_bytes(b"frozen base")
                else:
                    self.assertTrue(command[1].endswith("merge_lora_adapter.py"))
                    stage = Path(command[command.index("--output") + 1])
                    stage.mkdir()
                    (stage / "model.safetensors").write_bytes(b"merged trained weights")
                    (stage / "merge_manifest.json").write_text(json.dumps({
                        "source": {"model_type": "qwen3_5"}, "operation": "peft_merge_and_unload"}))

            with patch("scripts.export_grpo.subprocess.run", side_effect=execute) as run:
                export_actor(checkpoint, output)
            self.assertEqual(run.call_count, 2)
            self.assertEqual((output / "model.safetensors").read_bytes(), b"merged trained weights")
            manifest = json.loads((output / "merge_manifest.json").read_text())
            self.assertEqual(manifest["source"]["actor_checkpoint"], str(checkpoint))
            self.assertEqual(manifest["source"]["lora_alpha"], 32)
            self.assertEqual(list(root.glob(".grpo-export-*")), [])

    def test_refuses_zero_alpha_without_publishing_frozen_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "actor"
            checkpoint.mkdir()
            output = root / "final"

            def execute(command, **kwargs):
                adapter = Path(command[command.index("--target_dir") + 1]) / "lora_adapter"
                adapter.mkdir(parents=True)
                (adapter / "adapter_config.json").write_text('{"r": 16, "lora_alpha": 0}')

            with patch("scripts.export_grpo.subprocess.run", side_effect=execute) as run:
                with self.assertRaisesRegex(ValueError, "alpha"):
                    export_actor(checkpoint, output)
            self.assertEqual(run.call_count, 1)
            self.assertFalse(output.exists())

    def test_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "final"
            output.mkdir()
            sentinel = output / "model.safetensors"
            sentinel.write_bytes(b"existing model")
            with patch("scripts.export_grpo.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "already exist"):
                    export_actor(root, output)
            run.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"existing model")


if __name__ == "__main__":
    unittest.main()
