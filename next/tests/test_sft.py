"""SFT objective equivalence, accumulation normalization and dev selection."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/sft"))
from train_sft import masked_loss, supervised_positions
from supervise_sft import completed_training, selection_key, validate_post_training_update
from sft_data import selected_data
from monitor_gpu import check_keeper


class SFTTests(unittest.TestCase):
    def test_keeper_health_distinguishes_phase_transition_from_crash(self):
        for last_read in (FileNotFoundError(), "124"):
            with patch.object(Path, "read_text", side_effect=["123", FileNotFoundError(), last_read]):
                check_keeper(Path("/keep"), 0)
        for status in (FileNotFoundError(), "123 (python) Z 1"):
            with patch.object(Path, "read_text", side_effect=["123", status, "123"]):
                with self.assertRaisesRegex(RuntimeError, "keeper 0 exited"):
                    check_keeper(Path("/keep"), 0)
        with patch.object(Path, "read_text", side_effect=["123", "123 (python) S 1"]):
            check_keeper(Path("/keep"), 0)

    def test_sparse_projection_matches_dense_loss_and_gradients(self):
        torch.manual_seed(42)
        hidden = torch.randn(1, 11, 7, requires_grad=True)
        head = torch.randn(19, 7, requires_grad=True)
        labels = torch.tensor([[-100, -100, 4, 9, -100, -100, 2, 3, 5, -100, 6]])
        logits = F.linear(hidden, head)
        dense = F.cross_entropy(logits[:, :-1].reshape(-1, 19), labels[:, 1:].reshape(-1))
        gradients = torch.autograd.grad(dense, (hidden, head), retain_graph=True)
        positions, targets = supervised_positions(labels)
        sparse = masked_loss(F.linear(hidden[:, positions], head), targets)
        actual = torch.autograd.grad(sparse, (hidden, head))
        torch.testing.assert_close(sparse, dense)
        for observed, expected in zip(actual, gradients):
            torch.testing.assert_close(observed, expected)

    def test_accumulation_and_ddp_match_global_token_average(self):
        torch.manual_seed(1)
        head = torch.randn(5, 3, requires_grad=True)
        lengths = (2, 7, 3, 5)  # two ranks, each accumulating two unequal micro-batches.
        hidden = [torch.randn(n, 3) for n in lengths]
        targets = [torch.randint(0, 5, (n,)) for n in lengths]
        dense = F.cross_entropy(F.linear(torch.cat(hidden), head), torch.cat(targets))
        expected, = torch.autograd.grad(dense, head)
        normalized = sum(masked_loss(F.linear(h, head), t, sum(lengths), world_size=2)
                         for h, t in zip(hidden, targets)) / 2  # DDP gradient averaging.
        observed, = torch.autograd.grad(normalized, head)
        torch.testing.assert_close(normalized, dense)
        torch.testing.assert_close(observed, expected)

    def test_qwen_native_logit_selection_and_lora_backward(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from peft import LoraConfig, TaskType, get_peft_model

        config = Qwen3_5Config(text_config={"vocab_size": 32, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 16, "layer_types": ["full_attention"],
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.5}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16,
                "patch_size": 2}, image_token_id=30, video_token_id=31)
        model = Qwen3_5ForConditionalGeneration(config)
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
        model.eval()
        ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
        labels = torch.tensor([[-100, -100, 6, 7, -100, 9, 10, 11]])
        positions, targets = supervised_positions(labels)
        full = model(input_ids=ids, use_cache=False).logits
        sparse = model(input_ids=ids, logits_to_keep=positions, use_cache=False).logits
        torch.testing.assert_close(sparse, full[:, positions])
        dense_loss = F.cross_entropy(full[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1))
        torch.testing.assert_close(masked_loss(sparse, targets), dense_loss)
        masked_loss(sparse, targets).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))

    def test_selected_teacher_and_holdout_boundaries(self):
        rows, _, manifest = selected_data()
        self.assertEqual(manifest["teacher"], "gpt-6-astra-2026-09-03")
        self.assertEqual({k: len(v) for k, v in rows.items()}, {"train": 2000, "validation": 200})

    def test_selection_requires_complete_dev_and_prespecified_ties(self):
        candidate = {"step": 250, "summary": {"status": "complete", "completed_tasks": 300,
            "expected_tasks": 300, "unresolved_infrastructure_tasks": [], "strict_successes": 180,
            "mean_final_reward": 0.6, "average_steps": 10.0}}
        later = deepcopy(candidate)
        later["step"] = 500
        self.assertGreater(selection_key(candidate), selection_key(later))
        later["summary"]["strict_successes"] = 181
        self.assertGreater(selection_key(later), selection_key(candidate))
        later["summary"]["completed_tasks"] = 299
        with self.assertRaises(ValueError):
            selection_key(later)

    def test_post_training_repairs_preserve_training_identity(self):
        original = {"settings": {"epochs": 3}, "entry_files": {
            "scripts/sft/train_sft.py": "train", "scripts/sft/sft_data.py": "data",
            "scripts/sft/export_model.py": "old-export", "scripts/sft/supervise_sft.py": "old-supervisor"}}
        current = deepcopy(original)
        current["entry_files"]["scripts/sft/export_model.py"] = "fixed-export"
        self.assertEqual(validate_post_training_update(original, current, {"status": "complete"}),
                         ["scripts/sft/export_model.py"])
        current["entry_files"]["scripts/sft/monitor_gpu.py"] = "fixed-monitor"
        self.assertIn("scripts/sft/monitor_gpu.py",
                      validate_post_training_update(original, current, {"status": "complete"}))
        with self.assertRaises(ValueError):
            validate_post_training_update(original, current, None)
        current["entry_files"]["scripts/sft/train_sft.py"] = "changed-training"
        with self.assertRaises(ValueError):
            validate_post_training_update(original, current, {"status": "complete"})
        current = deepcopy(original)
        current["settings"]["epochs"] = 4
        with self.assertRaises(ValueError):
            validate_post_training_update(original, current, {"status": "complete"})

    def test_completion_marker_requires_saved_optimizer_and_epoch_weights(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {"world_size": 2, "micro_batch_size": 1, "gradient_accumulation_steps": 4, "epochs": 1}
            self.assertIsNone(completed_training(root, root, settings))
            checkpoint = root / "checkpoints/checkpoint-1"
            checkpoint.mkdir(parents=True)
            (root / "train_summary.json").write_text(json.dumps({"status": "complete", "global_step": 1,
                "train_examples": 8, "epoch_checkpoints": [str(checkpoint)]}))
            (checkpoint / "trainer_state.json").write_text('{"global_step":1}')
            for name in ("adapter_model.safetensors", "scheduler.pt"):
                (checkpoint / name).write_bytes(b"saved")
            with self.assertRaises(ValueError):
                completed_training(root, root, settings)
            (checkpoint / "optimizer.pt").write_bytes(b"saved")
            final = root / "checkpoints/final-adapter"
            final.mkdir()
            (final / "adapter_model.safetensors").write_bytes(b"saved")
            self.assertEqual(completed_training(root, root, settings)["global_step"], 1)


if __name__ == "__main__":
    unittest.main()
