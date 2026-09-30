"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from scripts.train_grpo import build_command, parse_args
from scripts import train_grpo
from shopping_grpo.cli import main as cli_main
from shopping_grpo.smoke import run_cpu_smoke


class PublicEntrypointTest(unittest.TestCase):
    def test_real_launch_preflights_identical_overrides_and_propagates_status(self):
        for preflight_status, trainer_status in ((0, 0), (0, 7), (9, 0)):
            with self.subTest(preflight=preflight_status, trainer=trainer_status):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "training"
                    args = SimpleNamespace(
                        config=Path(directory) / "custom.yaml",
                        logger="console", dry_run=False,
                    )
                    command = [
                        sys.executable, "-m", "verl.trainer.main_ppo",
                        f"--config-path={directory}", "--config-name=custom",
                        "trainer.logger=[console]", "trainer.total_training_steps=2",
                    ]
                    environment = {
                        "GRPO_MODEL_PATH": "model", "GRPO_TRAIN_FILE": "train",
                        "GRPO_VAL_FILE": "validation", "SHOPSIM_BASE_URL": "http://env",
                        "GRPO_OUTPUT_DIR": str(output),
                    }
                    with patch.object(train_grpo, "parse_args", return_value=args), \
                         patch.object(train_grpo, "build_command", return_value=(command, environment)), \
                         patch.object(train_grpo.subprocess, "call", side_effect=[preflight_status, trainer_status]) as call, \
                         patch("builtins.print"), self.assertRaises(SystemExit) as raised:
                        train_grpo.main()
                    self.assertEqual(raised.exception.code, preflight_status or trainer_status)
                    self.assertEqual(call.call_args_list[0].args[0][2:], command[5:])
                    self.assertEqual(call.call_args_list[0].kwargs["env"], environment)
                    self.assertEqual(call.call_count, 1 if preflight_status else 2)
                    self.assertEqual(output.exists(), not bool(preflight_status))
                    if not preflight_status:
                        self.assertEqual(call.call_args_list[1].args[0], command)

    def test_cpu_smoke_covers_public_contracts(self):
        result = run_cpu_smoke()

        self.assertEqual(
            result["checks"],
            [
                "action_schema",
                "trajectory_normalization",
                "reward_sample",
                "sft_label_mask",
                "dynamic_sampling_grouping",
            ],
        )

    def test_offline_example_cli_runs_without_models_or_environment(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(
            sys,
            "argv",
            [
                "shopping-grpo",
                "evaluate",
                str(root / "examples/trajectories.jsonl"),
            ],
        ), patch("builtins.print") as output:
            cli_main()

        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["trajectory_count"], 3)
        self.assertEqual(summary["strict_gold_success_count"], 1)

    def test_public_grpo_launcher_accepts_sharded_weights_and_console(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--logger",
                    "console",
                    "--dry-run",
                ],
            ):
                args = parse_args()
            command, environment = build_command(args)
            output.mkdir()
            checkpoint = output / "global_step_100"
            (checkpoint / "actor").mkdir(parents=True)
            with self.assertRaisesRegex(SystemExit, "new or empty"):
                build_command(args)
            args.resume_from = checkpoint
            args.logger = "tensorboard"
            with self.assertRaisesRegex(SystemExit, "actor/ and data.pt"):
                build_command(args)
            (checkpoint / "data.pt").write_bytes(b"dataloader state")
            resumed, resumed_env = build_command(args)
            self.assertIn("trainer.logger=[console,tensorboard]", resumed)
            self.assertIn("trainer.resume_mode=resume_path", resumed)
            self.assertIn(f"trainer.resume_from_path={checkpoint}", resumed)
            self.assertEqual(resumed_env["TENSORBOARD_DIR"], str(output / "tensorboard"))

        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["GRPO_MODEL_PATH"], str(model.resolve()))
        self.assertEqual(environment["GRPO_TRAIN_FILE"], str(train.resolve()))
        self.assertEqual(environment["GRPO_VAL_FILE"], str(validation.resolve()))
        self.assertEqual(
            environment["SHOPPING_GRPO_DIAGNOSTICS_PATH"],
            str(output.resolve() / "training_diagnostics.jsonl"),
        )
        self.assertIn("trainer.logger=[console]", command)
        self.assertEqual(environment["GRPO_CONFIG_DIR"], str(root / "configs"))


if __name__ == "__main__":
    unittest.main()
