#!/usr/bin/env python3
"""Export a veRL FSDP actor as a standalone model, including its trained LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def export_actor(checkpoint: Path, output: Path) -> None:
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"actor checkpoint does not exist: {checkpoint}")
    if output.exists():
        raise ValueError(f"output must not already exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # veRL 0.8 exports frozen base weights and lora_adapter separately.
    # Publishing that base alone would silently discard the GRPO update.
    with tempfile.TemporaryDirectory(prefix=".grpo-export-", dir=output.parent) as temporary:
        exported = Path(temporary) / "actor"
        subprocess.run(
            [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
             "--local_dir", str(checkpoint), "--target_dir", str(exported),
             "--trust-remote-code"],
            check=True,
        )
        adapter = exported / "lora_adapter"
        if adapter.exists():
            config = json.loads((adapter / "adapter_config.json").read_text())
            if float(config.get("lora_alpha", 0)) <= 0:
                raise ValueError("exported LoRA alpha is missing/zero; check actor/lora_train_meta.json")
            if not (adapter / "adapter_model.safetensors").is_file():
                raise ValueError("exported LoRA weights are missing")
            merged = Path(temporary) / "merged"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/merge_lora_adapter.py"),
                 "--base-model", str(exported), "--adapter", str(adapter),
                 "--output", str(merged), "--bf16"],
                check=True,
            )
            manifest_path = merged / "merge_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["source"] = {
                "actor_checkpoint": str(checkpoint),
                "model_type": manifest["source"]["model_type"],
                "lora_rank": config["r"],
                "lora_alpha": config["lora_alpha"],
            }
            manifest["output"] = str(output)
            manifest["next_step"] = "serve this standalone GRPO model for evaluation"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            shutil.move(str(merged), str(output))
        else:
            shutil.move(str(exported), str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "outputs/models/grpo-merged")
    args = parser.parse_args()
    export_actor(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
