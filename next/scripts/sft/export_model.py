#!/usr/bin/env python3
"""Export an SFT LoRA checkpoint; adapted from the reference merge_lora_adapter.py."""

import argparse
from pathlib import Path

from sft_data import NEXT
from shopping_agent.protocol import sha256
from shopping_agent.runs import write_json


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("export directory must be empty")
    base = AutoModelForMultimodalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.output)
    write_json(args.output / "merge_manifest.json", {
        "operation": "peft_merge_and_unload", "dtype": "bfloat16", "device": "cpu",
        "base_model": str(args.base_model), "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "files": {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    main()
