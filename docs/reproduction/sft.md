# 该历史实验：固定 800 / 200 的 LoRA SFT

> 历史200题实验记录；本页选模与“主结果”仅适用于该历史系列。当前项目结论见[核心结果](../results.md)。

## Purpose

SFT teaches the base model to issue legal tool calls, use observations as
evidence, select product variants and terminate in ShopSimulator v2.1.

## Fixed inputs

- Base model: `Qwen/Qwen3.5-2B`
- Training: `data/sft/train.jsonl`, 800 trajectories
- Validation: `data/sft/validation.jsonl`, 200 trajectories
- Target: assistant tokens only; user and tool observations are masked
- No task ID overlap with GRPO train/validation or Final-200

The 200 validation trajectories are distinct from the 200 final evaluation
tasks. File hashes and teacher provenance are recorded in
[`data/sft/metadata.json`](../../data/sft/metadata.json); see also the
[data description](data.md). Use the fixed split without resampling.

## Run

After `bash scripts/setup.sh` and model download:

```bash
bash scripts/sft.sh
```

`BASE_MODEL` can point to a local snapshot of the base model. Record its model
revision so Baseline and SFT start from the same weights. The launcher trains
an adapter, then merges it into the base model:

```text
outputs/models/sft-lora/
outputs/models/sft-merged/
```

| Setting | Value |
|---|---|
| Maximum sequence length | 24,576 |
| Epochs | 3 |
| Per-device train / validation batch size | 1 / 1 |
| Gradient accumulation | 8 |
| Learning rate / warmup ratio | `1e-4` / `0.03` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Gradient checkpointing | enabled |
| Attention implementation | SDPA |
| Precision | BF16 on supported CUDA devices |
| Seed | 42 |
| Save / validate | each epoch |

The context budget preserves complete multi-turn interactions. The loader
**drops** examples exceeding the limit rather than truncating their terminal
actions. Record both input counts and retained counts from the preprocessing
statistics; 800/200 file rows do not necessarily mean 800/200 training examples.

The trainer supports `--resume-from-checkpoint` for an interrupted run. Optional
QLoRA or fused loss must be selected explicitly and recorded; neither is enabled
by the default launcher. GPU capacity must be verified with the actual recipe.

## Evaluate and continue to GRPO

```bash
bash scripts/serve_model.sh outputs/models/sft-merged
bash scripts/evaluate.sh sft
```

Validation loss measures training health. Freeze the recipe before evaluating
Final-200; do not select checkpoints using the final test score.

GRPO starts from the merged model:

```text
GRPO_MODEL_PATH=outputs/models/sft-merged
```

Use the same environment, final tasks and evaluation settings for Baseline,
SFT and GRPO. Previously published scores are not measurements of this new run.
