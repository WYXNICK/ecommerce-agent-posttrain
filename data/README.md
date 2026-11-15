# Historical data and shared environment contract

Current training and test1000 data live in [next/data](../next/data/). This directory retains early task splits for historical recovery and leakage exclusion. `environment.json` remains a shared execution contract. The counts below describe the earlier 200-task protocol, not current training.

| Stage | Files | Rows |
|---|---|---:|
| SFT | `sft/train.jsonl`, `sft/validation.jsonl` | 800 / 200 |
| GRPO | `grpo/train.parquet`, `grpo/validation.parquet` | 1000 / 50 |
| Evaluation | `evaluation/tasks.jsonl` (Final-200 Clean) | 200 |

Adjacent `metadata.json` files record SHA256 checksums and collection
provenance. All SFT, GRPO and evaluation splits are task-disjoint. Generated
trajectories belong under `outputs/`, never under `data/`. Use
`scripts/collect_sft_data.py` to create a new audited SFT dataset before
promoting its train/validation files into this directory.
