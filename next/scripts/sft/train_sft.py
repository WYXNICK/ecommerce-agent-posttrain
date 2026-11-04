#!/usr/bin/env python3
"""Two-GPU LoRA SFT, adapted from the reference train_lora_sft.py."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from sft_data import NEXT
from artifacts import persist
from shopping_agent.protocol import StudentTokenizer, load_config, sha256, validate_execution
from shopping_agent.runs import write_json


def supervised_positions(labels):
    """For micro-batch one, logit at position i predicts label at i+1."""
    if labels.shape[0] != 1:
        raise ValueError("sparse output projection requires the frozen micro-batch size of one")
    positions = (labels[0, 1:] != -100).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("example has no supervised next-token targets")
    return positions, labels[:, positions + 1]


def masked_loss(logits, targets, num_items_in_batch=None, world_size=1):
    import torch.nn.functional as F

    numerator = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
    if num_items_in_batch is None:
        return numerator / targets.numel()
    # Trainer counts tokens across accumulated micro-batches and ranks.
    # DDP averages gradients, so compensate once by world_size.
    return numerator / num_items_in_batch * world_size


def main():
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForMultimodalLM, Trainer, TrainerCallback, TrainingArguments, set_seed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--settings", type=Path, default=NEXT / "configs/sft.json")
    args = parser.parse_args()
    validate_execution()
    settings = json.loads(args.settings.read_text())
    config = load_config()
    if int(os.environ.get("WORLD_SIZE", "1")) != settings["world_size"]:
        raise ValueError("launch with the configured two torchrun ranks")
    prepared = json.loads((args.prepared / "prepared.json").read_text())
    datasets = {}
    for role in ("train", "validation"):
        path = args.prepared / f"{role}.pt"
        if sha256(path) != prepared["roles"][role]["sha256"]:
            raise ValueError("prepared data changed")
        datasets[role] = torch.load(path, map_location="cpu", weights_only=True)
    set_seed(settings["seed"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16,
               attn_implementation="sdpa", device_map={"": local_rank})
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = settings["lora"]
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora["r"],
        lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none", target_modules=lora["target_modules"]))
    kernels = sorted({m.chunk_gated_delta_rule.__module__ for m in model.modules() if callable(getattr(m, "chunk_gated_delta_rule", None))})
    if local_rank == 0:
        model.print_trainable_parameters()
        print("linear_attention_kernels:", kernels, flush=True)

    class AssistantTokenTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            positions, targets = supervised_positions(inputs["labels"])
            # Frozen Qwen3.5 drops num_items_in_batch in its native loss.
            # Project vocabulary logits only at supervised assistant positions,
            # then compute the same masked CE with explicit token normalization.
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            logits_to_keep=positions, use_cache=False)
            loss = masked_loss(outputs.logits, targets, num_items_in_batch,
                               self.accelerator.num_processes if self.args.average_tokens_across_devices else 1)
            return (loss, outputs) if return_outputs else loss

    started = time.monotonic()

    class Metrics(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            # All ranks must finish writing RNG state before the snapshot is copied.
            torch.distributed.barrier()
            if state.is_world_process_zero:
                persist(args.weights / f"checkpoint-{state.global_step}")
            torch.distributed.barrier()

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs:
                return
            if any(not math.isfinite(float(logs[k])) for k in ("loss", "grad_norm", "eval_loss") if k in logs):
                raise FloatingPointError(f"non-finite training metric: {logs}")
            elapsed = time.monotonic() - started
            item = {"step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch,
                    "elapsed_seconds": elapsed, "gpu_peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    **logs}
            if state.global_step:
                item["estimated_seconds_remaining"] = elapsed / state.global_step * (state.max_steps-state.global_step)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            write_json(args.output / "progress.json", item)
            print("SFT_PROGRESS " + json.dumps(item, ensure_ascii=False), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(args.output / "tensorboard")
    tokenizer = StudentTokenizer.load(config, path=args.model)
    training_args = TrainingArguments(output_dir=str(args.weights), num_train_epochs=settings["epochs"],
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"], warmup_ratio=settings["warmup_ratio"],
        weight_decay=settings["weight_decay"], max_grad_norm=settings["max_grad_norm"],
        bf16=True, tf32=True, optim="adamw_torch_fused", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
        average_tokens_across_devices=True, prediction_loss_only=True,
        logging_steps=1, logging_first_step=True, logging_nan_inf_filter=False,
        save_strategy="steps", save_steps=settings["save_steps"], save_total_limit=settings["save_total_limit"],
        eval_strategy="epoch", report_to=["tensorboard"],
        remove_unused_columns=False, label_names=["labels"], train_sampling_strategy="group_by_length",
        seed=settings["seed"], dataloader_num_workers=2, dataloader_persistent_workers=True)

    def collate(batch):
        if len(batch) != 1:
            raise ValueError("configured micro-batch is one")
        return {k:v.unsqueeze(0) for k,v in batch[0].items()}

    trainer = AssistantTokenTrainer(model=model, args=training_args,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=collate, callbacks=[Metrics()])
    trainer.model_accepts_loss_kwargs = True
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.weights / "final-adapter"))
    if trainer.is_world_process_zero():
        tokenizer.tokenizer.save_pretrained(args.weights / "final-adapter")
    torch.distributed.barrier()
    if trainer.is_world_process_zero():
        persist(args.weights / "final-adapter")
    torch.distributed.barrier()
    memory = torch.tensor([torch.cuda.max_memory_allocated() / 1024**3],device=f"cuda:{local_rank}")
    memories = [torch.zeros_like(memory) for _ in range(settings["world_size"])]
    torch.distributed.all_gather(memories,memory)
    if trainer.is_world_process_zero():
        effective_batch = settings["world_size"] * settings["gradient_accumulation_steps"]
        steps_per_epoch = math.ceil(len(datasets["train"]) / effective_batch)
        summary = {"status":"complete", "student":config["student"], "teacher":prepared["teacher"],
            "train_examples":len(datasets["train"]), "validation_examples":len(datasets["validation"]),
            "global_step":trainer.state.global_step,"train_loss":result.training_loss,
            "metrics":result.metrics,"log_history":trainer.state.log_history,
            "peak_gpu_memory_gib_by_rank":[m.item() for m in memories],
            "epoch_checkpoints":[str(args.weights/f"checkpoint-{steps_per_epoch*i}") for i in range(1,settings["epochs"]+1)],
            "sparse_assistant_projection":True,"linear_attention_kernels":kernels,
            "runtime":{p:importlib.metadata.version(p) for p in ["torch","transformers","peft","accelerate"]}}
        write_json(args.output / "train_summary.json",summary)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
