"""Offline two-rank update benchmark; never writes production checkpoints.

Replays frozen accepted rollout groups against one saved actor/optimizer state.
Variants change execution only. Results do not establish final reward equality.
"""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, set_seed
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

from train import balance_turns


def forward_turns(model, turns, device, causal_padding=False):
    """Right-pad independent histories; gather only each action's logits."""
    if len(turns) == 1:
        turn = turns[0][0]
        prompt, response = turn['prompt_ids'], turn['response_ids']
        ids = torch.tensor([prompt + response], device=device)
        positions = torch.arange(len(prompt)-1, len(prompt)+len(response)-1, device=device)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       logits_to_keep=positions, use_cache=False)
        return [-F.cross_entropy(output.logits[0].float(),
            torch.tensor(response, device=device), reduction='none')]
    length = max(len(t['prompt_ids']) + len(t['response_ids']) for t, _ in turns)
    ids = torch.zeros((len(turns), length), dtype=torch.long, device=device)
    # Trailing padding has no path to earlier causal targets. No padded target
    # is scored; all-one masks permit the same causal SDPA route as single turns.
    mask = torch.ones_like(ids) if causal_padding else torch.zeros_like(ids)
    positions = sorted({p for t, _ in turns for p in range(
        len(t['prompt_ids']) - 1, len(t['prompt_ids']) + len(t['response_ids']) - 1)})
    index = {p: i for i, p in enumerate(positions)}
    for i, (turn, _) in enumerate(turns):
        tokens = turn['prompt_ids'] + turn['response_ids']
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
        mask[i, :len(tokens)] = 1
    output = model(input_ids=ids, attention_mask=mask,
                   logits_to_keep=torch.tensor(positions, device=device), use_cache=False)
    values = []
    for i, (turn, _) in enumerate(turns):
        start, n = len(turn['prompt_ids']) - 1, len(turn['response_ids'])
        take = torch.tensor([index[p] for p in range(start, start + n)], device=device)
        target = torch.tensor(turn['response_ids'], device=device)
        values.append(-F.cross_entropy(output.logits[i, take].float(), target, reduction='none'))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--batches', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'no_recompute', 'selective_half', 'microbatch2', 'microbatch2_causalpad'], required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    assert dist.get_world_size() == 2
    # Leave room for the production inference server on rank 0 and a safety margin.
    torch.cuda.set_per_process_memory_fraction(.85, rank)
    reservation = torch.empty(20 * 1024**3 if rank == 0 else 0, dtype=torch.uint8, device=rank)
    set_seed(42)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': rank})
    model.config.use_cache = False
    model.enable_input_require_grads()
    if args.variant != 'no_recompute':
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if args.variant == 'selective_half':
        layers = [m for m in model.modules() if type(m).__name__ == 'Qwen3_5DecoderLayer']
        for layer in layers[::2]:
            layer.gradient_checkpointing = False
    model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().cpu().clone() for p in params]
    state = torch.load(args.checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
    optimizer = torch.optim.AdamW(params, lr=1e-6, weight_decay=0, fused=True)
    config = OmegaConf.create({'clip_ratio': .2, 'clip_ratio_low': None,
        'clip_ratio_high': None, 'clip_ratio_c': 3., 'global_batch_info': {}})
    batches = json.loads(args.batches.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    microbatch = 2 if args.variant.startswith('microbatch2') else 1
    causal_padding = args.variant == 'microbatch2_causalpad'

    def reset():
        with torch.no_grad():
            for p, value in zip(params, initial):
                p.copy_(value)
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        for group in optimizer.param_groups:
            group['lr'] = 1e-6
        optimizer.zero_grad(set_to_none=True)

    # Warm the actual longest sequences before timing, without changing the actor.
    warm = balance_turns(batches[0]['rows'], 2)[0][rank][:microbatch]
    sum(x.mean() for x in forward_turns(model, warm, rank, causal_padding)).backward()
    reset()
    torch.cuda.synchronize()
    dist.barrier()
    print(f'BENCH_READY rank={rank} variant={args.variant}', flush=True)
    # Baseline repeat measures the numerical/time floor rather than guessing it.
    cases = [(i, 0) for i in range(len(batches))]
    if args.variant in ('baseline', 'microbatch2'):
        cases.append((0, 1))
    if causal_padding:
        cases = [(0, 0), (0, 1)]
    for batch_index, repeat in cases:
        reset()
        batch = batches[batch_index]
        partitions, loads = balance_turns(batch['rows'], 2)
        turns = partitions[rank]
        count = sum(len(t['response_ids']) for row in batch['rows'] for t in row['rollout_turns'])
        metric = torch.zeros(4, dtype=torch.float64, device=rank)
        logprobs = []
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        for offset in range(0, len(turns), microbatch):
            group = turns[offset:offset + microbatch]
            probs = forward_turns(model, group, rank, causal_padding)
            objectives = []
            for (turn, advantage), logprob in zip(group, probs):
                n = len(turn['response_ids'])
                old = torch.tensor(turn['old_logprobs'], device=rank)
                loss, stats = compute_policy_loss_vanilla(old[None], logprob[None],
                    torch.full_like(logprob[None], advantage), torch.ones_like(logprob[None]), config=config)
                objectives.append(loss * n / count)
                delta = logprob.detach() - old
                values = [loss.detach()*n, stats['actor/pg_clipfrac']*n,
                    delta.abs().sum(), ((delta.exp()-1)-delta).sum()]
                # Match production's per-turn host synchronization in the baseline.
                if args.variant == 'baseline':
                    metric += torch.tensor([v.item() if torch.is_tensor(v) else v for v in values], device=rank)
                else:
                    metric += torch.stack([torch.as_tensor(v, device=rank) for v in values]).double()
                logprobs.append(logprob.detach())
            sum(objectives).backward()
            del probs, objectives, loss, logprob
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for p in params:
            p.grad = flat[offset:offset+p.numel()].view_as(p)
            offset += p.numel()
        # Preserve raw gradients before in-place clipping; outside timing after sync.
        raw_gradient = flat.clone()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        dist.all_reduce(metric)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        timings = [None, None]
        dist.all_gather_object(timings, {'seconds': seconds,
            'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30})
        name = f'{args.variant}-batch{batch_index}-repeat{repeat}'
        torch.save(torch.cat(logprobs).cpu(), args.output / f'{name}-rank{rank}-logprobs.pt')
        if rank == 0:
            update = torch.cat([(p.detach().float().cpu()-v.float()).flatten() for p, v in zip(params, initial)])
            torch.save({'gradient': raw_gradient.cpu(), 'update': update}, args.output / f'{name}-vectors.pt')
            record = {'variant': args.variant, 'batch': batch_index, 'repeat': repeat,
                'source': batch['label'], 'seconds': max(t['seconds'] for t in timings),
                'rank_metrics': timings, 'input_tokens_by_rank': loads, 'action_tokens': count,
                'loss': metric[0].item()/count, 'grad_norm': norm.item(),
                'clip_fraction': metric[1].item()/count}
            with (args.output/'measurements.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print('BENCH_RESULT '+json.dumps(record), flush=True)
        del flat, raw_gradient, logprobs
        dist.barrier()
    if rank == 0:
        (args.output/f'{args.variant}-complete.json').write_text(json.dumps({'status': 'complete'}))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
