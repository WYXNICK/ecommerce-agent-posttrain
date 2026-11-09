"""Synchronous two-GPU terminal GRPO using veRL's clipped policy loss.

Turn-unpacked teacher forcing retains each exact serving prompt and sampled action.
All turns in an episode share its group-centered return. Gradients are summed
once across ranks and normalized by the global count of sampled action tokens.
"""
import argparse
from datetime import timedelta
from functools import partial
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from urllib.request import Request, urlopen

from artifacts import persist, inventory
from rollout import TrainingClient, collect_update
from local_credit import apply_local_credit
from settings import load_settings, starting_candidate
from shopping_agent.clients import LocalChatClient, request_json
from shopping_agent.protocol import ROOT, StudentTokenizer, load_config, sha256, validate_execution
from shopping_agent.runs import RunStore, evaluate, write_json
from shopping_agent.runtime import run_episode
from shopping_agent.tasks import TaskPool
from supervise_sft import selection_key


class TaskStream:
    def __init__(self, tasks, seed):
        self.tasks, self.random = tasks, random.Random(seed)
        self.order, self.cursor = [], 0

    def __next__(self):
        if self.cursor == len(self.order):
            self.order = list(self.tasks)
            self.random.shuffle(self.order)
            self.cursor = 0
        task = self.order[self.cursor]
        self.cursor += 1
        return task

    def state(self):
        return {'order': self.order, 'cursor': self.cursor, 'rng': self.random.getstate()}


def balance_turns(rows, world_size):
    """Assign independent action histories by length; keep the global objective."""
    turns = [(turn, row['advantage'] + turn.get('local_increment', 0.0))
             for row in rows for turn in row['rollout_turns']]
    partitions, loads = [[] for _ in range(world_size)], [0] * world_size
    for turn, advantage in sorted(turns, key=lambda item: len(item[0]['prompt_ids']) + len(item[0]['response_ids']), reverse=True):
        rank = min(range(world_size), key=loads.__getitem__)
        partitions[rank].append((turn, advantage))
        loads[rank] += len(turn['prompt_ids']) + len(turn['response_ids'])
    return partitions, loads


def validate_resume_settings(saved, current):
    # Evaluation frequency affects selection, not the optimizer or rollout state.
    if {k: v for k, v in saved.items() if k != 'evaluate_every'} != {
            k: v for k, v in current.items() if k != 'evaluate_every'}:
        raise ValueError('resume checkpoint belongs to another training contract')


def compare_successes(summary, baseline):
    current = set(summary['strict_success_task_ids'])
    initial = set(baseline['strict_success_task_ids'])
    return {'gained_task_ids': sorted(current - initial),
            'lost_task_ids': sorted(initial - current),
            'gained': len(current - initial), 'lost': len(initial - current)}


def control(url, payload):
    request = Request(url, data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=180) as response:
        response.read()


def publish_adapter(model, work, url, step, previous):
    name = f'grpo-step-{step}'
    directory = work / 'live' / name
    model.save_pretrained(directory)
    if previous:
        control(url + '/unload_lora_adapter', {'lora_name': previous})
    control(url + '/load_lora_adapter', {'lora_name': name, 'lora_path': str(directory)})
    if previous:
        shutil.rmtree(work / 'live' / previous)
    return name


def eval_dev(step, name, args, config, tokenizer, fingerprint, adapter, baseline):
    tasks = TaskPool().tasks('dev')
    spec = {'kind': 'evaluate', 'split': 'dev', 'formal': False,
            'execution_fingerprint': fingerprint, 'candidate_task_ids': [t['task_id'] for t in tasks],
            'base_model': str(args.model), 'base_sha256': sha256(args.model / 'model.safetensors'),
            'adapter': str(adapter), 'adapter_files': inventory(adapter), 'step': step,
            'served_model': name, 'workers': 8}
    episode = partial(run_episode, client=LocalChatClient(args.model_url, name), config=config,
                      tokenizer=tokenizer, base_url=args.environment_url, fingerprint=fingerprint)
    output = args.output / 'dev' / f'step-{step}'
    with RunStore(output, spec) as store:
        summary = evaluate(store, tasks=tasks, episode=episode, workers=8)
    item = {'step': step, 'checkpoint': str(adapter), 'summary': summary,
            'vs_starting_sft': compare_successes(summary, baseline),
            'evidence': {p.name: sha256(p) for p in output.glob('*.json*')}}
    selection_key(item)
    return item


def main():
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForMultimodalLM, set_seed
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    import importlib.metadata
    from torch.utils.tensorboard import SummaryWriter
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from regularization import reference_logprobs, reference_kl, action_entropy

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'work', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--variant', choices=('outcome', 'local', 'stable', 's0', 'batch'), default='outcome')
    parser.add_argument('--model-url', required=True)
    parser.add_argument('--environment-url', required=True)
    args = parser.parse_args()
    settings = load_settings(args.variant)
    regularized = settings['kl_coefficient'] > 0
    fingerprint, config = validate_execution(), load_config()
    pool = TaskPool()
    tasks = pool.tasks('grpo_train')
    if len(tasks) != 2000 or {t['task_id'] for t in tasks} & pool.protected_ids('train'):
        raise ValueError('RL tasks are not the isolated 2000-task training pool')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    if dist.get_world_size() != settings['world_size']:
        raise ValueError('requires two GPU ranks')
    if qwen.chunk_gated_delta_rule is None or qwen.FusedRMSNormGated is None:
        raise RuntimeError('GRPO actor requires the validated FLA kernels')
    if rank == 0:
        print('ACTOR_KERNELS '+json.dumps({'fla_core': importlib.metadata.version('fla-core'),
            'delta_rule': qwen.chunk_gated_delta_rule.__module__,
            'normalization': qwen.FusedRMSNormGated.__module__, 'convolution': 'torch'}), flush=True)
    set_seed(settings['seed'])
    tokenizer = StudentTokenizer.load(config, args.model)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                attn_implementation='sdpa', device_map={'': rank})
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = PeftModel.from_pretrained(model, args.resume, is_trainable=True) if args.resume else get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=settings['lora_rank'],
        lora_alpha=settings['lora_alpha'], lora_dropout=0.0, target_modules='all-linear',
        exclude_modules='.*visual.*', bias='none'))
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], weight_decay=0.0, fused=True)
    resume = torch.load(args.resume/'training_state.pt', map_location='cpu', weights_only=False) if args.resume else None
    if resume:
        validate_resume_settings(resume['settings'], settings)
        if resume['execution_fingerprint'] != fingerprint:
            raise ValueError('resume checkpoint belongs to another training contract')
        optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['torch_rng'])
        torch.cuda.set_rng_state(resume['cuda_rng'])
    step = resume['step'] if resume else 0
    generation = resume['generation'] if resume else 0
    policy_config = OmegaConf.create({'clip_ratio': settings['clip_ratio'], 'clip_ratio_low': None,
                                     'clip_ratio_high': None, 'clip_ratio_c': 3.0, 'global_batch_info': {}})
    if rank == 0:
        model.print_trainable_parameters()
        writer = SummaryWriter(str(args.output / 'tensorboard'))
        stream = TaskStream(tasks, settings['seed'])
        base = starting_candidate(args.variant)
        candidates = [{**base, 'step': 0, 'role': 'starting_sft'}]
        if resume:
            saved = resume['task_stream']
            stream.order, stream.cursor = saved['order'], saved['cursor']
            stream.random.setstate(saved['rng'])
            candidates = resume['candidates']
        active_name = publish_adapter(model, args.work, args.model_url, step, None)
        if resume and step % settings['evaluate_every'] == 0 and not any(c['step'] == step for c in candidates):
            candidates.append(eval_dev(step, active_name, args, config, tokenizer, fingerprint, args.resume, candidates[0]['summary']))
            write_json(args.output/'selection.json', {'candidates': candidates,
                'selected': max(candidates, key=selection_key), 'test_evaluation': 'not scheduled'})
    dist.barrier()
    start_step, skipped = step, 0
    prior_elapsed = json.loads((args.resume/'training.json').read_text())['elapsed_seconds'] if resume else 0.0
    started = time.monotonic()
    while step < settings['updates']:
        generation += 1
        packet = [None]
        rollout_started = time.monotonic()
        if rank == 0:
            factory = lambda seed: TrainingClient(args.model_url, active_name, tokenizer, seed, settings)
            rows, diagnostic = collect_update(stream, client_factory=factory, config=config,
                tokenizer=tokenizer, env_url=args.environment_url, fingerprint=fingerprint,
                settings=settings, generation=generation, evidence=args.output/'rollouts')
            if rows:
                diagnostic['local_credit'] = apply_local_credit(rows, client_factory=factory,
                    config=config, tokenizer=tokenizer, env_url=args.environment_url, fingerprint=fingerprint,
                    settings=settings, generation=generation, evidence=args.output/'rollouts')
            packet[0] = (rows, diagnostic)
        dist.broadcast_object_list(packet, src=0, device=torch.device('cuda', rank))
        rows, diagnostic = packet[0]
        rollout_seconds = time.monotonic() - rollout_started
        if not rows:
            skipped += 1
            if rank == 0:
                with (args.output/'sampling.jsonl').open('a') as f:
                    f.write(json.dumps({'generation': generation, 'step': step, **diagnostic})+'\n')
            if skipped >= settings['max_consecutive_skipped_updates']:
                raise RuntimeError('bounded sampler repeatedly found no valid reward-varying groups')
            continue
        skipped = 0
        # Full history is input only. Each sampled action appears once as a target,
        # even if it recurs as context in later turns. No observation-token loss.
        token_count = sum(len(t['response_ids']) for row in rows for t in row['rollout_turns'])
        optimizer.zero_grad(set_to_none=True)
        update_started = time.monotonic()
        lr = settings['learning_rate'] * min(1.0, (step+1)/settings['warmup_updates'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        metric = torch.zeros(7, device=rank, dtype=torch.float64)
        partitions, input_tokens_by_rank = balance_turns(rows, settings['world_size'])
        for turn, episode_advantage in partitions[rank]:
            prompt, response = turn['prompt_ids'], turn['response_ids']
            ids = torch.tensor([prompt + response], device=rank)
            positions = torch.arange(len(prompt)-1, len(prompt)+len(response)-1, device=rank)
            targets = torch.tensor(response, device=rank)
            ref = reference_logprobs(model, ids, positions, targets) if regularized else None
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           logits_to_keep=positions, use_cache=False)
            logprob = -F.cross_entropy(output.logits[0].float(), targets, reduction='none')
            old = torch.tensor(turn['old_logprobs'], device=rank)
            advantage = torch.full_like(logprob, episode_advantage)
            loss, stats = compute_policy_loss_vanilla(old[None], logprob[None], advantage[None],
                torch.ones_like(logprob)[None], config=policy_config)
            kl = reference_kl(logprob, ref).mean() if regularized else logprob.new_zeros(())
            entropy = action_entropy(output.logits[0]) if regularized else logprob.new_zeros(())
            objective = loss + settings['kl_coefficient'] * kl
            (objective * len(response) / token_count).backward()
            delta = (logprob.detach()-old)
            metric += torch.tensor([loss.item()*len(response), stats['actor/pg_clipfrac']*len(response),
                delta.abs().sum().item(), ((delta.exp()-1)-delta).sum().item(), len(response),
                kl.item()*len(response), entropy.item()], device=rank)
            del output, loss, objective, kl, entropy, logprob, ref, ids
        # Each rank accumulated numerator/global_token_count; sum, do not average.
        gradients = [p.grad if p.grad is not None else torch.zeros_like(p) for p in parameters]
        flat = torch.cat([g.reshape(-1) for g in gradients])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for p in parameters:
            p.grad = flat[offset:offset+p.numel()].view_as(p)
            offset += p.numel()
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings['max_grad_norm'], error_if_nonfinite=True)
        dist.all_reduce(metric)
        if step == 0 and (metric[2].item()/token_count > 0.2 or metric[1].item()/token_count > 0.25):
            raise ValueError('actor/rollout likelihood mismatch exceeds first-update acceptance thresholds')
        optimizer.step()
        step += 1
        update_seconds = time.monotonic()-update_started
        if rank == 0:
            active_name = publish_adapter(model, args.work, args.model_url, step, active_name)
            session_elapsed = time.monotonic()-started
            elapsed = prior_elapsed + session_elapsed
            result = {'step': step, 'max_steps': settings['updates'], 'generation': generation,
                'elapsed_seconds': elapsed, 'estimated_training_seconds_remaining': session_elapsed/(step-start_step)*(settings['updates']-step),
                'rollout_seconds': rollout_seconds, 'update_seconds': update_seconds, 'learning_rate': lr,
                'loss': metric[0].item()/token_count, 'clip_fraction': metric[1].item()/token_count,
                'rollout_actor_logprob_abs_difference': metric[2].item()/token_count,
                'rollout_actor_kl_estimate': metric[3].item()/token_count,
                'grad_norm': norm.item(), 'trained_action_tokens': token_count, 'input_tokens_by_rank': input_tokens_by_rank,
                'mean_training_reward': sum(r['training_reward'] for r in rows)/len(rows),
                'strict_success_rate': sum((r['terminal_result'].get('reward_detail') or {}).get('reward_type')=='gold_purchase' for r in rows)/len(rows),
                'sampling': diagnostic, 'peak_memory_gib_rank0': torch.cuda.max_memory_allocated()/2**30}
            if regularized:
                result.update(reference_kl_k3=metric[5].item()/token_count,
                    action_entropy=metric[6].item()/token_count,
                    kl_loss=settings['kl_coefficient']*metric[5].item()/token_count,
                    total_loss=(metric[0].item()+settings['kl_coefficient']*metric[5].item())/token_count,
                    accepted_trajectories=len(rows),
                    accepted_tasks=len({row['task_id'] for row in rows}))
            if not all(math.isfinite(v) for v in result.values() if isinstance(v, (int, float))):
                raise FloatingPointError('non-finite GRPO metric')
            with (args.output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(result)+'\n')
            write_json(args.output/'progress.json', result)
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar('train/'+key, value, step)
            for key, value in diagnostic['counts'].items():
                writer.add_scalar('sampling/'+key, value, step)
            for key, value in diagnostic.get('local_credit', {}).get('counts', {}).items():
                writer.add_scalar('local_credit/'+key, value, step)
            for key, value in diagnostic.get('local_credit', {}).get('metrics', {}).items():
                writer.add_scalar('local_credit/'+key, value, step)
            writer.flush()
            print('GRPO_PROGRESS '+json.dumps({k:v for k,v in result.items() if k != 'sampling'}), flush=True)
            if step % settings['save_every'] == 0:
                checkpoint = args.work / 'checkpoints' / f'step-{step}'
                model.save_pretrained(checkpoint)
                tokenizer.tokenizer.save_pretrained(checkpoint)
                torch.save({'optimizer': optimizer.state_dict(), 'step': step, 'generation': generation,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                    'task_stream': stream.state(), 'candidates': candidates, 'settings': settings,
                    'execution_fingerprint': fingerprint}, checkpoint/'training_state.pt')
                write_json(checkpoint/'training.json', result)
                persist(checkpoint)
                evidence = args.work/'evidence'/f'step-{step}'
                evidence.mkdir(parents=True)
                for name in ('run.json', 'metrics.jsonl', 'sampling.jsonl', 'train.log', 'gpu.jsonl'):
                    source = args.output/name
                    if source.exists():
                        shutil.copy2(source, evidence/name)
                shutil.copytree(args.output/'tensorboard', evidence/'tensorboard')
                for source in sorted((args.output/'rollouts').glob('*.jsonl.gz')):
                    if not (args.work/'evidence'/'archived-rollouts'/source.name).exists():
                        (evidence/'rollouts').mkdir(exist_ok=True)
                        shutil.copy2(source, evidence/'rollouts'/source.name)
                persist(evidence)
                marker = args.work/'evidence'/'archived-rollouts'
                marker.mkdir(exist_ok=True)
                for source in (evidence/'rollouts').glob('*'):
                    (marker/source.name).touch()
            if step % settings['evaluate_every'] == 0:
                candidate = eval_dev(step, active_name, args, config, tokenizer, fingerprint, checkpoint, candidates[0]['summary'])
                candidates.append(candidate)
                best = max(candidates, key=selection_key)
                report = {'status': 'running' if step < settings['updates'] else 'training_complete',
                          'candidates': candidates, 'selected': best, 'test_evaluation': 'not scheduled'}
                write_json(args.output/'selection.json', report)
                writer.add_scalar('dev/strict_success_rate', candidate['summary']['strict_success_rate'], step)
                writer.add_scalar('dev/mean_final_reward', candidate['summary']['mean_final_reward'], step)
                writer.add_scalar('dev/average_steps', candidate['summary']['average_steps'], step)
                for key in ('wrong_purchase', 'repeat_loop'):
                    writer.add_scalar('dev/'+key, candidate['summary']['reward_type_counts'].get(key, 0), step)
                for key in ('gained', 'lost'):
                    writer.add_scalar('dev/'+key, candidate['vs_starting_sft'][key], step)
                writer.flush()
                dev_evidence = args.work/'evidence'/f'dev-{step}'
                shutil.copytree(args.output/'dev'/f'step-{step}', dev_evidence)
                persist(dev_evidence)
        dist.barrier()
    if rank == 0:
        writer.close()
        write_json(args.output/'training-complete.json', {'status': 'complete', 'step': step,
                  'elapsed_seconds': prior_elapsed + time.monotonic()-started})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
