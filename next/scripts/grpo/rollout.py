"""Frozen actor execution with exact per-turn rollout tokens (no retokenized targets)."""
from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from constraint_reward import constrained_return
from success_reward import success_group_filter, success_return
from shopping_agent.evaluation.summary import _is_strict_success
from shopping_agent.clients import ModelReply, decode_completion, request_json, wire_messages
from shopping_agent.runtime import run_episode
from shopping_agent.training.sft.dataset import normalize_messages_for_chat_template


class TrainingClient:
    def __init__(self, url, model, tokenizer, seed, settings):
        self.url, self.model, self.tokenizer = url, model, tokenizer
        self.seed, self.settings, self.turns = seed, settings, []
        self.cost = Counter()

    def complete(self, messages, tools, max_tokens):
        payload = {
            'model': self.model, 'messages': wire_messages(messages), 'tools': tools,
            'tool_choice': 'auto', 'parallel_tool_calls': False,
            'temperature': self.settings['temperature'], 'top_p': self.settings['top_p'],
            'seed': self.seed + len(self.turns), 'max_tokens': max_tokens,
            'chat_template_kwargs': {'enable_thinking': False},
            'logprobs': True, 'top_logprobs': 0,
            'return_token_ids': True, 'return_tokens_as_token_ids': True,
        }
        if self.settings.get('self_distillation'):
            payload['top_k'] = -1
        request_id, started = uuid4().hex, time.monotonic()
        self.cost['requests'] += 1
        response = request_json(self.url + '/chat/completions', payload,
            headers={'Content-Type': 'application/json'}, timeout=300)
        choice = response['choices'][0]
        prompt, tokens = response['prompt_token_ids'], choice['token_ids']
        self.cost.update(prompt_tokens=len(prompt), completion_tokens=len(tokens))
        logs = choice['logprobs']['content']
        canonical = self.tokenizer.apply_chat_template(normalize_messages_for_chat_template(messages),
            tools=tools, tokenize=True, return_dict=False, add_generation_prompt=True)
        if prompt != canonical or not tokens or len(tokens) != len(logs):
            raise ValueError('rollout token/prompt contract mismatch')
        if [int(x['token'].removeprefix('token_id:')) for x in logs] != tokens:
            raise ValueError('rollout log probabilities refer to different tokens')
        logprobs = [x['logprob'] for x in logs]
        if not all(math.isfinite(x) for x in logprobs):
            raise ValueError('non-finite rollout log probability')
        # Record BEFORE parsing: malformed/length-limited model output must learn.
        turn = {'prompt_ids': prompt, 'response_ids': tokens, 'old_logprobs': logprobs}
        if self.settings.get('self_distillation'):
            turn['messages'] = deepcopy(messages)
        self.turns.append(turn)
        message, audit = decode_completion(response, request_id, time.monotonic() - started)
        if self.settings.get('local_credit', {}).get('weight', 0):
            turn['replay'] = {'messages': deepcopy(messages), 'max_tokens': max_tokens,
                              'message': deepcopy(message)}
        return ModelReply({k: message[k] for k in ('role', 'content', 'tool_calls') if k in message}, audit)


def training_signal(trajectory, failure_cost):
    if trajectory['failure_kind'] in {'infrastructure', 'unverifiable'}:
        return None
    if not trajectory['done']:
        return failure_cost
    value = float(trajectory['final_reward'])
    if not math.isfinite(value):
        raise ValueError('non-finite terminal reward')
    return value


def group_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def collect_update(tasks, *, client_factory, config, tokenizer, env_url, fingerprint,
                   settings, generation, evidence, group_transform=None):
    """Bounded dynamic sampling uses exactly the returns used by GRPO."""
    accepted, counts, groups = [], Counter(), []
    batch_groups = settings.get('sampling_groups_per_batch', settings['groups_per_update'])
    seen_tasks = set()
    evidence.mkdir(parents=True, exist_ok=True)
    def episode(item):
        task, seed = item
        client = client_factory(seed)
        row = run_episode(task, client=client, config=config, tokenizer=tokenizer,
                          base_url=env_url, fingerprint=fingerprint)
        if (row.get('error') or {}).get('type') == 'ValueError':
            raise ValueError(f"execution or token contract violation: {row['error']}")
        reward = training_signal(row, settings['model_failure_reward'])
        if settings.get('constraint_reward'):
            row['original_training_reward'] = reward
            reward, row['constraint_reward'] = constrained_return(row, reward, settings['constraint_reward'])
        if settings.get('success_objective'):
            row['constraint_training_reward'] = reward
            reward = success_return(row, reward, settings['rollouts_per_group'],
                                    settings['constraint_reward']['wrong_purchase_reward'])
        row.update(training_reward=reward, policy_generation=generation, rollout_turns=client.turns,
                   sampling_cost=dict(client.cost))
        return row
    with gzip.open(evidence / f'generation-{generation:06d}.jsonl.gz', 'wt') as log:
        for batch in range(settings['max_generation_batches']):
            selected = []
            while len(selected) < batch_groups:
                task = next(tasks)
                if task['task_id'] in seen_tasks:
                    continue
                seen_tasks.add(task['task_id'])
                selected.append(task)
            jobs = [(task, settings['seed'] + generation * 10000 + batch * 100 + i * 10 + j)
                    for i, task in enumerate(selected) for j in range(settings['rollouts_per_group'])]
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                rows = list(executor.map(episode, jobs))
            for row in rows:
                counts['trajectories'] += 1
                counts['model_failures'] += row['failure_kind'] == 'model'
                counts['infrastructure_failures'] += row['failure_kind'] == 'infrastructure'
                counts['unverifiable'] += row['failure_kind'] == 'unverifiable'
                if settings.get('success_objective'):
                    counts['strict_success_trajectories'] += _is_strict_success(row)
                if settings.get('constraint_reward'):
                    counts['constraint_penalized_trajectories'] += row['constraint_reward']['decision'] == 'penalized'
                    counts['constraint_restored_model_failures'] += row['constraint_reward']['restored_model_failure']
                    counts.update('constraint_'+v['reason'] for v in row['constraint_reward']['violations'])
                for key, value in row['sampling_cost'].items():
                    counts['main_' + key] += value
            n = settings['rollouts_per_group']
            for i in range(0, len(rows), n):
                group = rows[i:i+n]
                rewards = [r['training_reward'] for r in group]
                counts['generated_groups'] += 1
                if group_transform is not None:
                    eligible = group_transform(group, tokenizer, settings, counts)
                    reason = None if eligible else 'no_feedback'
                else:
                    eligible = group
                    reason = ('invalid' if any(r is None for r in rewards) or any(not r['rollout_turns'] for r in group)
                              else 'equal' if max(rewards) - min(rewards) <= 1e-8 else None)
                    if settings.get('success_objective') and reason != 'invalid':
                        reason = success_group_filter(group) or reason
                if reason is None and counts['accepted_groups'] < settings['groups_per_update']:
                    if group_transform is None:
                        for row, advantage in zip(group, group_advantages(rewards)):
                            row['advantage'] = advantage
                            if settings.get('success_objective'):
                                success = _is_strict_success(row)
                                counts['accepted_strict_successes'] += success
                                counts['accepted_failures'] += not success
                                counts['failure_positive_advantages'] += not success and advantage > 0
                                counts['success_nonpositive_advantages'] += success and advantage <= 0
                            if settings.get('constraint_reward') and row['constraint_reward']['decision'] == 'penalized':
                                counts['constraint_accepted_trajectories'] += 1
                                counts['constraint_positive_advantages'] += advantage > 0
                    accepted.extend(eligible)
                    counts['accepted_groups'] += 1
                else:
                    reason = reason or 'surplus'
                    counts['filtered_' + reason] += 1
                groups.append({'task_id': group[0]['task_id'], 'rewards': rewards, 'filter': reason})
                for row in group:
                    log.write(json.dumps(row, ensure_ascii=False) + '\n')
            if counts['accepted_groups'] == settings['groups_per_update']:
                return accepted, {'counts': dict(counts), 'groups': groups, 'generation_batches': batch+1}
    counts['discarded_partial_groups'] = counts['accepted_groups']
    return [], {'counts': dict(counts), 'groups': groups, 'generation_batches': settings['max_generation_batches']}
