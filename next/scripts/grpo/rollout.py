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
                   settings, generation, evidence):
    """Bounded dynamic sampling uses exactly the returns used by GRPO."""
    accepted, counts, groups = [], Counter(), []
    evidence.mkdir(parents=True, exist_ok=True)
    def episode(item):
        task, seed = item
        client = client_factory(seed)
        row = run_episode(task, client=client, config=config, tokenizer=tokenizer,
                          base_url=env_url, fingerprint=fingerprint)
        if (row.get('error') or {}).get('type') == 'ValueError':
            raise ValueError(f"execution or token contract violation: {row['error']}")
        reward = training_signal(row, settings['model_failure_reward'])
        row.update(training_reward=reward, policy_generation=generation, rollout_turns=client.turns,
                   sampling_cost=dict(client.cost))
        return row
    with gzip.open(evidence / f'generation-{generation:06d}.jsonl.gz', 'wt') as log:
        for batch in range(settings['max_generation_batches']):
            selected = [next(tasks) for _ in range(settings['groups_per_update'])]
            jobs = [(task, settings['seed'] + generation * 10000 + batch * 100 + i * 10 + j)
                    for i, task in enumerate(selected) for j in range(settings['rollouts_per_group'])]
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                rows = list(executor.map(episode, jobs))
            for row in rows:
                log.write(json.dumps(row, ensure_ascii=False) + '\n')
                counts['trajectories'] += 1
                counts['model_failures'] += row['failure_kind'] == 'model'
                counts['infrastructure_failures'] += row['failure_kind'] == 'infrastructure'
                counts['unverifiable'] += row['failure_kind'] == 'unverifiable'
                for key, value in row['sampling_cost'].items():
                    counts['main_' + key] += value
            n = settings['rollouts_per_group']
            for i in range(0, len(rows), n):
                group = rows[i:i+n]
                rewards = [r['training_reward'] for r in group]
                counts['generated_groups'] += 1
                reason = ('invalid' if any(r is None for r in rewards) or any(not r['rollout_turns'] for r in group)
                          else 'equal' if max(rewards) - min(rewards) <= 1e-8 else None)
                if reason is None and len(accepted) < settings['groups_per_update'] * n:
                    for row, advantage in zip(group, group_advantages(rewards)):
                        row['advantage'] = advantage
                    accepted.extend(group)
                    counts['accepted_groups'] += 1
                else:
                    counts['filtered_' + (reason or 'surplus')] += 1
                groups.append({'task_id': group[0]['task_id'], 'rewards': rewards, 'filter': reason})
            if len(accepted) == settings['groups_per_update'] * n:
                return accepted, {'counts': dict(counts), 'groups': groups, 'generation_batches': batch+1}
    return [], {'counts': dict(counts), 'groups': groups, 'generation_batches': settings['max_generation_batches']}
