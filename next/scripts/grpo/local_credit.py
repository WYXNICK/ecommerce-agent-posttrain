"""Same-policy Monte Carlo action comparison, without changing the actor protocol."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
import random
from statistics import mean, variance
import time

from shopping_agent.clients import ModelOutputError, ModelReply, ModelServiceError, wire_messages
from shopping_agent.environment.client import ShopAgentEnv
from shopping_agent.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_agent.runtime import run_episode
from rollout import training_signal


class ReplayMismatch(ValueError):
    """A reset/replayed state no longer matches the sampled decision."""


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:4], 'big')


def action_key(message):
    """Compare executable calls, ignoring request IDs and JSON formatting."""
    calls = message.get('tool_calls') or []
    if not calls:
        return ('final', message.get('content'))
    call = calls[0]['function']
    arguments = call.get('arguments') or '{}'
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return ('malformed', call['name'], arguments)
    return (call['name'], json.dumps(arguments, sort_keys=True, ensure_ascii=False))


def decision_for(row, rng):
    # Choose without looking at success/reward/error labels. Undecodable responses
    # remain in terminal GRPO, but cannot be faithfully injected as fixed actions.
    eligible = [i for i, t in enumerate(row['rollout_turns']) if 'replay' in t]
    return rng.choice(eligible) if eligible else None


class BranchClient:
    def __init__(self, turns, decision, action, continuation):
        self.turns, self.decision, self.action = turns, decision, action
        self.continuation = continuation
        self.cursor, self.verified = 0, False

    def complete(self, messages, tools, max_tokens):
        if self.cursor <= self.decision:
            saved = self.turns[self.cursor]['replay']
            if wire_messages(messages) != wire_messages(saved['messages']) or max_tokens != saved['max_tokens']:
                raise ReplayMismatch('visible history or remaining token budget changed')
            selected = self.cursor == self.decision
            self.cursor += 1
            self.verified = selected
            return ModelReply(deepcopy(self.action if selected else saved['message']),
                              {'replayed': True, 'intervention': selected})
        return self.continuation.complete(messages, tools, max_tokens)


class PrefixEnvironment:
    """Verify raw pre-intervention transitions as well as model-visible history."""
    def __init__(self, env, source, decision):
        self.env, self.source = env, source
        prefix = source['rollout_turns'][decision]['replay']['messages']
        call_ids = {c['id'] for m in prefix for c in m.get('tool_calls', [])}
        self.steps = [s for s in source['steps'] if s['tool_call']['id'] in call_ids and s['env_action'] is not None]
        self.cursor = 0
        self.mismatch = None

    @staticmethod
    def stable(result, *, reset=False):
        # shop_agent exposes slot identity and optional cached persona metadata.
        # The frozen actor ignores persona/reason_key; reward reads the task goal,
        # not these cached env attributes. Preserve all decision/reward state.
        ignored = {'env_idx', 'user_persona', 'reason_key'} if reset else {'env_idx'}
        state = {k: v for k, v in result.items() if k not in ignored}
        if not reset and isinstance(state.get('idx'), str) and state['idx'].startswith('slot-'):
            state['idx'] = state['idx'].rsplit('-', 1)[1]
        return state

    def verify(self, expected, actual, *, reset=False):
        left, right = self.stable(expected, reset=reset), self.stable(actual, reset=reset)
        if left != right:
            keys = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
            self.mismatch = {'fields': keys, 'expected': expected, 'actual': actual}
            raise ReplayMismatch(f"{'reset' if reset else 'prefix transition'} changed: {keys}")

    def reset(self, task_id):
        result = self.env.reset(task_id)
        self.verify(self.source['initial_result'], result, reset=True)
        return result

    def step(self, action):
        result = self.env.step(action)
        if self.cursor < len(self.steps):
            expected = self.steps[self.cursor]
            if action != expected['env_action']:
                raise ReplayMismatch('prefix environment action changed')
            self.verify(expected['result'], result)
        self.cursor += 1
        return result

    def release(self):
        return self.env.release()


def comparison(returns, settings):
    """All eight continuations are required; never select successful subsets."""
    if any(r is None for arm in returns for r in arm):
        return None
    q = list(map(mean, returns))
    paired = [(a-b)/2 for a, b in zip(*returns)]
    raw = (q[0]-q[1])/2
    return {'q_original': q[0], 'q_alternative': q[1], 'raw_local_advantage': raw,
            'paired_standard_error': (variance(paired)/len(paired))**0.5,
            'return_variances': list(map(variance, returns)),
            'increment': settings['weight'] * max(-settings['clip'], min(settings['clip'], raw))}


def estimate_decision(source, decision, *, seed, client_factory, config, tokenizer,
                      env_url, fingerprint, settings, local, log, env_factory=ShopAgentEnv):
    started, cost = time.monotonic(), Counter()
    saved = source['rollout_turns'][decision]['replay']
    record = {'kind': 'local_comparison', 'source_trajectory_id': source['trajectory_id'],
              'task_id': source['task_id'], 'policy_generation': source['policy_generation'],
              'decision': decision, 'seed': seed, 'status': 'pending'}
    alternative_client = client_factory(seed_for(seed, 'alternative'))
    try:
        alternative = alternative_client.complete(saved['messages'], SHOP_TOOL_SCHEMAS, saved['max_tokens']).message
    except (ModelOutputError, ModelServiceError) as exc:
        record.update(status='alternative_unavailable', error=f'{type(exc).__name__}: {exc}')
        alternative = None
    cost.update(alternative_client.cost)
    # Preserve exact alternative request tokens/audit, including failed decoding.
    log.write(json.dumps({**record, 'kind': 'alternative', 'turns': alternative_client.turns,
                          'action': alternative}, ensure_ascii=False)+'\n')
    if alternative is not None:
        record['actions'] = [saved['message'], alternative]
        if action_key(saved['message']) == action_key(alternative):
            record['status'] = 'same_action'
        else:
            def branch(item):
                arm, replicate = item
                continuation = client_factory(seed_for(seed, 'continuation', replicate))
                client = BranchClient(source['rollout_turns'], decision, record['actions'][arm], continuation)
                env = PrefixEnvironment(env_factory(base_url=env_url), source, decision)
                row = run_episode({'task_id': source['task_id']}, client=client, config=config,
                    tokenizer=tokenizer, base_url=env_url, fingerprint=fingerprint, env_factory=lambda **_: env)
                error = row.get('error') or {}
                if error.get('type') == 'ValueError':
                    raise ValueError(f'branch execution/token contract violation: {error}')
                verified = client.verified and error.get('type') != 'ReplayMismatch' and env.cursor >= len(env.steps)
                reward = training_signal(row, settings['model_failure_reward']) if verified else None
                row.update(kind='local_branch', source_trajectory_id=source['trajectory_id'],
                           policy_generation=source['policy_generation'], decision=decision,
                           arm=arm, replicate=replicate, replay_verified=verified, training_reward=reward,
                           sampling_cost=dict(continuation.cost), rollout_turns=continuation.turns)
                if env.mismatch:
                    row['replay_mismatch'] = env.mismatch
                return row
            jobs = [(a, k) for a in range(2) for k in range(local['continuations'])]
            with ThreadPoolExecutor(max_workers=local['workers']) as executor:
                rows = list(executor.map(branch, jobs))
            for row in rows:
                cost.update(row['sampling_cost'])
                log.write(json.dumps(row, ensure_ascii=False)+'\n')
            returns = [[r['training_reward'] for r in rows if r['arm'] == a] for a in range(2)]
            estimate = comparison(returns, local)
            record.update(returns=returns, branches=len(rows),
                verified_branches=sum(r['replay_verified'] for r in rows),
                model_failures=sum(r['failure_kind'] == 'model' for r in rows),
                infrastructure_failures=sum(r['failure_kind'] == 'infrastructure' for r in rows),
                unverifiable=sum(r['failure_kind'] == 'unverifiable' for r in rows),
                status='valid' if estimate is not None else 'invalid_pair')
            if estimate is not None:
                record.update(estimate)
    record.update(cost=dict(cost), elapsed_seconds=time.monotonic()-started)
    log.write(json.dumps(record, ensure_ascii=False)+'\n')
    log.flush()
    return record


def apply_local_credit(rows, *, client_factory, config, tokenizer, env_url, fingerprint,
                       settings, generation, evidence):
    local = settings.get('local_credit')
    if not local or local['weight'] == 0:
        return {}  # Exact terminal objective and no additional sampling at lambda=0.
    counts, records = Counter(), []
    with gzip.open(evidence/f'local-{generation:06d}.jsonl.gz', 'wt') as log:
        for index, row in enumerate(rows):
            seed = seed_for(settings['seed'], generation, index, 'local')
            rng = random.Random(seed)
            if rng.random() >= local['trajectory_fraction']:
                continue
            counts['selected_trajectories'] += 1
            decision = decision_for(row, rng)
            if decision is None:
                counts['no_replayable_decision'] += 1
                continue
            record = estimate_decision(row, decision, seed=seed, client_factory=client_factory,
                config=config, tokenizer=tokenizer, env_url=env_url, fingerprint=fingerprint,
                settings=settings, local=local, log=log)
            records.append(record)
            counts[record['status']] += 1
            for key, value in record['cost'].items():
                counts[key] += value
            if record['status'] == 'valid':
                row['rollout_turns'][decision]['local_increment'] = record['increment']
                counts['nonzero_increments'] += record['increment'] != 0
    return {'counts': dict(counts), 'comparisons': records}
