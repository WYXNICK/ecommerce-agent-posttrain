"""Audit fixed training rollouts; never use dev/test trajectories to build rewards."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from rollout import group_advantages
from settings import load_settings
from success_reward import success_return, success_group_filter
from shopping_agent.evaluation.summary import _is_strict_success
from shopping_agent.protocol import ROOT, sha256


def audit(source):
    settings = load_settings('success')
    pool = ROOT/'data/task_pool/grpo_train.jsonl'
    allowed = {r['task_id'] for r in map(json.loads, pool.read_text().splitlines())}
    selected = {}
    for line in (source/'metrics.jsonl').read_text().splitlines():
        record = json.loads(line)
        selected[record['generation']] = {g['task_id']: g['rewards']
            for g in record['sampling']['groups'] if g['filter'] is None}
    skipped = {json.loads(line)['generation'] for line in (source/'sampling.jsonl').read_text().splitlines()}
    counts, failure_types, examples = Counter(), Counter(), []
    manifest, seen = hashlib.sha256(), set()
    for path in sorted((source/'rollouts').glob('generation-*.jsonl.gz')):
        generation = int(path.name.split('.')[0].split('-')[-1])
        seen.add(generation)
        manifest.update(f'{path.name} {sha256(path)}\n'.encode())
        groups = {}
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['task_id'] not in allowed:
                    raise ValueError('audit requires frozen GRPO training tasks')
                groups.setdefault(row['task_id'], []).append(row)
        usable = 0
        for task_id, group in groups.items():
            if len(group) != settings['rollouts_per_group']:
                raise ValueError('incomplete group')
            rewards = [r['training_reward'] for r in group]
            successes = [_is_strict_success(r) for r in group]
            new = [success_return(r, v, len(group), settings['constraint_reward']['wrong_purchase_reward'])
                   for r, v in zip(group, rewards)]
            counts['generated_groups'] += 1
            counts['trajectories'] += len(group)
            invalid = any(v is None for v in new) or any(not r['rollout_turns'] for r in group)
            reason = 'invalid' if invalid else success_group_filter(group)
            counts['new_filter_'+(reason or 'accepted_candidate')] += 1
            if reason is None:
                usable += 1
                advantages = group_advantages(new)
                counts['new_candidate_failure_positive_advantages'] += sum(
                    not s and a > 0 for s, a in zip(successes, advantages))
                counts['new_candidate_success_nonpositive_advantages'] += sum(
                    s and a <= 0 for s, a in zip(successes, advantages))
            if task_id not in selected.get(generation, {}):
                continue
            if rewards != selected[generation][task_id]:
                raise ValueError('accepted rewards disagree with training metrics')
            counts['old_updated_groups'] += 1
            counts['old_updated_groups_without_success'] += not any(successes)
            old_advantages = group_advantages(rewards)
            for row, success, advantage in zip(group, successes, old_advantages):
                tokens = sum(len(t['response_ids']) for t in row['rollout_turns'])
                counts['old_update_action_tokens'] += tokens
                if not success:
                    counts['old_failure_action_tokens'] += tokens
                if not success and advantage > 1e-8:
                    counts['old_positive_failure_trajectories'] += 1
                    counts['old_positive_failure_in_mixed_group' if any(successes)
                           else 'old_positive_failure_in_all_failure_group'] += 1
                    counts['old_positive_failure_action_tokens'] += tokens
                    counts['old_positive_failure_advantage_token_mass'] += advantage*tokens
                    detail = (row.get('terminal_result') or {}).get('reward_detail') or {}
                    failure_types[detail.get('reward_type', row['status'])] += 1
                elif success and advantage > 0:
                    counts['old_positive_success_advantage_token_mass'] += advantage*tokens
            if not any(successes) and len(examples) < 5:
                examples.append({'generation': generation, 'task_id': task_id,
                                 'rewards': rewards, 'advantages': old_advantages})
        counts['observed_prefix_has_full_new_batch'] += usable >= settings['groups_per_update']
    if seen != set(selected) | skipped:
        raise ValueError('rollout and metrics generation sets differ')
    return {'status': 'offline_audit_complete', 'training_started': False,
        'scope': 'fixed constraints-v1 training trajectories only; not a new policy evaluation',
        'counts': dict(counts), 'positive_failure_types': dict(failure_types), 'examples': examples,
        'limits': 'Old sampling stops once its own batch is full. Prefix yield is not the new sampler runtime/cost; no improvement claim.',
        'inputs': {'metrics_sha256': sha256(source/'metrics.jsonl'),
                   'sampling_sha256': sha256(source/'sampling.jsonl'),
                   'rollouts_manifest_sha256': manifest.hexdigest(), 'rollout_files': len(seen),
                   'training_pool_sha256': sha256(pool)},
        'recipe': settings,
        'implementation': {n:sha256(Path(__file__).with_name(n))
                           for n in ['audit_success.py','success_reward.py','rollout.py','settings.py']}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['counts','positive_failure_types']}, ensure_ascii=False))
