"""Replay frozen training evidence, without sampling or updating any model."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from constraint_reward import constrained_return
from rollout import group_advantages
from settings import load_settings
from shopping_agent.protocol import ROOT, sha256


def audit(rollouts, metrics, sampling):
    settings = load_settings('constraints')
    pool = ROOT / 'data/task_pool/grpo_train.jsonl'
    allowed = {row['task_id'] for row in map(json.loads, pool.read_text().splitlines())}
    selected = {}
    for line in metrics.read_text().splitlines():
        record = json.loads(line)
        # The original run left surplus groups unmarked; only the first accepted_groups trained.
        groups = [g for g in record['sampling']['groups'] if g['filter'] is None]
        selected[record['generation']] = {
            g['task_id']: g['rewards'] for g in groups[:record['sampling']['counts']['accepted_groups']]}
    for line in sampling.read_text().splitlines():
        selected[json.loads(line)['generation']] = {}
    counts, reasons, tasks, examples = Counter(), Counter(), set(), []
    manifest = hashlib.sha256()
    seen_generations = set()
    for path in sorted(rollouts.glob('generation-*.jsonl.gz')):
        generation = int(path.name.split('.')[0].split('-')[-1])
        seen_generations.add(generation)
        manifest.update(f'{path.name} {sha256(path)}\n'.encode())
        groups = {}
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['task_id'] not in allowed:
                    raise ValueError('audit accepts frozen GRPO training tasks only')
                old = row['training_reward']
                new, evidence = constrained_return(row, old, settings['constraint_reward'])
                changed = evidence['decision'] == 'penalized'
                counts['trajectories'] += 1
                counts['penalized_trajectories'] += changed
                counts['filtered_trajectories'] += old is None
                counts['restored_model_failures'] += evidence['restored_model_failure']
                detail = (row.get('terminal_result') or {}).get('reward_detail') or {}
                counts['partial_purchases'] += detail.get('reward_type') == 'partial_alternative_purchase'
                if changed:
                    tasks.add(row['task_id'])
                    reasons.update(v['reason'] for v in evidence['violations'])
                groups.setdefault(row['task_id'], []).append((old, new, changed))
                if changed and len(examples) < 5:
                    examples.append({'generation': generation, 'task_id': row['task_id'],
                        'trajectory_id': row['trajectory_id'], 'old_reward': old, 'new_reward': new,
                        'violations': evidence['violations']})
        for task_id, group in groups.items():
            old, new, changed = map(list, zip(*group))
            if len(group) != settings['rollouts_per_group']:
                raise ValueError('incomplete rollout group')
            if any(r is None for r in old):
                counts['invalid_groups'] += 1
                counts['restored_valid_groups'] += all(r is not None for r in new)
                counts['restored_informative_groups'] += (all(r is not None for r in new)
                                                          and max(new)-min(new) > 1e-8)
                continue
            old_signal, new_signal = max(old)-min(old) > 1e-8, max(new)-min(new) > 1e-8
            counts['valid_groups'] += 1
            counts['groups_losing_signal'] += old_signal and not new_signal
            counts['groups_gaining_signal'] += new_signal and not old_signal
            if task_id not in selected[generation]:
                continue
            if old != selected[generation][task_id]:
                raise ValueError('rollout rewards differ from original accepted group')
            counts['accepted_groups'] += 1
            counts['accepted_groups_losing_signal'] += not new_signal
            a_old, a_new = group_advantages(old), group_advantages(new)
            for index, affected in enumerate(changed):
                counts['accepted_penalized_trajectories'] += affected
                counts['penalized_positive_before'] += affected and a_old[index] > 1e-8
                counts['penalized_positive_after'] += affected and a_new[index] > 1e-8
                counts['unaffected_failure_positive_before'] += not affected and old[index] != 1 and a_old[index] > 1e-8
                counts['unaffected_failure_positive_after'] += not affected and old[index] != 1 and a_new[index] > 1e-8
    if seen_generations != set(selected):
        raise ValueError('metrics and rollout generations differ')
    return {'status': 'offline_audit_complete', 'training_started': False,
        'scope': 'original outcome training rollouts; fixed trajectories, not new-policy results',
        'recipe': settings, 'counts': dict(counts), 'penalized_unique_tasks': len(tasks),
        'violation_reasons': dict(reasons), 'examples': examples,
        'inputs': {'metrics_sha256': sha256(metrics), 'sampling_sha256': sha256(sampling),
                   'rollouts_manifest_sha256': manifest.hexdigest(),
                   'rollout_files': len(seen_generations), 'training_pool_sha256': sha256(pool)},
        'implementation': {name: sha256(Path(__file__).with_name(name))
                           for name in ('constraint_reward.py', 'audit_constraints.py')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--sampling', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.rollouts, args.metrics, args.sampling)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('counts', 'penalized_unique_tasks', 'violation_reasons')}, ensure_ascii=False))
