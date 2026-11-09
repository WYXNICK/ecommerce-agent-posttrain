"""Baseline recipe with explicit, isolated experimental overlays."""
import json

from shopping_agent.protocol import ROOT


def load_settings(variant='outcome'):
    settings = json.loads((ROOT / 'configs/grpo.json').read_text())
    if variant == 'local':
        local = json.loads((ROOT / 'configs/grpo_local.json').read_text())
        if not (0 <= local['weight'] <= 1 and 0 < local['clip'] <= 1
                and 0 < local['trajectory_fraction'] <= 1
                and local['continuations'] >= 2 and 1 <= local['workers'] <= 8):
            raise ValueError('invalid local credit sampling/scale configuration')
        settings.update(version='local-grpo-v1', local_credit=local)
    elif variant in ('stable', 'batch'):
        settings.update(json.loads((ROOT / f'configs/grpo_{variant}.json').read_text()))
        if not (1 <= settings['sampling_groups_per_batch'] <= settings['groups_per_update']
                and settings['sampling_groups_per_batch'] * settings['rollouts_per_group'] <= 8
                and settings['max_generation_batches'] * settings['sampling_groups_per_batch'] >= settings['groups_per_update']
                and settings['save_every'] == settings['evaluate_every']
                and settings['updates'] % settings['save_every'] == 0):
            raise ValueError('invalid GRPO sampling/checkpoint configuration')
        if variant == 'stable' and not (0 < settings['kl_coefficient'] and settings['kl_estimator'] == 'k3+'):
            raise ValueError('invalid stable GRPO KL configuration')
    elif variant == 's0':
        settings.update(json.loads((ROOT / 'configs/grpo_s0.json').read_text()))
    elif variant != 'outcome':
        raise ValueError('unknown GRPO variant')
    return settings


def starting_result_path(variant):
    return ROOT / ('results/sft/astra-s0/summary.json' if variant == 's0'
                   else 'results/data-reflow/r1/summary.json')


def starting_candidate(variant):
    result = json.loads(starting_result_path(variant).read_text())
    return result['selected'] if variant == 's0' else result['arms']['repair']['selected']
