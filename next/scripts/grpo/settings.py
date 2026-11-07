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
    elif variant == 'stable':
        settings.update(json.loads((ROOT / 'configs/grpo_stable.json').read_text()))
        if not (0 < settings['kl_coefficient'] and settings['kl_estimator'] == 'k3+'
                and 1 <= settings['sampling_groups_per_batch'] <= settings['groups_per_update']
                and settings['sampling_groups_per_batch'] * settings['rollouts_per_group'] <= 8
                and settings['max_generation_batches'] * settings['sampling_groups_per_batch'] >= settings['groups_per_update']
                and settings['save_every'] == settings['evaluate_every']
                and settings['updates'] % settings['save_every'] == 0):
            raise ValueError('invalid stable GRPO sampling/KL/checkpoint configuration')
    elif variant != 'outcome':
        raise ValueError('unknown GRPO variant')
    return settings
