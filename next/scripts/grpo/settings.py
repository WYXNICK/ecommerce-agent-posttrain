"""One baseline recipe; local credit is an explicit, isolated overlay."""
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
    elif variant != 'outcome':
        raise ValueError('unknown GRPO variant')
    return settings
