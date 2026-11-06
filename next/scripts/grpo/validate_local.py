"""One bounded, training-pool-only branch acceptance run; never updates weights."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random
import time

from local_credit import decision_for, estimate_decision, seed_for
from rollout import TrainingClient, collect_update
from settings import load_settings
from shopping_agent.protocol import StudentTokenizer, load_config, sha256, validate_execution
from shopping_agent.runs import write_json
from shopping_agent.tasks import TaskPool
from train import TaskStream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-url', required=True)
    parser.add_argument('--environment-url', required=True)
    args = parser.parse_args()
    settings, config, fingerprint = load_settings('local'), load_config(), validate_execution()
    if sha256(args.model/'model.safetensors') != settings['base_weights_sha256']:
        raise ValueError('validation requires the frozen S-repair start')
    pool = TaskPool()
    pool.validate_source()
    tasks = pool.tasks('grpo_train')
    if {t['task_id'] for t in tasks} & pool.protected_ids('train'):
        raise ValueError('validation must not sample held-out tasks')
    tokenizer = StudentTokenizer.load(config, args.model)
    stream = TaskStream(tasks, settings['seed'])
    factory = lambda seed: TrainingClient(args.model_url, 'shopping-agent', tokenizer, seed, settings)
    target = settings['local_credit']['validation_decisions']
    records, counts = [], Counter()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output/'branches.jsonl.gz', 'wt') as log:
        # Bounded source sampling also exposes a broken dynamic sampler.
        for generation in range(1, target+1):
            rows, diagnostic = collect_update(stream, client_factory=factory, config=config,
                tokenizer=tokenizer, env_url=args.environment_url, fingerprint=fingerprint,
                settings=settings, generation=generation, evidence=args.output/'rollouts')
            counts.update(diagnostic['counts'])
            for index, row in enumerate(rows):
                seed = seed_for(settings['seed'], generation, index, 'validation')
                decision = decision_for(row, random.Random(seed))
                if decision is None:
                    continue
                record = estimate_decision(row, decision, seed=seed, client_factory=factory,
                    config=config, tokenizer=tokenizer, env_url=args.environment_url, fingerprint=fingerprint,
                    settings=settings, local=settings['local_credit'], log=log)
                records.append(record)
                write_json(args.output/'validation-progress.json', {'decisions': len(records), 'target': target,
                    'statuses': dict(Counter(r['status'] for r in records)), 'elapsed_seconds': time.monotonic()-started})
                print(f"LOCAL_VALIDATION {len(records)}/{target} {record['status']}", flush=True)
                if len(records) == target:
                    break
            if len(records) == target:
                break
    valid = [r for r in records if r['status'] == 'valid']
    branches = sum(r.get('branches', 0) for r in records)
    verified = sum(r.get('verified_branches', 0) for r in records)
    nonzero = sum(r['increment'] != 0 for r in valid)
    ready = len(records) == target and branches > 0 and verified == branches and nonzero > 0
    costs = Counter()
    for r in records:
        costs.update(r['cost'])
    result = {'status': 'complete', 'ready_for_training': ready, 'optimizer_updates': 0,
        'execution_fingerprint': fingerprint, 'base_weights_sha256': settings['base_weights_sha256'],
        'settings': settings, 'decisions': len(records), 'statuses': dict(Counter(r['status'] for r in records)),
        'branches': branches, 'verified_branches': verified, 'nonzero_comparisons': nonzero,
        'valid_pair_fraction': len(valid)/len(records) if records else 0,
        'branch_cost': dict(costs), 'main_sampling': dict(counts),
        'elapsed_seconds': time.monotonic()-started, 'comparisons': records,
        'limits': 'mechanical acceptance and signal existence only; no evidence of training gain or estimator convergence'}
    write_json(args.output/'validation.json', result)
    if not ready:
        raise RuntimeError('branch acceptance incomplete or no reliable nonzero signal; inspect validation.json')


if __name__ == '__main__':
    main()
