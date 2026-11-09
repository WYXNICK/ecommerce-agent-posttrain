"""Evaluate one frozen test candidate with the existing execution contract."""
import argparse
from functools import partial
import json
from pathlib import Path
from urllib.request import urlopen

from shopping_agent.clients import LocalChatClient
from shopping_agent.protocol import ROOT, StudentTokenizer, load_config, sha256, validate_execution
from shopping_agent.runs import RunStore, evaluate
from shopping_agent.runtime import run_episode
from shopping_agent.tasks import TaskPool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-url', required=True)
    parser.add_argument('--environment-url', required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/test_comparison.json')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    model = next(m for m in plan['models'] if m['name'] == args.name)
    fingerprint = validate_execution()
    if fingerprint != plan['execution_fingerprint'] or sha256(ROOT/'data/task_pool/test.jsonl') != plan['test_sha256']:
        raise ValueError('test protocol or task split changed')
    base = args.models_root/model['base']
    for name, digest in model['base_files'].items():
        if sha256(base/name) != digest:
            raise ValueError(f'base identity mismatch: {name}')
    if model['adapter']:
        for name, digest in model['adapter_files'].items():
            if sha256(args.models_root/model['adapter']/name) != digest:
                raise ValueError(f'adapter identity mismatch: {name}')
    with urlopen(args.model_url+'/models', timeout=30) as response:
        cards = json.load(response)['data']
    card = next(c for c in cards if c['id'] == args.name)
    expected = args.models_root/(model['adapter'] or model['base'])
    if Path(card['root']).resolve() != expected.resolve():
        raise ValueError('served model differs from frozen candidate')
    pool = TaskPool()
    pool.validate_source()
    tasks = pool.tasks('test')
    if len(tasks) != plan['tasks']:
        raise ValueError('test count changed')
    spec = {'kind':'evaluate', 'split':'test', 'formal':True,
            'execution_fingerprint':fingerprint, 'comparison_sha256':sha256(args.plan),
            'entry_sha256':sha256(Path(__file__)), 'model':model,
            'candidate_task_ids':[t['task_id'] for t in tasks], 'workers':8,
            'model_url':args.model_url, 'environment_url':args.environment_url,
            'served_model':args.name, 'format':'base_plus_adapter' if model['adapter'] else 'merged_sft'}
    config = load_config()
    episode = partial(run_episode, client=LocalChatClient(args.model_url,args.name), config=config,
                      tokenizer=StudentTokenizer.load(config), base_url=args.environment_url, fingerprint=fingerprint)
    with RunStore(args.output,spec) as store:
        summary = evaluate(store,tasks=tasks,episode=episode,workers=8)
    if summary['status'] != 'complete':
        raise RuntimeError('test incomplete; inspect infrastructure status before resuming')


if __name__ == '__main__':
    main()
