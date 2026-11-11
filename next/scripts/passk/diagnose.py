"""Fixed-policy, independently reset dev trajectories and task-paired pass@k."""
import argparse
from collections import Counter
import hashlib
import json
from math import comb
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from shopping_agent.clients import LocalChatClient
from shopping_agent.evaluation.summary import _is_strict_success
from shopping_agent.protocol import ROOT, StudentTokenizer, load_config, sha256, validate_execution
from shopping_agent.runs import RunStore, evaluate, write_json
from shopping_agent.runtime import run_episode
from shopping_agent.tasks import TaskPool, read_jsonl


def load_plan(path):
    plan = json.loads(path.read_text())
    for relative, digest in plan['input_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise ValueError(f'frozen input changed: {relative}')
    if validate_execution() != plan['execution_fingerprint']:
        raise ValueError('execution protocol changed')
    models = []
    for item in plan['models']:
        registry = json.loads((ROOT/item['registry']).read_text())
        model = next(m for m in registry['models'] if m['name'] == item['name'])
        models.append({**model, 'greedy_source':item['greedy_source'], 'greedy_keys':item['greedy_keys']})
    return plan, models


class SamplingClient(LocalChatClient):
    """Reuse wire format/error handling; change only explicitly frozen decoding."""
    def __init__(self, url, model, *, task_id, replicate, seed, temperature, top_p):
        super().__init__(url, model, transport=self.send)
        self.identity = f'{seed}:{task_id}:{replicate}'
        self.temperature, self.top_p = temperature, top_p

    def call_seed(self, messages):
        return int.from_bytes(hashlib.sha256(f'{self.identity}:{len(messages)}'.encode()).digest()[:4], 'big')

    def send(self, endpoint, payload, headers, timeout):
        payload = {**payload, 'temperature':self.temperature, 'top_p':self.top_p,
                   'seed':self.call_seed(payload['messages'])}
        request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method='POST')
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def complete(self, messages, tools, max_tokens):
        reply = super().complete(messages, tools, max_tokens)
        reply.audit.update(sampling_seed=self.call_seed(messages), temperature=self.temperature, top_p=self.top_p)
        return reply


def collect(args, plan, model):
    for folder, files in [(model['base'],model['base_files']),
                          (model['adapter'],model.get('adapter_files',{}))]:
        for name, digest in files.items():
            if sha256(args.models_root/folder/name) != digest:
                raise ValueError(f'model identity mismatch: {folder}/{name}')
    with urlopen(args.model_url+'/models', timeout=30) as response:
        cards = json.load(response)['data']
    for name, relative in [(args.name,model['adapter'] or model['base']), ('student-base',model['base'])]:
        card = next(c for c in cards if c['id'] == name)
        if Path(card['root']).resolve() != (args.models_root/relative).resolve():
            raise ValueError('served policy differs from frozen model')
    pool = TaskPool(); pool.validate_source()
    tasks = pool.tasks('dev')
    if len(tasks) != plan['tasks']:raise ValueError('dev task count changed')
    config = load_config()
    tokenizer = StudentTokenizer.load(config, path=args.models_root/model['base'])
    for replicate in range(plan['samples_per_task']):
        spec = {'kind':'passk', 'split':'dev', 'formal':False, 'model':model,
                'execution_fingerprint':plan['execution_fingerprint'], 'plan_sha256':sha256(args.plan),
                'entry_sha256':sha256(Path(__file__)), 'replicate':replicate, 'sampling':plan['sampling'],
                'candidate_task_ids':[t['task_id'] for t in tasks], 'workers':plan['workers_per_model'],
                'environment_url':args.environment_url,'model_url':args.model_url}
        def episode(task, attempt_index=0):
            client = SamplingClient(args.model_url,args.name,task_id=task['task_id'],replicate=replicate,**plan['sampling'])
            row = run_episode(task,client=client,config=config,tokenizer=tokenizer,base_url=args.environment_url,
                              fingerprint=plan['execution_fingerprint'],attempt_index=attempt_index)
            row.update(replicate=replicate, sampling=plan['sampling'])
            return row
        directory = args.output/args.name/f'replicate-{replicate:02d}'
        with RunStore(directory,spec) as store:
            for _ in range(plan['infrastructure_passes']):
                summary = evaluate(store,tasks=tasks,episode=episode,workers=plan['workers_per_model'])
                if summary['status'] != 'infrastructure_error':break
        if summary['status'] != 'complete':
            raise RuntimeError(f'incomplete replicate: {directory}; no partial pass@k report')


def task_values(counts, n, k):
    if not 1 <= k <= n or any(c < 0 or c > n for c in counts):
        raise ValueError('invalid sample counts')
    return np.array([1-comb(n-int(c),k)/comb(n,k) if n-int(c)>=k else 1. for c in counts])


def estimate(values, indices):
    samples = values[indices].mean(axis=1)
    return {'estimate':float(values.mean()), 'ci95_task_bootstrap':np.quantile(samples,[.025,.975]).tolist()}


def complete_rows(directory, ids, replicate, fingerprint):
    summary = json.loads((directory/'summary.json').read_text())
    if summary['status'] != 'complete':raise ValueError(f'incomplete replicate: {directory}')
    latest = {}
    for row in read_jsonl(directory/'raw.jsonl'):
        task_id = row['task_id']
        if task_id not in ids or row['replicate'] != replicate or row['execution_fingerprint'] != fingerprint:
            raise ValueError('trajectory identity mismatch')
        if task_id in latest and latest[task_id]['failure_kind'] != 'infrastructure':
            raise ValueError('a completed model outcome was resampled')
        latest[task_id] = row
    if set(latest) != set(ids) or any(r['failure_kind']=='infrastructure' for r in latest.values()):
        raise ValueError('missing or unresolved infrastructure trajectories')
    return latest


def report(args, plan, models):
    ids = sorted(t['task_id'] for t in TaskPool().tasks('dev'))
    n = plan['samples_per_task']
    indices = np.random.default_rng(plan['bootstrap_seed']).integers(0,len(ids),(plan['bootstrap_resamples'],len(ids)))
    result = {'status':'complete','plan_sha256':sha256(args.plan),'tasks':len(ids),'samples_per_task':n,
              'sampling':plan['sampling'],'models':{},'comparisons':{},
              'uncertainty':'Task-cluster percentile bootstrap, paired task indices; one fixed checkpoint per model, n=8; no training-seed uncertainty.',
              'scope':'dev diagnostic; not test or deployed retry/reranking performance'}
    values, counts_by_model = {}, {}
    for model in models:
        name = model['name']; counts = Counter(); rewards = []; steps = []; evidence = {}; usage = Counter()
        for replica in range(n):
            directory = args.output/name/f'replicate-{replica:02d}'
            spec = json.loads((directory/'run.json').read_text())
            if spec['plan_sha256'] != sha256(args.plan) or spec['model'] != model:
                raise ValueError('replicate uses another model or plan')
            latest = complete_rows(directory,ids,replica,plan['execution_fingerprint'])
            for task_id,row in latest.items():
                counts[task_id] += int(_is_strict_success(row))
                rewards.append(row['final_reward']); steps.append(len(row['steps']))
            s = json.loads((directory/'summary.json').read_text())
            usage.update(s['usage']['reported_tokens'])
            evidence[str(replica)] = {p:sha256(directory/p) for p in ['run.json','raw.jsonl','summary.json']}
        counts_by_model[name] = np.array([counts[t] for t in ids])
        values[name] = {k:task_values(counts_by_model[name],n,k) for k in plan['k']}
        greedy = json.loads((ROOT/model['greedy_source']).read_text())
        for key in model['greedy_keys']:greedy=greedy[key]
        if greedy['expected_tasks'] != len(ids) or greedy['completed_tasks'] != len(ids):
            raise ValueError('greedy reference incomplete')
        greedy_ids = set(greedy['strict_success_task_ids'])
        if not greedy_ids <= set(ids):raise ValueError('greedy reference split mismatch')
        result['models'][name] = {'pass_at_k':{str(k):estimate(v,indices) for k,v in values[name].items()},
            'success_count_histogram':{str(c):sum(counts[t]==c for t in ids) for c in range(n+1)},
            'per_task_successes':{str(t):counts[t] for t in ids},'mean_reward':float(np.mean(rewards)),
            'mean_steps':float(np.mean(steps)),'reported_tokens_including_retries':dict(usage),
            'greedy_reference':{'successes':len(greedy_ids),'rate':len(greedy_ids)/len(ids),
                                'source':model['greedy_source'],'temperature':0},
            'greedy_failures_reached_by_sampling':sum(t not in greedy_ids and counts[t]>0 for t in ids),
            'greedy_successes_missed_by_sampling':sum(t in greedy_ids and counts[t]==0 for t in ids),
            'evidence_sha256':evidence}
    for before,after in plan['comparisons']:
        a,b = counts_by_model[before],counts_by_model[after]
        result['comparisons'][before+'__'+after] = {
            'delta_pass_at_k':{str(k):estimate(values[after][k]-values[before][k],indices) for k in plan['k']},
            'observed_coverage_gained':int(np.sum((a==0)&(b>0))),
            'observed_coverage_lost':int(np.sum((a>0)&(b==0))),
            'success_count_transition_matrix':[[int(np.sum((a==i)&(b==j))) for j in range(n+1)] for i in range(n+1)],
            'interpretation':'finite-sample observations, not proof of capability acquisition/forgetting'}
    write_json(args.output/'summary.json',result)
    destination=ROOT/'results/passk/dev300';destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'summary.json',result)
    lines=['# 四模型 dev300 多次采样诊断结果','',
           '固定每题8次独立执行；temperature=1、top_p=1。贪心参考与随机采样分别报告，未执行test1000。','',
           '| 模型 | 贪心参考 | pass@1 | pass@2 | pass@4 | pass@8 | 0/8任务 | 8/8任务 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in result['models'].items():
        cells=[name,f"{m['greedy_reference']['rate']:.2%}"]
        cells += [f"{m['pass_at_k'][str(k)]['estimate']:.2%}" for k in plan['k']]
        cells += [str(m['success_count_histogram'][str(c)]) for c in [0,n]]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## 配对差异（百分点，95%任务bootstrap区间）','', '| 对照 | Δpass@1 | Δpass@8 |','|---|---:|---:|']
    for name,c in result['comparisons'].items():
        cells=[name]
        for k in ['1','8']:
            s=c['delta_pass_at_k'][k];lo,hi=s['ci95_task_bootstrap']
            cells.append(f"{100*s['estimate']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '区间按任务成组重采样，不代表多训练种子复现；未做多重比较校正。0/8不证明完全不会，pass@8不等于真实部署成功率。',
              '', '[完整计数、置信区间与证据](../../results/passk/dev300/summary.json) · [方案](README.md)']
    (ROOT/'docs/passk/results.md').write_text('\n'.join(lines)+'\n')
    for p in [ROOT/'docs/passk/README.md',ROOT/'docs/project-completion-plan.md']:
        p.write_text(p.read_text().replace('**Pass@k 诊断状态：运行中。**','**Pass@k 诊断状态：已完成。**'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['collect','report'])
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/passk.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models-root',type=Path)
    parser.add_argument('--name');parser.add_argument('--model-url');parser.add_argument('--environment-url')
    args=parser.parse_args();plan,models=load_plan(args.plan)
    if args.stage=='report':report(args,plan,models)
    else:collect(args,plan,next(m for m in models if m['name']==args.name))


if __name__=='__main__':main()
