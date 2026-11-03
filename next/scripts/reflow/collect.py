"""Mine S0 failures, replay repairs, collect a domain-matched ordinary-data control."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from data import NEXT, ordered, repair_point, ReplayClient, repaired_row, annotate, example_for
from sft_data import selected_data
from shopping_agent.clients import LocalChatClient, ModelHubClient
from shopping_agent.cli import checkpoint_manifest, verify_served_checkpoint
from shopping_agent.protocol import StudentTokenizer, load_config, validate_execution, sha256
from shopping_agent.runs import RunStore, evaluate, collect, write_json, usage_summary, stop_on_signal
from shopping_agent.runtime import run_episode
from shopping_agent.tasks import TaskPool, read_jsonl
from shopping_agent.collection.sft import write_jsonl


def accepted_repairs(rows, tokenizer, config):
    accepted = {}
    for row in rows:
        meta = row['repair']
        replay = SimpleNamespace(verified=meta['verified'], teacher_call_ids=meta['teacher_call_ids'],
                                 source={'trajectory_id':meta['source_trajectory_id']}, point=meta['point'])
        sft, reasons = repaired_row(row, replay)
        if sft:
            sft = annotate(sft, tokenizer, config['budgets']['context_tokens'])
            if sft is None:reasons.append('teacher_suffix_not_trainable')
        row['repair']['rejection_reasons'] = reasons
        if sft:
            accepted.setdefault(row['task_id'], sft)
    return accepted


def repair(store, sources, *, teacher, config, tokenizer, url, fingerprint, settings):
    target = settings['repair_target']
    accepted = accepted_repairs(store.rows, tokenizer, config)
    attempted = {r['task_id'] for r in store.rows if r.get('failure_kind') != 'infrastructure'}
    remaining = [r for r in sources if r['task_id'] not in attempted and r['task_id'] not in accepted and repair_point(r)]

    def episode(source):
        replay = ReplayClient(source, repair_point(source), teacher)
        row = run_episode({'task_id':source['task_id']}, client=replay, config=config, tokenizer=tokenizer,
                          base_url=url, fingerprint=fingerprint)
        row['repair'] = {'source_trajectory_id':source['trajectory_id'], 'point':replay.point,
                         'verified':replay.verified, 'teacher_call_ids':replay.teacher_call_ids}
        return row

    def publish(status):
        usage_rows = [{**r,'model_calls':[c for c in r['model_calls'] if not c.get('replayed')]} for r in store.rows]
        report = {'status':status, 'target':target, 'accepted':len(accepted), 'attempts':len(store.rows),
                  'minimum_pairs':settings['minimum_pairs'],
                  'intervention_counts':dict(Counter(r['intervention']['reason'] for r in accepted.values())),
                  'failure_counts':dict(Counter(r.get('failure_kind') or 'strict_success' for r in store.rows)),
                  'rejection_counts':dict(Counter(reason for r in store.rows for reason in r['repair'].get('rejection_reasons',[]))),
                  'usage':usage_summary(usage_rows)}
        write_jsonl(store.directory/'accepted.jsonl', accepted.values())
        write_json(store.directory/'summary.json', report)
        return report

    with stop_on_signal() as stop, ThreadPoolExecutor(max_workers=settings['workers']) as executor:
        publish('running')
        cursor = 0
        while cursor < len(remaining) and len(accepted) < target and not stop['requested']:
            batch = remaining[cursor:cursor+min(settings['workers'], target-len(accepted))]
            cursor += len(batch)
            halted = False
            # Source-order commits make selection independent of API response speed.
            for row in executor.map(episode,batch):
                accepted.update(accepted_repairs([row], tokenizer, config))
                store.append(row)
                halted |= row.get('failure_kind') == 'infrastructure'
                print(json.dumps({'task_id':row['task_id'],'status':row['status'],'accepted':len(accepted),'target':target}),flush=True)
                publish('running')
            if halted:
                return publish('infrastructure_error')
        status = 'interrupted' if stop['requested'] else 'complete' if len(accepted)>=settings['minimum_pairs'] else 'insufficient_repairs'
        return publish(status)


def prepare(args, settings, config, tokenizer, base_rows, pool):
    import torch
    repaired = read_jsonl(args.output/'repair/accepted.jsonl')
    ordinary = read_jsonl(args.output/'general/train.jsonl')
    n=len(repaired)
    if n < settings['minimum_pairs'] or len(ordinary)!=n:
        raise ValueError('paired data is incomplete')
    replay = ordered(base_rows['train'],settings['seed'])[:n]
    groups = [repaired, ordinary, base_rows['train']]
    ids = [{r['task_id'] for r in values} for values in groups]
    if any(len(s)!=len(v) or s & pool.protected_ids('train') for s,v in zip(ids,groups)) or any(ids[i]&ids[j] for i in range(3) for j in range(i)):
        raise ValueError('new data/replay has duplicates or intersects protected tasks')
    result={'status':'complete','pairs':n,'replay_task_ids':[r['task_id'] for r in replay],
            'teacher':config['teacher']['expected_response_model'],'arms':{},'files':{}}
    destination=NEXT/'data/reflow/r1'
    destination.mkdir(parents=True,exist_ok=True)
    for arm,new in [('repair',repaired),('general',ordinary)]:
        raw=('\n'.join(json.dumps(r,ensure_ascii=False) for r in new)+'\n').encode()
        path=destination/f'{arm}.jsonl.gz';path.write_bytes(gzip.compress(raw,mtime=0))
        result['files'][path.name]={'sha256':sha256(path),'rows':len(new),'uncompressed_sha256':__import__('hashlib').sha256(raw).hexdigest()}
        prepared=args.work/arm/'prepared';prepared.mkdir(parents=True,exist_ok=True)
        report={'status':'complete','teacher':result['teacher'],'roles':{}}
        for role,values in [('train',ordered([*replay,*new],settings['seed'])),('validation',base_rows['validation'])]:
            examples=[]
            for row in values:
                ex=example_for(row,tokenizer,config['budgets']['context_tokens'])
                if ex is None or len(ex['input_ids'])!=row['sequence_tokens'] or sum(t!=-100 for t in ex['labels'])!=row['supervised_tokens']:
                    raise ValueError(f"token/supervision mismatch: {arm}/{role}/{row['task_id']}")
                examples.append({k:torch.tensor(v,dtype=torch.long) for k,v in ex.items()})
            path=prepared/f'{role}.pt';torch.save(examples,path)
            report['roles'][role]={'examples':len(values),'sha256':sha256(path),'input_tokens':sum(len(x['input_ids']) for x in examples),
                                  'supervised_tokens':sum(int((x['labels']!=-100).sum()) for x in examples),
                                  'task_ids':[r['task_id'] for r in values]}
        write_json(prepared/'prepared.json',report)
        result['arms'][arm]=report['roles']
    training=json.loads((NEXT/'configs/sft.json').read_text())
    training.update(epochs=settings['epochs'],learning_rate=settings['learning_rate'],
                    save_steps=__import__('math').ceil(2*n/8),save_total_limit=settings['epochs'],
                    base_model_weights={'model.safetensors':sha256(args.model/'model.safetensors')})
    write_json(args.output/'training-settings.json',training)
    result['training_settings']=training
    result['budget_note']='Equal new/replay example counts and optimizer updates. Suffix supervision has fewer supervised tokens; actual token and GPU budgets must be reported, not claimed equal.'
    write_json(destination/'manifest.json',result)
    write_json(args.output/'data-summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['mine','repair','general','prepare'])
    for name in ['model','output','work']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--environment-url',required=True)
    p.add_argument('--model-url',default='http://127.0.0.1:8016/v1')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    settings=json.loads((NEXT/'configs/reflow.json').read_text())
    config=load_config();fingerprint=validate_execution();pool=TaskPool();pool.validate_source()
    base_rows,_,_=selected_data();tokenizer=StudentTokenizer.load(config,path=args.model)
    train_ids={r['task_id'] for r in base_rows['train']}
    tasks=ordered([r for r in pool.tasks(settings['mining_split']) if r['task_id'] not in train_ids],settings['seed'])
    if {r['task_id'] for r in tasks}&pool.protected_ids('train'):
        raise ValueError('mining intersects protected tasks')
    if args.stage=='prepare':
        prepare(args,settings,config,tokenizer,base_rows,pool);return
    spec={'execution_fingerprint':fingerprint,'kind':'reflow_'+args.stage,'candidate_task_ids':[r['task_id'] for r in tasks],
          'settings':settings,'split':'training_failures','formal':False,'checkpoint_files':checkpoint_manifest(args.model)}
    if args.stage=='mine':
        verify_served_checkpoint(args.model_url,'shopping-agent',args.model)
        client=LocalChatClient(args.model_url)
        ep=partial(run_episode,client=client,config=config,tokenizer=tokenizer,base_url=args.environment_url,fingerprint=fingerprint)
        with RunStore(args.output/'mine',spec) as store:
            result=evaluate(store,tasks=tasks,episode=ep,workers=settings['workers'])
    elif args.stage=='repair':
        sources=read_jsonl(args.output/'mine/raw.jsonl')
        latest={r['task_id']:r for r in sources}
        sources=[latest[t['task_id']] for t in tasks]
        spec['source_sha256']=sha256(args.output/'mine/raw.jsonl');spec['teacher']=config['teacher']
        with RunStore(args.output/'repair',spec) as store:
            result=repair(store,sources,teacher=ModelHubClient(config['teacher']),config=config,tokenizer=tokenizer,
                          url=args.environment_url,fingerprint=fingerprint,settings=settings)
    else:
        repairs=read_jsonl(args.output/'repair/accepted.jsonl');domains=pool.domains()
        tasks=ordered(pool.tasks(settings['general_split']),settings['seed'])
        if {r['task_id'] for r in tasks}&(pool.protected_ids('train')|train_ids):
            raise ValueError('general candidates overlap protected or S0 training tasks')
        quotas=dict(Counter(domains[r['task_id']] for r in repairs))
        # Domains absent in the repair set have a zero quota.
        quotas.update({domains[r['task_id']]:0 for r in tasks if domains[r['task_id']] not in quotas})
        spec.update(kind='collect',role='train',candidate_task_ids=[r['task_id'] for r in tasks],
                    teacher=config['teacher'],domain_quotas=quotas,target=len(repairs))
        ep=partial(run_episode,client=ModelHubClient(config['teacher']),config=config,tokenizer=tokenizer,
                   base_url=args.environment_url,fingerprint=fingerprint)
        with RunStore(args.output/'general',spec) as store:
            result=collect(store,tasks=tasks,domains=domains,quotas=quotas,tokenizer=tokenizer,config=config,
                           episode=ep,workers=settings['workers'])
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if result['status']!='complete':
        raise SystemExit(2)

if __name__=='__main__':
    main()
