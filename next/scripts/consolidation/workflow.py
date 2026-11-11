"""One bounded, train-only correction round after the selected terminal GRPO."""
import argparse
from collections import Counter
from functools import partial
import gzip
import json
import math
from pathlib import Path
import sys

NEXT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(NEXT/'src'), str(NEXT/'scripts/reflow'), str(NEXT/'scripts/sft')]
from data import ordered, annotate, example_for, repair_point
from collect import repair
from sft_data import selected_data
from supervise_sft import selection_key
from shopping_agent.clients import LocalChatClient, ModelHubClient
from shopping_agent.cli import verify_served_checkpoint
from shopping_agent.collection.sft import acceptance_reasons, build_sft_row
from shopping_agent.protocol import StudentTokenizer, load_config, sha256, validate_execution
from shopping_agent.runs import RunStore, evaluate, write_json
from shopping_agent.runtime import run_episode
from shopping_agent.tasks import TaskPool, read_jsonl


def settings_and_policy():
    settings = json.loads((NEXT/'configs/consolidation.json').read_text())
    registry = json.loads((NEXT/settings['model_registry']).read_text())
    policy = next(p for p in registry['models'] if p['name'] == settings['initial_policy'])
    return settings, policy


def verify_initial_policy(root, policy):
    for key in ['base', 'adapter']:
        for name, digest in policy[key+'_files'].items():
            if sha256(root/policy[key]/name) != digest:
                raise ValueError(f'initial {key} identity changed: {name}')


def make_arms(repairs, successes, teacher_rows, protected, settings):
    """Share two anchor blocks; replace only the third block in the control."""
    n = len(repairs)
    if not settings['minimum_pairs'] <= n <= settings['repair_target']:
        raise ValueError('insufficient or excess verified repairs; no automatic expansion')
    repairs = ordered(repairs, settings['seed'])
    anchors = ordered(successes, settings['seed'])[:n]
    teachers = ordered(teacher_rows, settings['seed'])[:2*n]
    if len(anchors) != n or len(teachers) != 2*n:
        raise ValueError('not enough distinct successful replay examples')
    arms = {'repair': [*anchors, *teachers[:n], *repairs],
            'replay': [*anchors, *teachers]}
    for rows in arms.values():
        ids = [r['task_id'] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) & protected:
            raise ValueError('duplicate or protected training task')
    return {arm: ordered(rows, settings['seed']) for arm, rows in arms.items()}


def paired_changes(summary, baseline):
    before, after = set(baseline['strict_success_task_ids']), set(summary['strict_success_task_ids'])
    return {'gained_task_ids': sorted(after-before), 'lost_task_ids': sorted(before-after),
            'gained': len(after-before), 'lost': len(before-after),
            'retained': len(before & after), 'retention_rate': len(before & after)/len(before)}


def select_candidate(baseline, candidates):
    """A reward-only improvement never replaces the successful starting policy."""
    selection_key(baseline)
    initial = baseline['summary']
    eligible = []
    for item in candidates:
        selection_key(item)
        s = item['summary']
        if s['strict_successes'] > initial['strict_successes'] and s['mean_final_reward'] >= initial['mean_final_reward']:
            eligible.append(item)
    return max(eligible, key=lambda c: (c['summary']['strict_successes'],
        -paired_changes(c['summary'], initial)['lost'], *selection_key(c)[1:])) if eligible else baseline


def prepare(args, settings, policy, config, tokenizer, pool):
    import torch
    teacher_rows, manifest_path, _ = selected_data()
    mined = {r['task_id']: r for r in read_jsonl(args.output/'mine/raw.jsonl')}
    allowed = {r['task_id'] for r in pool.tasks(settings['mining_split'])}
    if set(mined) != allowed or any(r.get('failure_kind') == 'infrastructure' for r in mined.values()):
        raise ValueError('mining must cover the complete training split without unresolved infrastructure errors')
    repaired = read_jsonl(args.output/'repair/accepted.jsonl')
    if any(r['task_id'] not in allowed or not r['replay_verified'] or
           r['source_trajectory_id'] != mined[r['task_id']]['trajectory_id'] or
           not repair_point(mined[r['task_id']]) for r in repaired):
        raise ValueError('repair does not refer to a verified current-policy training failure')
    successes = []
    for row in mined.values():
        if acceptance_reasons(row)[0]:
            example = annotate(build_sft_row(row), tokenizer, config['budgets']['context_tokens'])
            if example is not None:
                successes.append(example)
    arms = make_arms(repaired, successes, teacher_rows['train'], pool.protected_ids('train'), settings)
    report = {'status': 'complete', 'repairs': len(repaired), 'trainable_student_successes': len(successes),
              'teacher_manifest_sha256': sha256(manifest_path), 'arms': {},
              'initial_policy': policy,
              'source_files': {name: sha256(args.output/name) for name in ['mine/raw.jsonl','repair/raw.jsonl']},
              'interventions': dict(Counter(r['intervention']['reason'] for r in repaired))}
    for arm, rows in arms.items():
        prepared = args.work/arm/'prepared'
        prepared.mkdir(parents=True, exist_ok=True)
        metadata = {'status': 'complete', 'teacher': config['teacher']['expected_response_model'], 'roles': {}}
        for role, values in [('train', rows), ('validation', teacher_rows['validation'])]:
            examples = []
            for row in values:
                ex = example_for(row, tokenizer, config['budgets']['context_tokens'])
                if ex is None or len(ex['input_ids']) != row['sequence_tokens'] or sum(x != -100 for x in ex['labels']) != row['supervised_tokens']:
                    raise ValueError(f"supervision changed: {arm}/{role}/{row['task_id']}")
                examples.append({k: torch.tensor(v, dtype=torch.long) for k,v in ex.items()})
            path = prepared/f'{role}.pt'
            torch.save(examples, path)
            metadata['roles'][role] = {'examples': len(values), 'sha256': sha256(path),
                'task_ids': [r['task_id'] for r in values],
                'input_tokens': sum(r['sequence_tokens'] for r in values),
                'supervised_tokens': sum(r['supervised_tokens'] for r in values)}
        write_json(prepared/'prepared.json', metadata)
        data_path = args.output/'dataset'/f'{arm}.jsonl.gz'
        data_path.parent.mkdir(exist_ok=True)
        data_path.write_bytes(gzip.compress(('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)+'\n').encode(), mtime=0))
        report['arms'][arm] = {**metadata, 'dataset_sha256': sha256(data_path)}
    training = json.loads((NEXT/'configs/sft.json').read_text())
    batch = training['world_size']*training['gradient_accumulation_steps']
    training.update(epochs=settings['epochs'], learning_rate=settings['learning_rate'],
                    save_steps=math.ceil(3*len(repaired)/batch), save_total_limit=settings['epochs'],
                    base_model_weights=policy['base_files'], initial_adapter_files=policy['adapter_files'])
    write_json(args.output/'training-settings.json', training)
    write_json(args.output/'dataset/manifest.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['mine','repair','prepare','dev','report'])
    for key in ['models-root','work','output']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--environment-url', default='http://127.0.0.1:5736')
    parser.add_argument('--model-url', default='http://127.0.0.1:8036/v1')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--arm', choices=['replay','repair'])
    parser.add_argument('--step', type=int)
    args = parser.parse_args()
    settings, policy = settings_and_policy()
    fingerprint, config, pool = validate_execution(), load_config(), TaskPool()
    verify_initial_policy(args.models_root, policy)
    pool.validate_source()
    if args.stage == 'report':
        initial = json.loads((NEXT/settings['baseline_result']).read_text())['selected']
        baseline = {**initial, 'arm': 'initial', 'step': 0,
                    'checkpoint': str(args.models_root/policy['adapter'])}
        candidates, budgets = [], {}
        for arm in settings['arms']:
            training = json.loads((args.output/arm/'sft/train_summary.json').read_text())
            budgets[arm] = {key: training[key] for key in ['train_examples','global_step','metrics','peak_gpu_memory_gib_by_rank']}
            for checkpoint in training['epoch_checkpoints']:
                step = int(Path(checkpoint).name.split('-')[-1])
                summary = json.loads((args.output/arm/'dev'/f'step-{step}'/'summary.json').read_text())
                candidates.append({'arm': arm, 'step': step, 'checkpoint': checkpoint, 'summary': summary,
                                   'vs_initial': paired_changes(summary, baseline['summary'])})
        report = {'status':'complete', 'baseline':baseline, 'candidates':candidates,
            'selected':select_candidate(baseline,candidates), 'selection':settings['selection'],
            'training_budgets':budgets, 'data':json.loads((args.output/'dataset/manifest.json').read_text()),
            'interpretation':'post-GRPO supervised correction; not an additional GRPO gain', 'test_evaluation':'not run'}
        write_json(args.output/'summary.json', report)
        destination = NEXT/'results/consolidation/r1'
        destination.mkdir(parents=True,exist_ok=True)
        write_json(destination/'summary.json',report)
        lines = ['# GRPO 后纠错与回放结果', '', '采集、两组训练及完整开发集评测已完成；未运行新的测试集评测。', '',
                 '| 候选 | 严格成功 | Reward v3 | 新增 / 丢失成功 | 平均步骤 |', '|---|---:|---:|---:|---:|']
        for item in [baseline,*candidates]:
            s = item['summary']; changes = paired_changes(s,baseline['summary'])
            lines.append(f"| {item['arm']} step{item['step']} | {s['strict_successes']}/300 | {s['mean_final_reward']:.4f} | {changes['gained']} / {changes['lost']} | {s['average_steps']:.3f} |")
        selected = report['selected']
        lines += ['',f"按预设规则选中 **{selected['arm']} step{selected['step']}**。监督纠错收益与原GRPO收益分开报告；单种子开发集选模不能证明稳定泛化。",
                  '', '[方法、对照与局限](README.md)；[数据、训练成本及完整指标](../../results/consolidation/r1/summary.json)。HDFS是否完成以运行目录归档回执为准。']
        (NEXT/'docs/consolidation/results.md').write_text('\n'.join(lines)+'\n')
        for path in [NEXT/'docs/consolidation/README.md',NEXT/'docs/project-completion-plan.md']:
            text = path.read_text().replace('**R-consolidation 状态：运行中。**',
                '**R-consolidation 状态：采集、训练及开发集评测已完成。**')
            path.write_text(text)
        return
    tokenizer = StudentTokenizer.load(config, path=args.models_root/policy['base'])
    if args.stage == 'prepare':
        prepare(args, settings, policy, config, tokenizer, pool)
        return
    split = 'dev' if args.stage == 'dev' else settings['mining_split']
    tasks = ordered(pool.tasks(split), settings['seed'])
    if args.stage != 'dev' and {t['task_id'] for t in tasks} & pool.protected_ids('train'):
        raise ValueError('training candidates intersect protected tasks')
    adapter = args.adapter or args.models_root/policy['adapter']
    if args.stage == 'dev' and (args.arm is None or args.step is None or args.adapter is None):
        parser.error('dev requires --arm, --step and --adapter')
    destination = args.output/args.stage if args.stage != 'dev' else args.output/args.arm/'dev'/f'step-{args.step}'
    spec = {'kind': 'post_grpo_'+args.stage, 'split':split, 'formal':False,
            'execution_fingerprint':fingerprint, 'settings':settings, 'initial_policy':policy,
            'candidate_task_ids':[t['task_id'] for t in tasks]}
    if args.stage == 'repair':
        mining = json.loads((args.output/'mine/summary.json').read_text())
        if mining['status'] != 'complete' or mining['unresolved_infrastructure_tasks']:
            raise ValueError('complete student mining before teacher repairs')
        rows = {r['task_id']:r for r in read_jsonl(args.output/'mine/raw.jsonl')}
        spec.update(teacher=config['teacher'], source_sha256=sha256(args.output/'mine/raw.jsonl'))
        with RunStore(destination,spec) as store:
            result = repair(store,[rows[t['task_id']] for t in tasks],teacher=ModelHubClient(config['teacher']),
                config=config,tokenizer=tokenizer,url=args.environment_url,fingerprint=fingerprint,settings=settings)
    else:
        verify_served_checkpoint(args.model_url,'repair-base',args.models_root/policy['base'])
        verify_served_checkpoint(args.model_url,'actor',adapter)
        spec['adapter_files'] = {name:sha256(adapter/name) for name in ['adapter_model.safetensors','adapter_config.json']}
        episode = partial(run_episode,client=LocalChatClient(args.model_url,'actor'),config=config,tokenizer=tokenizer,
                          base_url=args.environment_url,fingerprint=fingerprint)
        with RunStore(destination,spec) as store:
            result = evaluate(store,tasks=tasks,episode=episode,workers=settings['workers'])
    if result['status'] != 'complete':
        raise RuntimeError(f"{args.stage} incomplete: {result['status']}; evidence preserved")


if __name__ == '__main__':
    main()
