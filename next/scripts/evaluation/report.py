"""Validate fixed test results and publish an unbiased comparison."""
import argparse
import json
import math
from pathlib import Path

from shopping_agent.evaluation.summary import summarize_trajectories
from shopping_agent.protocol import ROOT, sha256
from shopping_agent.runs import write_json
from shopping_agent.tasks import TaskPool, read_jsonl


def validate_result(directory, ids, fingerprint):
    spec = json.loads((directory/'run.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if (spec['split'] != 'test' or not spec['formal'] or spec['candidate_task_ids'] != ids
            or spec['execution_fingerprint'] != fingerprint or summary['status'] != 'complete'
            or summary['completed_tasks'] != len(ids) or summary['expected_tasks'] != len(ids)
            or summary['missing_tasks'] or summary['unresolved_infrastructure_tasks']):
        raise ValueError(f'incomplete or incompatible test result: {directory}')
    recomputed = summarize_trajectories(ids, read_jsonl(directory/'raw.jsonl'))
    for key in ['strict_successes','strict_success_task_ids','mean_final_reward','average_steps','reward_type_counts']:
        if recomputed[key] != summary[key]:
            raise ValueError(f'raw evidence disagrees with summary: {directory}/{key}')
    return summary, {n:sha256(directory/n) for n in ['run.json','summary.json','raw.jsonl']}


def paired(a,b,n):
    before,after=set(a['strict_success_task_ids']),set(b['strict_success_task_ids'])
    gain,loss=sorted(after-before),sorted(before-after)
    discordant=len(gain)+len(loss)
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(len(gain),len(loss))+1))/2**discordant) if discordant else 1.
    return {'gained':gain,'lost':loss,'net':len(gain)-len(loss),
            'delta_percentage_points':100*(len(gain)-len(loss))/n,'mcnemar_exact_two_sided_p':p}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/test_comparison.json')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    ids=[t['task_id'] for t in TaskPool().tasks('test')]
    summaries,evidence={},{}
    for item in plan['reuse']:
        directory=ROOT/item['directory']
        summary,hashes=validate_result(directory,ids,item['execution_fingerprint'])
        if hashes != item['files']:raise ValueError('reused result changed')
        summaries[item['name']]=summary
        evidence[item['name']]={'directory':str(directory),'sha256':hashes,'reused':True}
    for item in plan['models']:
        directory=args.output/item['name']
        summary,hashes=validate_result(directory,ids,plan['execution_fingerprint'])
        spec=json.loads((directory/'run.json').read_text())
        if spec['model'] != item or spec['comparison_sha256'] != sha256(args.plan):
            raise ValueError('evaluated candidate differs from frozen comparison')
        summaries[item['name']]=summary
        evidence[item['name']]={'directory':str(directory),'sha256':hashes,'reused':False}
    pairs={f'{a}_to_{b}':paired(summaries[a],summaries[b],len(ids)) for a,b in
           plan.get('pairs', [('baseline','s0'),('s0','s-repair'),('s-repair','grpo-400'),('s-repair','grpo-500'),('grpo-400','grpo-500')])}
    result={'status':'complete','split':'test','tasks':len(ids),'plan':plan,'summaries':summaries,
            'paired':pairs,'evidence':evidence,
            'limitations':plan.get('limitations', ['One execution per task/model; no multiple training seeds.',
                          'Both GRPO candidates reported; test does not select the winner.',
                          'Paired p values are descriptive, unadjusted across multiple comparisons.',
                          'Teacher high reasoning has different inference compute; not a controlled student ablation.'])}
    target=ROOT/plan.get('result_directory','results/evaluation/test-comparison')
    target.mkdir(parents=True,exist_ok=True)
    write_json(target/'summary.json',result)
    lines=[plan.get('report_title','# 冻结测试集：六模型对照'),'','**已完成：每个模型完整 1,000 题。** 复用项通过原始证据校验；新增模型按冻结名单评测。',
           '', '| 模型 | 严格成功 | 成功率 | 平均 Reward v3 | 平均步骤 | 错误购买 | 重复循环 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        counts=s['reward_type_counts']
        lines.append(f"| {name} | {s['strict_successes']}/1000 | {s['strict_success_rate']:.2%} | {s['mean_final_reward']:.4f} | {s['average_steps']:.3f} | {counts.get('wrong_purchase',0)} | {counts.get('repeat_loop',0)} |")
    lines += ['', '## 同题对照','','| 对照 | 新增成功 | 退步 | 净变化 | 配对检验 p（未作多重校正） |','|---|---:|---:|---:|---:|']
    for name,p in pairs.items():lines.append(f"| {name} | {len(p['gained'])} | {len(p['lost'])} | {p['net']:+d} | {p['mcnemar_exact_two_sided_p']:.4f} |")
    lines += ['', '## 解释边界','',plan.get('selection_note','step500 是开发集成功率规则选出的主候选；step400 是预先固定的 Reward 最优次候选。两者都报告，不根据测试分数追认最优模型。'),
              '单次执行与单训练种子不能证明多次训练的稳定收益。教师使用 high 私有推理，推理成本与学生不同。错误购买和循环等指标与成功率共同呈现，避免只看奖励或交互步数。',
              '完整汇总包含 token 用量、失败类型、成功任务 ID 和逐题得失。测试题及轨迹不用于回流训练或调参。', '',
              f"[完整指标与证据](../../{plan.get('result_directory','results/evaluation/test-comparison')}/summary.json) · [方案](README.md)。"]
    (ROOT/plan.get('report_path','docs/evaluation/results.md')).write_text('\n'.join(lines)+'\n')
    path=ROOT/plan.get('readme_path','docs/evaluation/README.md')
    path.write_text(path.read_text().replace('**状态：运行中。**','**状态：已完成。**'))
    path=ROOT/'docs/project-completion-plan.md'
    text=path.read_text().replace('S0 GRPO test1000 **运行中**','S0 GRPO test1000 **已完成**').replace('六模型冻结测试集对照正在执行','六模型冻结测试集对照已完成').replace('**运行中**；六模型冻结 test1000 对照','**本轮对照已完成**；六模型冻结 test1000 对照').replace('最终学生 test1000 对照正在执行','最终学生 test1000 对照已完成')
    path.write_text(text)
    if plan.get('readme_path'):
        path=(ROOT/plan['readme_path']).parent/'README.md'
        path.write_text(path.read_text().replace('S0 GRPO test1000 **运行中**','S0 GRPO test1000 **已完成**'))


if __name__=='__main__':main()
