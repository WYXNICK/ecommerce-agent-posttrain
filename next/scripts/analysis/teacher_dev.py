"""Compare a complete teacher dev run with the frozen SFT selections."""
import argparse
import json
from pathlib import Path
import sys

NEXT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NEXT / 'src'))
from shopping_agent.protocol import sha256
from shopping_agent.runs import write_json
from shopping_agent.tasks import TaskPool


def read_run(directory):
    spec = json.loads((directory / 'run.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    ids = [task['task_id'] for task in TaskPool().tasks('dev')]
    if (spec['split'] != 'dev' or spec['candidate_task_ids'] != ids
            or summary['status'] != 'complete' or summary['completed_tasks'] != len(ids)
            or summary['expected_tasks'] != len(ids) or summary['unresolved_infrastructure_tasks']):
        raise ValueError(f'{directory} is not a complete, valid frozen dev evaluation')
    return spec, summary


def compare(teacher, student):
    a, b = set(teacher['strict_success_task_ids']), set(student['strict_success_task_ids'])
    return {'both_success': len(a & b),
            'teacher_only_success': sorted(a - b), 'student_only_success': sorted(b - a),
            'both_failure': teacher['expected_tasks'] - len(a | b),
            'teacher_minus_student_percentage_points': 100 * (len(a) - len(b)) / teacher['expected_tasks']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-output', type=Path, required=True)
    parser.add_argument('--pipeline-output', type=Path, required=True)
    args = parser.parse_args()
    teacher_dir = args.teacher_output / 'dev'
    teacher_spec, teacher = read_run(teacher_dir)
    sft = json.loads((args.pipeline_output / 'sft/summary.json').read_text())
    reflow = json.loads((args.pipeline_output / 'reflow/summary.json').read_text())
    directories = {'Astra': teacher_dir,
                   'S0': args.pipeline_output / f"sft/dev/step-{sft['selected']['step']}"}
    directories.update({f'S-{arm}': args.pipeline_output / f"reflow/{arm}/dev/step-{value['selected']['step']}"
                        for arm, value in reflow['arms'].items()})
    summaries, evidence, paired = {}, {}, {}
    for name, directory in directories.items():
        spec, summary = read_run(directory)
        if spec['execution_fingerprint'] != teacher_spec['execution_fingerprint']:
            raise ValueError('teacher and student execution protocols differ')
        summaries[name] = summary
        evidence[name] = {'directory': str(directory), 'sha256': {
            filename: sha256(directory / filename) for filename in ('run.json', 'raw.jsonl', 'summary.json')}}
        if name != 'Astra':
            paired[name] = compare(teacher, summary)
    result = {'status': 'complete', 'split': 'dev', 'tasks': 300,
              'execution_fingerprint': teacher_spec['execution_fingerprint'],
              'summaries': summaries, 'paired_teacher_vs_student': paired, 'evidence': evidence,
              'limitations': ['Single execution per model; student checkpoints selected on this dev set.',
                             'Teacher high private reasoning and student non-thinking inference differ in compute.',
                             'Development diagnostics only; teacher score is not a student ceiling or expected RL gain.']}
    result_dir = NEXT / 'results/teacher-evaluation/astra/dev'
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / 'student-comparison.json', result)
    lines = ['# Astra 开发集评测与学生对照', '',
             '**已完成。** 所有模型使用同一冻结 dev300、环境、守卫及可见预算；每题一次有效行为，模型失败保留在分母。', '',
             '| 模型 | 严格成功 | 平均 Reward v3 | 平均步骤 |', '|---|---:|---:|---:|']
    for name, summary in summaries.items():
        lines.append(f"| {name} | {summary['strict_successes']}/300（{summary['strict_success_rate']:.2%}） | {summary['mean_final_reward']:.4f} | {summary['average_steps']:.3f} |")
    lines += ['', '## 同题得失', '',
              '| 学生 | 共同成功 | 仅教师成功 | 仅学生成功 | 共同失败 |', '|---|---:|---:|---:|---:|']
    for name, pair in paired.items():
        lines.append(f"| {name} | {pair['both_success']} | {len(pair['teacher_only_success'])} | {len(pair['student_only_success'])} | {pair['both_failure']} |")
    pair = paired['S-repair']
    lines += ['', f"Astra 相对选中修复 SFT 的开发集成功率差为 **{pair['teacher_minus_student_percentage_points']:+.2f} 个百分点**；教师独有成功题可供分析学生能力缺口，不能直接回流为训练数据。",
              '净分差不是 GRPO 的预计收益，也不是学生能力上限。教师 high 私有推理与学生关闭 thinking 的推理成本不同；学生在本开发集选模，因此本对照属于开发诊断，不是独立泛化结论。', '',
              '## 完整性与用量', '', '300 题全部完成，无未解决的基础设施失败；请求重试与模型失败分别留档。',
              f"教师已报告 token：`{json.dumps(teacher['usage']['reported_tokens'], ensure_ascii=False)}`。未返回 usage 的请求费用不能当作已完整计费；实际金额以账单为准。", '',
              '[教师完整结果](../../../results/teacher-evaluation/astra/dev/summary.json) · [逐题配对与证据哈希](../../../results/teacher-evaluation/astra/dev/student-comparison.json) · [方法与边界](../README.md)。']
    (NEXT / 'docs/teacher-evaluation/results/astra-dev.md').write_text('\n'.join(lines) + '\n')
    path = NEXT / 'docs/teacher-evaluation/README.md'
    text = path.read_text().replace('**状态：执行中。** 在进入 GRPO 前补齐 Astra 的完整 dev300，',
        f"**状态：已完成。** Astra 完整 dev300 严格成功 **{teacher['strict_successes']}/300（{teacher['strict_success_rate']:.2%}）**，")
    text = text.replace('完整结束后在[开发集报告]', '结果及局限见[开发集报告]')
    path.write_text(text)
    path = NEXT / 'docs/project-completion-plan.md'
    text = path.read_text().replace('**当前位置：第 5 步已完成，进入第 6 步前补充 Astra dev300 能力对照（执行中）。**',
        f"**当前位置：第 5 步及 Astra dev300 能力对照均已完成；下一步为第 6 步 GRPO。** Astra 开发集严格成功 {teacher['strict_successes']}/300（{teacher['strict_success_rate']:.2%}），详见[师生对照](teacher-evaluation/results/astra-dev.md)。")
    text = text.replace('复用学生已有 dev300 结果，补测教师同一开发集以诊断差距；', '已复用学生 dev300 结果完成教师同题对照；')
    path.write_text(text)
    print(json.dumps({'status': 'complete', 'teacher_successes': teacher['strict_successes'],
                      'paired': paired}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
