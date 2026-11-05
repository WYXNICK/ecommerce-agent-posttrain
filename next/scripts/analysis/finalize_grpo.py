"""Publish one completed GRPO stage; large evidence stays in verified artifact storage."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--artifact-prefix', nargs='+', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    result = repo/'next/results/grpo/outcome'
    lock = (args.output/'finalize.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = args.output/'publication.json'
    if marker.exists() and json.loads(marker.read_text()).get('status') == 'complete':
        return
    while True:
        state = json.loads((args.output/'status.json').read_text())
        if state['status'] == 'failed':
            marker.write_text(json.dumps({'status': 'not_published', 'reason': state.get('error')}, indent=2)+'\n')
            return
        if state['status'] == 'complete' and state.get('keepalive') == 'restored on both GPUs':
            break
        time.sleep(30)
    deadline = time.monotonic()+900
    while True:
        requests = {p.name: json.loads(p.read_text()) for p in (args.queue/'requests').glob('*.json')
                    if json.loads(p.read_text())['relative'].startswith(tuple(prefix+'/' for prefix in args.artifact_prefix))}
        receipts = [{'artifact': row['relative'], 'request': key,
                     'devbox_verified': (args.queue/'receipts'/key).exists() and
                         json.loads((args.queue/'receipts'/key).read_text())['status']=='complete',
                     'hdfs_complete': (args.queue/'hdfs'/key).exists()}
                    for key,row in requests.items()]
        all_done = any(r['artifact'].endswith('/evidence/final') for r in receipts) and all(
            r['devbox_verified'] and r['hdfs_complete'] for r in receipts)
        if all_done or time.monotonic() > deadline:
            break
        time.sleep(30)
    archive = {'status': 'complete' if all_done else 'pending', 'artifacts': receipts,
               'queue': str(args.queue), 'retry': 'independent artifact archive daemon'}
    (result/'archive.json').write_text(json.dumps(archive, indent=2)+'\n')
    summary = json.loads((result/'summary.json').read_text())
    baseline, selected = summary['candidates'][0], summary['selected']
    before = set(baseline['summary']['strict_success_task_ids'])
    after = set(selected['summary']['strict_success_task_ids'])
    metrics = [json.loads(line) for line in (args.output/'metrics.jsonl').read_text().splitlines()]
    analysis = {'selected_step': selected['step'], 'paired_dev_gains': sorted(after-before),
                'paired_dev_losses': sorted(before-after), 'net_dev_success_gain': len(after)-len(before),
                'optimizer_updates_in_log': len(metrics),
                'trained_action_tokens': sum(m['trained_action_tokens'] for m in metrics),
                'rollout_seconds': sum(m['rollout_seconds'] for m in metrics),
                'update_seconds': sum(m['update_seconds'] for m in metrics),
                'final_test': 'not run; dev selection is not a generalization claim'}
    (result/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
    report = repo/'next/docs/grpo/results.md'
    with report.open('a') as f:
        f.write(f"\n与 SFT 同题配对：新增成功 {len(after-before)} 题、退步 {len(before-after)} 题，净变化 {len(after)-len(before):+d} 题。单种子开发集结果，不能据此声称显著泛化收益。\n")
        f.write('\n[配对与计算统计](../../results/grpo/outcome/analysis.json)；[归档清单](../../results/grpo/outcome/archive.json)。\n')
    readme = repo/'next/docs/grpo/README.md'
    text = readme.read_text().replace('执行 **500 次有效更新**；尚无可报告的 GRPO 最终成绩。',
        f"已完成 **500 次有效更新**；选中 step {selected['step']}，dev {selected['summary']['strict_successes']}/300。详见 [结果汇总](results.md)。")
    readme.write_text(text)
    paths = ['next/results/grpo/outcome', 'next/docs/grpo', 'next/docs/project-completion-plan.md']
    # Explicit path-limited commit preserves unrelated work and an existing index.
    subprocess.run(['git','add','--',*paths], cwd=repo, check=True)
    subprocess.run(['git','commit','--only','-m','Publish terminal GRPO development results and artifact inventory','--',*paths], cwd=repo, check=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    subprocess.run(['git','push','origin','HEAD'], cwd=repo, check=True)
    marker.write_text(json.dumps({'status': 'complete','commit':commit,'archive_status':archive['status']},indent=2)+'\n')


if __name__ == '__main__':
    main()
