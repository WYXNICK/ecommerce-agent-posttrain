"""Publish a completed stage after verified evidence archival."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--artifact-prefix',required=True)
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[3]
    lock=(args.output/'publish.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        p=args.output/'status.json'
        state=json.loads(p.read_text()) if p.exists() else {}
        if state.get('status')=='failed':return
        if state.get('status')=='complete' and state.get('keepalive')=='restored on both GPUs':break
        time.sleep(20)
    deadline=time.monotonic()+1800
    while True:
        rows=[]
        for f in (args.queue/'requests').glob('*.json'):
            request=json.loads(f.read_text())
            if not request['relative'].startswith(args.artifact_prefix+'/'):continue
            receipt=args.queue/'receipts'/f.name
            archived=args.queue/'hdfs'/f.name
            rows.append({'artifact':request['relative'],'devbox_verified':receipt.exists() and json.loads(receipt.read_text())['status']=='complete',
                         'hdfs_complete':archived.exists() and json.loads(archived.read_text())['status']=='complete'})
        complete=any(r['artifact'].endswith('/evidence/final') for r in rows) and all(r['devbox_verified'] and r['hdfs_complete'] for r in rows)
        if complete or time.monotonic()>deadline:break
        time.sleep(20)
    archive={'status':'complete' if complete else 'pending','artifacts':rows,'retry':'independent archive daemon'}
    plan=json.loads(args.plan.read_text()) if args.plan else {}
    result=repo/'next'/plan.get('result_directory','results/evaluation/test-comparison')
    (result/'archive.json').write_text(json.dumps(archive,indent=2)+'\n')
    report=repo/'next'/plan.get('report_path','docs/evaluation/results.md')
    with report.open('a') as f:f.write(f"\n归档状态：**{archive['status']}**；[归档清单](../../{plan.get('result_directory','results/evaluation/test-comparison')}/archive.json)。\n")
    paths=[plan.get('implementation','next/scripts/evaluation'), plan.get('configuration', str(args.plan.relative_to(repo)) if args.plan else 'next/configs/test_comparison.json'), str(result.relative_to(repo)), str(report.parent.relative_to(repo)), 'next/docs/project-completion-plan.md']
    subprocess.run(['git','diff','--check'],cwd=repo,check=True)
    subprocess.run(['git','add','--',*paths],cwd=repo,check=True)
    subprocess.run(['git','commit','--only','-m','Publish completed stage results and archived evidence','--',*paths],cwd=repo,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    subprocess.run(['git','push','origin','HEAD'],cwd=repo,check=True)
    (args.output/'publication.json').write_text(json.dumps({'status':'complete','commit':commit,'archive':archive['status']},indent=2)+'\n')


if __name__=='__main__':main()
