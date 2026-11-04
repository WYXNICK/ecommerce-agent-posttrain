#!/usr/bin/env python3
"""Detached one-round reflow: mine → teacher repair/control → paired SFT → dev → archive."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

from data import NEXT
from supervise_sft import terminate, completed_training, selection_key
from artifacts import persist
from shopping_agent.cli import checkpoint_manifest
from shopping_agent.protocol import validate_execution, sha256
from shopping_agent.runs import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','work','output','keep-on','keep-off']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--archive',required=True)
    p.add_argument('--environment-port',type=int,default=5706)
    p.add_argument('--model-port',type=int,default=8016)
    p.add_argument('--frozen-data',type=Path)
    p.add_argument('--s0-summary',type=Path,default=NEXT/'outputs/sft/astra-s0/summary.json')
    args=p.parse_args();out=args.output.resolve();work=args.work.resolve()
    out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    lock=(out/'.supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec={'execution_fingerprint':validate_execution(),'model':str(args.model.resolve()),
          'model_files':checkpoint_manifest(args.model),
          'entry_files':{str(q.relative_to(NEXT)):sha256(q) for q in [NEXT/'configs/reflow.json',NEXT/'configs/sft.json',
              *sorted((NEXT/'scripts/reflow').glob('*.py')),*sorted((NEXT/'scripts/sft').glob('*.py'))]},
          's0_result_sha256':sha256(args.s0_summary),
          'frozen_data_manifest_sha256':sha256(args.frozen_data/'manifest.json') if args.frozen_data else None}
    if (out/'run.json').exists() and json.loads((out/'run.json').read_text())!=spec:
        raise ValueError('run identity changed; do not mix code/models in an active run')
    s0_result=json.loads(args.s0_summary.read_text())
    selected_model=Path(s0_result['selected']['model'])
    expected=json.loads((selected_model/'merge_manifest.json').read_text())['files']['model.safetensors']
    if spec['model_files'].get('model.safetensors')!=expected:
        raise ValueError('model is not the selected S0 checkpoint')
    write_json(out/'run.json',spec)
    state={'status':'starting','pid':os.getpid(),'hostname':os.uname().nodename,'work':str(work),
           'started_at':datetime.now(timezone.utc).isoformat(),'archive':args.archive}
    job=environment=serving=monitor=None
    stopped=False

    def publish(**changes):
        state.update(changes,updated_at=datetime.now(timezone.utc).isoformat());write_json(out/'status.json',state)
    def stop(*_):
        nonlocal stopped
        stopped=True
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    def keep(on,*gpus):
        subprocess.run(['bash',str(args.keep_on if on else args.keep_off),*map(str,gpus)],check=True)
    def launch(cmd,log,env=None):
        with (out/log).open('a') as stream:
            return subprocess.Popen(list(map(str,cmd)),env={**os.environ,**(env or {})},stdin=subprocess.DEVNULL,
                                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    def run(cmd,log,training=None):
        nonlocal job
        if stopped:raise InterruptedError('stop requested')
        if any(sha256(NEXT/name)!=digest for name,digest in spec['entry_files'].items()):
            raise ValueError('stage source changed during execution')
        job=launch(cmd,log);publish(child_pid=job.pid)
        finished=None
        while job.poll() is None:
            if stopped:raise InterruptedError('stop requested')
            if monitor.poll() is not None:raise RuntimeError('GPU monitoring exited')
            if training and completed_training(*training):
                finished=finished or time.monotonic()
                if time.monotonic()-finished>90:
                    terminate(job);job=None
                    return
            time.sleep(3)
        code=job.returncode;job=None
        if code:raise RuntimeError(f'command exited {code}; see {log}')
    def ready(proc,url):
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            if stopped or proc.poll() is not None:raise RuntimeError('service stopped during startup')
            try:
                with urlopen(url,timeout=3):return
            except HTTPError as e:
                if e.code==405:return
                raise
            except (URLError,OSError):time.sleep(3)
        raise TimeoutError('service startup timeout')
    def start_serving(model,label):
        nonlocal serving
        keep(False,0)
        serving=launch(['bash',NEXT/'scripts/serve_model.sh',model],f'serve-{label}.log',
                       {'CUDA_VISIBLE_DEVICES':'0','LLM_PORT':str(args.model_port),'TENSOR_PARALLEL_SIZE':'1'})
        ready(serving,f'http://127.0.0.1:{args.model_port}/v1/models')
    def stop_serving():
        nonlocal serving
        terminate(serving);serving=None;keep(True,0)
    def data_stage(stage):
        marker=out/stage/'summary.json'
        if stage!='prepare' and marker.exists() and json.loads(marker.read_text())['status']=='complete':return
        publish(status=stage)
        cmd=[sys.executable,NEXT/'scripts/reflow/collect.py',stage,'--model',args.model,'--work',work,'--output',out,
             '--environment-url',f'http://127.0.0.1:{args.environment_port}',
             '--model-url',f'http://127.0.0.1:{args.model_port}/v1']
        if args.frozen_data:
            cmd += ['--frozen-data',args.frozen_data]
        for attempt in range(3):
            try:
                run(cmd,f'{stage}.log');return
            except RuntimeError:
                # Retry only a recorded infrastructure interruption, never silently retry code errors.
                if not marker.exists() or json.loads(marker.read_text())['status']!='infrastructure_error' or attempt==2:raise
                time.sleep(10)
    try:
        publish();keep(True,0,1)
        monitor=launch([sys.executable,NEXT/'scripts/sft/monitor_gpu.py',out],'gpu-monitor.log')
        environment=launch(['bash',NEXT/'scripts/start_environment.sh'],'environment.log',
            {'SHOPSIM_PORT':str(args.environment_port),'SHOP_RUNTIME_DIR':str(work/'environment')})
        ready(environment,f'http://127.0.0.1:{args.environment_port}/api/shop_agent')
        mine=out/'mine/summary.json'
        if not args.frozen_data and (not mine.exists() or json.loads(mine.read_text())['status']!='complete'):
            start_serving(args.model,'s0');data_stage('mine');stop_serving()
        if not args.frozen_data:
            data_stage('repair');data_stage('general')
        data_stage('prepare')
        settings=json.loads((out/'training-settings.json').read_text())
        results={}
        for arm in ['general','repair']:
            arm_out=out/arm/'sft';arm_out.mkdir(parents=True,exist_ok=True)
            arm_work=work/arm;weights=arm_work/'checkpoints'
            if completed_training(arm_out,arm_work,settings) is None:
                publish(status='training',arm=arm);keep(False,0,1)
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                     NEXT/'scripts/sft/train_sft.py','--model',args.model,'--prepared',arm_work/'prepared',
                     '--weights',weights,'--output',arm_out,'--settings',out/'training-settings.json']
                checkpoints=sorted(weights.glob('checkpoint-*'),key=lambda q:int(q.name.split('-')[-1]))
                if checkpoints:cmd+=['--resume',checkpoints[-1]]
                run(cmd,f'train-{arm}.log',(arm_out,arm_work,settings));keep(True,0,1)
            training=json.loads((arm_out/'train_summary.json').read_text());candidates=[]
            for path in training['epoch_checkpoints']:
                checkpoint=Path(path);step=int(checkpoint.name.split('-')[-1]);model=arm_work/f'epoch-{step}'
                publish(status='exporting',arm=arm,step=step)
                if not (model/'merge_manifest.json').exists():
                    run([sys.executable,NEXT/'scripts/sft/export_model.py','--base-model',args.model,
                         '--adapter',checkpoint,'--output',model],f'export-{arm}-{step}.log')
                persist(model)
                dev=out/arm/'dev'/f'step-{step}'
                if not (dev/'summary.json').exists() or json.loads((dev/'summary.json').read_text())['status']!='complete':
                    publish(status='selecting_on_dev',arm=arm,step=step);start_serving(model,f'{arm}-{step}')
                    run([sys.executable,NEXT/'scripts/agent.py','evaluate','--split','dev','--checkpoint',model,
                         '--output',dev,'--environment-url',f'http://127.0.0.1:{args.environment_port}',
                         '--model-url',f'http://127.0.0.1:{args.model_port}/v1'],f'dev-{arm}-{step}.log')
                    stop_serving()
                summary=json.loads((dev/'summary.json').read_text())
                item={'step':step,'model':str(model),'checkpoint':str(checkpoint),'summary':summary,
                      'evidence_sha256':{q.name:sha256(q) for q in dev.glob('*') if q.suffix in {'.json','.jsonl'}}}
                selection_key(item);candidates.append(item)
            best=max(candidates,key=selection_key)
            results[arm]={'selected':best,'candidates':candidates,'training':{k:v for k,v in training.items() if k!='log_history'}}
            write_json(out/'comparison.json',results)
        s0=s0_result['selected']
        contenders={'s0':s0,**{a:v['selected'] for a,v in results.items()}}
        selected=max(contenders,key=lambda a:selection_key(contenders[a])[:3])
        summary={'status':'complete','s0':s0,'arms':results,'selected_for_next_stage':selected,
                 'selected_model':contenders[selected]['model'],'data':json.loads((out/'data-summary.json').read_text()),
                 'test_evaluation':'not run','limitations':['one round, one training seed','development selection only',
                 'updates and example counts matched; supervised token counts differ due to suffix masking']}
        write_json(out/'summary.json',summary)
        dest=NEXT/'results/data-reflow/r1';dest.mkdir(parents=True,exist_ok=True)
        write_json(dest/'summary.json',summary)
        lines=['# 第一轮失败回流结果','', '| 模型 | 开发集严格成功 | 平均奖励 | 平均步骤 |','|---|---:|---:|---:|']
        for name,item in contenders.items():
            s=item['summary'];lines.append(f"| {name} | {s['strict_successes']}/300 | {s['mean_final_reward']:.4f} | {s['average_steps']:.3f} |")
        lines+=['',f'按预定规则选择 `{selected}` 作为下一阶段候选；未运行最终测试或 GRPO。',
                '单轮、单种子结果；相同更新次数和样本数，不宣称监督 token 完全相同。',
                '[方法与边界](README.md)；[完整指标](../../results/data-reflow/r1/summary.json)。']
        (NEXT/'docs/data-reflow/results.md').write_text('\n'.join(lines)+'\n')
        if os.environ.get('SHOP_ARTIFACT_QUEUE'):
            archive_state={'status':'queued','destination':args.archive,
                           'devbox_receipts':os.environ['SHOP_ARTIFACT_QUEUE']+'/receipts',
                           'hdfs_receipts':os.environ['SHOP_ARTIFACT_QUEUE']+'/hdfs'}
            write_json(out/'archive.json',archive_state);write_json(dest/'archive.json',archive_state)
            publish(status='complete',archive_status='queued',selected=selected)
            return 0
        # Archive after useful computation; a team quota failure must not erase completed results.
        publish(status='archiving')
        archives=[args.model,Path(s0['checkpoint'])]
        for v in results.values():archives += [Path(v['selected']['model']),Path(v['selected']['checkpoint'])]
        archive_state={'status':'complete','destination':args.archive}
        try:
            run(['hdfs','dfs','-mkdir','-p',args.archive],'archive.log')
            # Each arm has its own namespace, avoiding equal-step checkpoint name collisions.
            sources=[('s0',archives[:2]),*[(a,[Path(v['selected']['model']),Path(v['selected']['checkpoint'])]) for a,v in results.items()]]
            for name,paths in sources:
                target=args.archive+'/'+name;run(['hdfs','dfs','-mkdir','-p',target],'archive.log')
                for path in paths:run(['hdfs','dfs','-put','-f',path,target+'/'],'archive.log')
            run(['hdfs','dfs','-put','-f',out,args.archive+'/evidence'],'archive.log')
        except RuntimeError as exc:
            archive_state.update(status='pending',error=str(exc))
        write_json(out/'archive.json',archive_state)
        write_json(dest/'archive.json',archive_state)
        publish(status='complete',archive_status=archive_state['status'],selected=selected)
    except Exception as exc:
        publish(status='interrupted' if stopped else 'failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        for proc in [job,serving,environment,monitor]:terminate(proc)
        keep(True,0,1);publish(keepalive='restored on both GPUs');lock.close()
    return 0 if state['status']=='complete' else 2

if __name__=='__main__':raise SystemExit(main())
