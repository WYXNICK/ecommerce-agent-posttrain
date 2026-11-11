"""Explicit execution: train-pool mining → verified repair → paired SFT → dev selection.

Run under the existing worker environment and artifact receiver, in a detached
session. No test evaluation, model merging, or change to the original GRPO run.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from workflow import NEXT, settings_and_policy, verify_initial_policy
from artifacts import persist
from supervise_sft import terminate, completed_training
from shopping_agent.protocol import sha256, validate_execution
from shopping_agent.runs import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['models-root','work','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    args.work = args.work.resolve(); args.output = args.output.resolve()
    # Formal execution uses the existing verified devbox receiver and asynchronous HDFS uploader.
    for name in ['SHOP_ARTIFACT_QUEUE','SHOP_ARTIFACT_ROOT','SHOP_KEEP_STATE','SHOP_IDLE_KEEP_ON','SHOP_IDLE_KEEP_OFF']:
        if not os.environ.get(name):
            raise ValueError(f'activate the worker/backup environment first: {name}')
    args.work.relative_to(Path(os.environ['SHOP_ARTIFACT_ROOT']).resolve())
    args.work.mkdir(parents=True,exist_ok=True); args.output.mkdir(parents=True,exist_ok=True)
    lock = (args.output/'.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings, policy = settings_and_policy()
    verify_initial_policy(args.models_root,policy)
    files = [NEXT/'configs/consolidation.json', NEXT/'configs/sft.json', NEXT/settings['model_registry'],
             NEXT/settings['baseline_result'], NEXT/'configs/data_selection.json',
             *sorted((NEXT/'scripts/consolidation').glob('*.py')),
             *sorted((NEXT/'scripts/reflow').glob('*.py')), *sorted((NEXT/'scripts/sft').glob('*.py')),
             NEXT/'scripts/grpo/keep_gpu.py', NEXT/'scripts/start_environment.sh']
    spec = {'execution_fingerprint':validate_execution(), 'settings':settings, 'policy':policy,
            'models_root':str(args.models_root.resolve()), 'work':str(args.work),
            'entry_files':{str(p.relative_to(NEXT)):sha256(p) for p in files}}
    if (args.output/'run.json').exists() and json.loads((args.output/'run.json').read_text()) != spec:
        raise ValueError('run identity changed; use an independent output directory')
    write_json(args.output/'run.json',spec)
    processes = []; services = []; server = None
    state = {'status':'starting','pid':os.getpid(),'started_at':time.time()}
    env_url, model_url = 'http://127.0.0.1:5736', 'http://127.0.0.1:8036/v1'
    def publish(**values):
        state.update(values,updated_at=time.time());write_json(args.output/'status.json',state)
    def keep(mode):
        subprocess.run([sys.executable,NEXT/'scripts/grpo/keep_gpu.py',mode,'0','1'],check=True)
    def launch(command,log,**env):
        with (args.output/log).open('a') as stream:
            p = subprocess.Popen(list(map(str,command)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                 start_new_session=True,env={**os.environ,**env})
        processes.append(p)
        return p
    def run(command,log,training=None):
        if any(sha256(NEXT/name) != digest for name,digest in spec['entry_files'].items()):
            raise ValueError('source changed during execution')
        job = launch(command,log); finished = None
        while job.poll() is None:
            if any(p.poll() is not None for p in services) or (server is not None and server.poll() is not None):
                raise RuntimeError('runtime service exited; stop rather than mislabel data')
            if training and completed_training(*training):
                finished = finished or time.monotonic()
                if time.monotonic()-finished > 90:
                    terminate(job)
                    return
            time.sleep(3)
        if job.returncode:
            raise RuntimeError(f'{log} exited {job.returncode}; resume only after checking retained evidence')
    def ready(process,url):
        deadline = time.monotonic()+900
        while time.monotonic() < deadline:
            if process.poll() is not None:raise RuntimeError('service failed to start')
            try:
                with urlopen(url,timeout=3):return
            except HTTPError as exc:
                if exc.code == 405:return
                raise
            except (OSError,URLError):time.sleep(3)
        raise TimeoutError('service startup timeout')
    def serve(adapter,label):
        nonlocal server
        keep('off')
        server = launch([sys.executable,'-m','vllm.entrypoints.openai.api_server',
            '--model',args.models_root/policy['base'],'--served-model-name','repair-base',
            '--host','127.0.0.1','--port','8036','--tensor-parallel-size','1',
            '--gpu-memory-utilization','0.60','--gdn-prefill-backend','triton',
            '--max-num-seqs','8','--max-model-len','24576','--max-num-batched-tokens','8192',
            '--enable-auto-tool-choice','--tool-call-parser','qwen3_coder',
            '--enable-lora','--max-loras','1','--max-lora-rank','16',
            '--lora-modules',f'actor={adapter}'],f'serve-{label}.log',CUDA_VISIBLE_DEVICES='0')
        ready(server,model_url+'/models')
    def unserve():
        nonlocal server
        terminate(server);server=None;keep('on')
    def stage(name,*extra):
        publish(status=name)
        run([sys.executable,Path(__file__).with_name('workflow.py'),name,
            '--models-root',args.models_root,'--work',args.work,'--output',args.output,
            '--environment-url',env_url,'--model-url',model_url,*extra],f'{name}.log')
    def interrupt(*_):
        raise InterruptedError('stop requested')
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupt)
    try:
        publish();keep('on')
        snapshot = args.work/'source';snapshot.mkdir(exist_ok=True)
        for name in spec['entry_files']:
            target = snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(NEXT/name,target)
        persist(snapshot)
        env = launch(['bash',NEXT/'scripts/start_environment.sh'],'environment.log',SHOPSIM_PORT='5736',
                     SHOP_RUNTIME_DIR=str(args.work/'environment'),SHOPSIM_ENV_SLOTS='8')
        ready(env,env_url+'/api/shop_agent')
        services = [env,launch([sys.executable,NEXT/'scripts/sft/monitor_gpu.py',args.output],'gpu-monitor.log')]
        mine = args.output/'mine/summary.json'
        if not mine.exists() or json.loads(mine.read_text())['status'] != 'complete':
            serve(args.models_root/policy['adapter'],'initial');stage('mine');unserve()
        stage('repair');stage('prepare')
        data = args.work/'dataset'
        shutil.copytree(args.output/'dataset',data,dirs_exist_ok=True);persist(data)
        training = json.loads((args.output/'training-settings.json').read_text())
        for arm in settings['arms']:
            work, out = args.work/arm, args.output/arm/'sft'
            weights = work/'checkpoints'
            if completed_training(out,work,training) is None:
                keep('off');publish(status='training',arm=arm)
                cmd = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    NEXT/'scripts/sft/train_sft.py','--model',args.models_root/policy['base'],
                    '--initial-adapter',args.models_root/policy['adapter'],'--prepared',work/'prepared',
                    '--weights',weights,'--output',out,'--settings',args.output/'training-settings.json']
                saved = sorted(weights.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
                if saved:cmd += ['--resume',saved[-1]]
                run(cmd,f'train-{arm}.log',(out,work,training));keep('on')
            trained = json.loads((out/'train_summary.json').read_text())
            for checkpoint in trained['epoch_checkpoints']:
                step = int(Path(checkpoint).name.split('-')[-1])
                result = args.output/arm/'dev'/f'step-{step}'/'summary.json'
                if result.exists() and json.loads(result.read_text())['status'] == 'complete':continue
                serve(checkpoint,f'{arm}-{step}')
                stage('dev','--adapter',checkpoint,'--arm',arm,'--step',step);unserve()
        stage('report');publish(status='complete',archive_status='devbox snapshots acknowledged; inspect HDFS receipts')
    except Exception as exc:
        publish(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for process in reversed(processes):terminate(process)
        keep('on');publish(keepalive='restored on both GPUs')
        evidence = args.work/'evidence'/str(time.time_ns())
        shutil.copytree(args.output,evidence)
        persist(evidence)
        lock.close()


if __name__ == '__main__':
    main()
