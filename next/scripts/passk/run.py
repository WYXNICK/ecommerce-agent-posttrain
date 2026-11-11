"""Two independent GPU lanes for frozen dev pass@k; preserve evidence and keepalive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from threading import Event
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from artifacts import persist
from supervise_sft import terminate
from shopping_agent.protocol import ROOT, sha256
from shopping_agent.runs import write_json
from diagnose import load_plan

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['models-root','work','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/passk.json')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);args.work.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan,models=load_plan(args.plan)
    files=[*HERE.glob('*.py'),args.plan,ROOT/'scripts/start_environment.sh',
           ROOT/'scripts/grpo/keep_gpu.py',ROOT/'scripts/sft/monitor_gpu.py',
           ROOT/'scripts/sft/artifacts.py',ROOT/'scripts/sft/supervise_sft.py']
    spec={'plan':plan,'models':models,'entry_files':{str(p.relative_to(ROOT)):sha256(p) for p in files},
          'runtime_versions':{name:version(name) for name in ['torch','transformers','vllm']},
          'models_root':str(args.models_root),'work':str(args.work)}
    if (args.output/'run.json').exists() and json.loads((args.output/'run.json').read_text()) != spec:
        raise ValueError('run configuration changed')
    write_json(args.output/'run.json',spec)
    state={'status':'starting','pid':os.getpid(),'started_at':time.time()}
    processes=[];abort=Event()
    def publish(**values):
        state.update(values,updated_at=time.time());write_json(args.output/'status.json',state)
    def keep(mode,gpus=('0','1')):
        subprocess.run([sys.executable,ROOT/'scripts/grpo/keep_gpu.py',mode,*gpus],check=True)
    def launch(cmd,log,**env):
        with (args.output/log).open('a') as stream:
            p=subprocess.Popen(list(map(str,cmd)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                               start_new_session=True,env={**os.environ,**env})
        processes.append(p);return p
    def ready(p,url):
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if abort.is_set() or p.poll() is not None:raise RuntimeError('service startup stopped')
            try:
                with urlopen(url,timeout=2):return
            except HTTPError as e:
                if e.code==405:return
                raise
            except (URLError,OSError):time.sleep(2)
        raise TimeoutError('service startup timeout')
    def interrupt(*_):
        abort.set()
        for p in reversed(processes):terminate(p)
        raise InterruptedError('diagnostic interrupted; evidence retained')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupt)
    try:
        publish();keep('on')
        source=args.work/'source';source.mkdir(exist_ok=True)
        for name in {*spec['entry_files'],*plan['input_sha256']}:
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        persist(source)
        env=launch(['bash',ROOT/'scripts/start_environment.sh'],'environment.log',SHOPSIM_PORT='5746',
                   SHOPSIM_ENV_SLOTS='16',SHOP_RUNTIME_DIR=str(args.work/'environment'))
        env_url='http://127.0.0.1:5746';ready(env,env_url+'/api/shop_agent')
        monitor=launch([sys.executable,ROOT/'scripts/sft/monitor_gpu.py',args.output],'gpu-monitor.log')
        def lane(gpu,candidates):
            server=None
            try:
                for model in candidates:
                    name=model['name']
                    if abort.is_set():raise RuntimeError('other lane failed')
                    if any(sha256(ROOT/p)!=digest for p,digest in spec['entry_files'].items()):
                        raise ValueError('implementation changed during diagnostic')
                    keep('off',(str(gpu),))
                    port=8046+gpu;url=f'http://127.0.0.1:{port}/v1'
                    cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server',
                         '--model',args.models_root/model['base'],'--served-model-name','student-base']
                    if not model['adapter']:cmd += [name]
                    cmd += ['--host','127.0.0.1','--port',port,'--tensor-parallel-size','1',
                            '--gpu-memory-utilization','0.60','--gdn-prefill-backend','triton',
                            '--max-num-seqs','8','--max-model-len','24576','--max-num-batched-tokens','8192',
                            '--enable-auto-tool-choice','--tool-call-parser','qwen3_coder',
                            '--generation-config','vllm','--seed',plan['sampling']['seed']]
                    if model['adapter']:
                        cmd += ['--enable-lora','--max-loras','1','--max-lora-rank','16',
                                '--lora-modules',f"{name}={args.models_root/model['adapter']}"]
                    server=launch(cmd,f'serve-{name}.log',CUDA_VISIBLE_DEVICES=str(gpu))
                    ready(server,url+'/models')
                    job=launch([sys.executable,HERE/'diagnose.py','collect','--name',name,
                        '--models-root',args.models_root,'--output',args.output,'--plan',args.plan,
                        '--model-url',url,'--environment-url',env_url],f'{name}.log')
                    while job.poll() is None:
                        if abort.is_set() or any(p.poll() is not None for p in [server,env,monitor]):
                            terminate(job);raise RuntimeError('diagnostic service exited')
                        time.sleep(3)
                    if job.returncode:raise RuntimeError(f'{name} incomplete; inspect log')
                    terminate(server);server=None;keep('on',(str(gpu),))
                    evidence=args.work/'evaluations'/name;evidence.parent.mkdir(exist_ok=True)
                    if not evidence.exists():shutil.copytree(args.output/name,evidence)
                    persist(evidence)
            except Exception:
                abort.set();raise
            finally:
                terminate(server);keep('on',(str(gpu),))
        publish(status='evaluating')
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(lane,gpu,models[gpu::2]) for gpu in range(2)]
            for future in futures:future.result()
        subprocess.run([sys.executable,HERE/'diagnose.py','report','--output',args.output,'--plan',args.plan],check=True)
        publish(status='complete')
    except Exception as exc:
        publish(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        for p in reversed(processes):terminate(p)
        keep('on');publish(keepalive='restored on both GPUs')
        evidence=args.work/'evidence'/str(time.time_ns());evidence.parent.mkdir(exist_ok=True)
        shutil.copytree(args.output,evidence,ignore=shutil.ignore_patterns('keepalive','.lock'))
        persist(evidence);lock.close()


if __name__=='__main__':main()
