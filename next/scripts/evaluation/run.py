"""Run fixed test candidates on two GPUs; restore keepalive and archive evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from urllib.request import Request, urlopen

from artifacts import persist
from supervise_sft import terminate
from shopping_agent.protocol import ROOT, sha256, validate_execution
from shopping_agent.runs import write_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['models-root','work','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    args.work.mkdir(parents=True,exist_ok=True)
    lock = (args.output/'.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = json.loads((ROOT/'configs/test_comparison.json').read_text())
    if validate_execution() != plan['execution_fingerprint']:
        raise ValueError('execution contract differs from fixed comparison')
    state = {'status':'starting','pid':os.getpid(),'started_at':time.time()}
    processes = []
    def publish(**values):
        state.update(values,updated_at=time.time())
        write_json(args.output/'status.json',state)
    def launch(cmd,log,**env):
        with (args.output/log).open('a') as stream:
            p = subprocess.Popen(list(map(str,cmd)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                 start_new_session=True,env={**os.environ,**env})
        processes.append(p)
        return p
    def keep(mode):
        subprocess.run([sys.executable,ROOT/'scripts/grpo/keep_gpu.py',mode,'0','1'],check=True)
    def ready(p,url):
        deadline = time.monotonic()+900
        while time.monotonic()<deadline:
            if p.poll() is not None:
                raise RuntimeError('service exited before ready')
            try:
                with urlopen(url,timeout=2):return
            except HTTPError as e:
                if e.code==405:return
                raise
            except (URLError,OSError):time.sleep(2)
        raise TimeoutError('service startup timeout')
    def interrupt(*_):
        for p in reversed(processes):terminate(p)
        raise InterruptedError('evaluation interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupt)
    try:
        publish()
        source = args.work/'source'
        source.mkdir(exist_ok=True)
        for p in HERE.glob('*.py'):shutil.copy2(p,source/p.name)
        shutil.copy2(ROOT/'configs/test_comparison.json',source/'test_comparison.json')
        persist(source)
        keep('off')
        monitor = launch([sys.executable,ROOT/'scripts/sft/monitor_gpu.py',args.output],'gpu-monitor.log')
        env = launch(['bash',ROOT/'scripts/start_environment.sh'],'environment.log',SHOPSIM_PORT='5726',
                     SHOPSIM_ENV_SLOTS='16',SHOP_RUNTIME_DIR=str(args.work/'environment'))
        env_url = 'http://127.0.0.1:5726'
        ready(env,env_url+'/api/shop_agent')
        def lane(gpu,names):
            server,previous = None,None
            try:
                for name in names:
                    model = next(m for m in plan['models'] if m['name']==name)
                    base = args.models_root/model['base']
                    port = 8026+gpu
                    url = f'http://127.0.0.1:{port}/v1'
                    if previous != model['base']:
                        terminate(server)
                        server = launch([sys.executable,'-m','vllm.entrypoints.openai.api_server',
                            '--model',base,'--served-model-name',name if not model['adapter'] else 'repair-base',
                            '--host','127.0.0.1','--port',port,'--tensor-parallel-size','1',
                            '--gpu-memory-utilization','0.35','--gdn-prefill-backend','triton',
                            '--max-num-seqs','8','--max-model-len','24576','--max-num-batched-tokens','8192',
                            '--enable-auto-tool-choice','--tool-call-parser','qwen3_coder',
                            '--enable-lora','--max-loras','1','--max-cpu-loras','2','--max-lora-rank','16'],
                            f'serve-{name}.log',CUDA_VISIBLE_DEVICES=str(gpu),VLLM_ALLOW_RUNTIME_LORA_UPDATING='True')
                        ready(server,url+'/models')
                        previous = model['base']
                    if model['adapter']:
                        request = Request(url+'/load_lora_adapter',data=json.dumps({'lora_name':name,
                            'lora_path':str(args.models_root/model['adapter'])}).encode(),
                            headers={'Content-Type':'application/json'},method='POST')
                        with urlopen(request,timeout=180) as response:response.read()
                    job = launch([sys.executable,HERE/'evaluate.py','--name',name,'--models-root',args.models_root,
                        '--output',args.output/name,'--model-url',url,'--environment-url',env_url],f'{name}.log')
                    while job.poll() is None:
                        if any(p.poll() is not None for p in [server,env,monitor]):
                            terminate(job)
                            raise RuntimeError('an evaluation service exited')
                        time.sleep(3)
                    if job.returncode:
                        raise RuntimeError(f'{name} failed; raw results retained')
                    evidence = args.work/'evaluations'/name
                    evidence.parent.mkdir(exist_ok=True)
                    shutil.copytree(args.output/name,evidence)
                    persist(evidence)
            finally:
                terminate(server)
        publish(status='evaluating')
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lane,0,['s0','grpo-400']),executor.submit(lane,1,['s-repair','grpo-500'])]
            for future in futures:future.result()
        subprocess.run([sys.executable,HERE/'report.py','--output',args.output],check=True)
        publish(status='complete')
    except Exception as exc:
        publish(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for p in reversed(processes):terminate(p)
        keep('on')
        publish(keepalive='restored on both GPUs')
        final = args.work/'evidence'/'final'
        final.parent.mkdir(exist_ok=True)
        shutil.copytree(args.output,final,ignore=shutil.ignore_patterns('keepalive','.lock'))
        persist(final)


if __name__=='__main__':
    main()
