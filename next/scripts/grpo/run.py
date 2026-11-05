"""Detached GRPO supervisor: services, training, selected export, durable evidence."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
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

from artifacts import persist, inventory
from shopping_agent.protocol import ROOT, load_config, sha256, validate_execution
from shopping_agent.runs import write_json
from shopping_agent.tasks import TaskPool
from supervise_sft import terminate

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'work', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--model-port', type=int, default=8016)
    parser.add_argument('--environment-port', type=int, default=5716)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output/'run.json').exists():
        raise ValueError('output already has a run; do not mix training runs')
    state = {'status': 'starting', 'pid': os.getpid(), 'hostname': os.uname().nodename,
             'started_at': datetime.now(timezone.utc).isoformat(), 'work': str(args.work)}
    processes = []
    def publish(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output/'status.json', state)
    def stop(*_):
        raise InterruptedError('stop requested')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    def launch(command, name, **environment):
        log = (args.output/name).open('a')
        process = subprocess.Popen(list(map(str, command)), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, **environment})
        log.close()
        processes.append(process)
        return process
    def wait_ready(process, url, seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('service exited; inspect service log')
            try:
                with urlopen(url, timeout=2):
                    return
            except HTTPError as exc:
                if exc.code == 405:
                    return
                raise
            except (URLError, OSError):
                time.sleep(2)
        raise TimeoutError('service readiness timed out')
    def keep(mode):
        subprocess.run([sys.executable, HERE/'keep_gpu.py', mode, '0', '1'], check=True)
    try:
        publish()
        settings = json.loads((ROOT/'configs/grpo.json').read_text())
        fingerprint = validate_execution()
        TaskPool().validate_source()
        if sha256(args.model/'model.safetensors') != settings['base_weights_sha256']:
            raise ValueError('base weights do not match selected S-repair')
        files = [ROOT/'configs/grpo.json', *HERE.glob('*.py'), *HERE.glob('*.sh'), *HERE.glob('*.txt'), ROOT/'scripts/sft/artifacts.py',
                 ROOT/'scripts/sft/export_model.py', HERE/'keep_gpu.py', ROOT/'scripts/sft/monitor_gpu.py']
        kernel_root = Path(os.environ['SHOP_GRPO_KERNELS'])
        kernel_files = inventory(kernel_root/'wheels')
        if not kernel_files or not (kernel_root/'site').is_dir():
            raise ValueError('missing isolated FLA kernel installation')
        spec = {'actor_kernel_wheels': kernel_files, 'settings': settings, 'execution_fingerprint': fingerprint,
                'entry_files': {str(p.relative_to(ROOT)): sha256(p) for p in files},
                'model': str(args.model), 'base_files': inventory(args.model),
                'resume': str(args.resume) if args.resume else None,
                'task_pool_sha256': sha256(ROOT/'data/task_pool/grpo_train.jsonl'),
                'runtime': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'vllm', 'verl')},
                'implementation': 'veRL clipped loss, exact per-turn chat tokens, synchronous two-rank gradient sum',
                'test_evaluation': 'not scheduled'}
        write_json(args.output/'run.json', spec)
        source = args.work/'evidence'/'run-source'
        source.mkdir(parents=True)
        for path in files:
            target = source/path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        shutil.copy2(args.output/'run.json', source/'run.json')
        shutil.copytree(kernel_root/'wheels', source/'kernel-wheels')
        persist(source)
        keep('off')
        monitor = launch([sys.executable, ROOT/'scripts/sft/monitor_gpu.py', args.output], 'gpu-monitor.log')
        env_url, model_url = f'http://127.0.0.1:{args.environment_port}', f'http://127.0.0.1:{args.model_port}/v1'
        env = launch(['bash', ROOT/'scripts/start_environment.sh'], 'environment.log',
                     SHOPSIM_PORT=str(args.environment_port), SHOP_RUNTIME_DIR=str(args.work/'environment'))
        wait_ready(env, env_url+'/api/shop_agent', 180)
        serve = launch([sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', args.model, '--served-model-name', 'shopping-agent', '--host', '127.0.0.1',
            '--port', args.model_port, '--tensor-parallel-size', '1', '--gpu-memory-utilization', '0.25',
            '--gdn-prefill-backend', 'triton', '--max-num-seqs', '8', '--max-model-len', '24576',
            '--max-num-batched-tokens', '8192', '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
            '--enable-lora', '--max-loras', '1', '--max-cpu-loras', '2', '--max-lora-rank', '16',
            '--logprobs-mode', 'raw_logprobs'], 'serve.log', CUDA_VISIBLE_DEVICES='0',
            VLLM_ALLOW_RUNTIME_LORA_UPDATING='True')
        wait_ready(serve, model_url+'/models', 1200)
        publish(status='training')
        train = launch([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
            HERE/'train.py', '--model', args.model, '--work', args.work, '--output', args.output,
            '--model-url', model_url, '--environment-url', env_url,
            *(['--resume', args.resume] if args.resume else [])], 'train.log', CUDA_VISIBLE_DEVICES='0,1',
            PYTHONPATH=str(kernel_root/'site') + os.pathsep + os.environ['PYTHONPATH'])
        publish(child_pid=train.pid)
        while train.poll() is None:
            if any(p.poll() is not None for p in (env, serve, monitor)):
                raise RuntimeError('a training service exited; inspect stage logs')
            time.sleep(5)
        if train.returncode:
            raise RuntimeError('GRPO exited unsuccessfully; inspect train.log')
        terminate(serve)
        terminate(env)
        if validate_execution() != fingerprint or any(sha256(ROOT/p) != digest for p,digest in spec['entry_files'].items()):
            raise ValueError('training source changed during execution')
        publish(status='exporting')
        selection = json.loads((args.output/'selection.json').read_text())
        selected = selection['selected']
        # Keep both the best checkpoint and the final policy. If RL did not improve,
        # retain SFT as the recommendation, without relabeling it an RL improvement.
        exports = {}
        for step in sorted({selected['step'], settings['updates']} - {0}):
            target = args.work/f'model-step-{step}'
            export = launch([sys.executable, ROOT/'scripts/sft/export_model.py', '--base-model', args.model,
                '--adapter', args.work/'checkpoints'/f'step-{step}', '--output', target], f'export-{step}.log')
            if export.wait():
                raise RuntimeError('model export failed')
            persist(target)
            exports[str(step)] = {'path': str(target), 'files': inventory(target)}
        result = {**selection, 'status': 'complete', 'run': spec, 'exports': exports,
                  'selected_model': {'base_model': str(args.model),
                      'adapter': None if selected['step'] == 0 else selected['checkpoint'],
                      'format': 'base_plus_adapter', 'evaluation': 'unmerged LoRA; authoritative'},
                  'export_note': 'BF16 merged exports are convenience copies, not silently substituted for the evaluated LoRA policy',
                  'limits': 'single-seed development-set selection; final test1000 remains locked'}
        write_json(args.output/'summary.json', result)
        final = ROOT/'results/grpo/outcome'
        final.mkdir(parents=True, exist_ok=True)
        write_json(final/'summary.json', result)
        lines = ['# 终局奖励 GRPO 结果', '', '训练与开发集选模已完成；最终测试集未运行。', '',
                 '| 检查点 | dev300 严格成功 | Reward v3 | 平均动作数 |', '|---|---:|---:|---:|']
        for item in result['candidates']:
            s = item['summary']
            lines.append(f"| {'SFT 起点' if item['step']==0 else item['step']} | {s['strict_successes']}/300 | {s['mean_final_reward']:.4f} | {s['average_steps']:.3f} |")
        lines += ['', f"选中 step {selected['step']}；step 0 表示保留原 SFT。开发集用于选模，不能据此宣称测试收益。",
                  '', '[完整结果与模型身份](../../results/grpo/outcome/summary.json)。']
        (ROOT/'docs/grpo/results.md').write_text('\n'.join(lines)+'\n')
        plan = ROOT/'docs/project-completion-plan.md'
        text = plan.read_text().replace('**运行中**；[方法与状态](grpo/README.md)',
            f"**已完成**；选中 step {selected['step']}，dev {selected['summary']['strict_successes']}/300；[结果](grpo/results.md)")
        text = text.replace('第 6 步终局奖励 GRPO 正在执行。', '第 6 步终局奖励 GRPO 已完成；结果见阶段总结。')
        plan.write_text(text)
        publish(status='complete', selected_step=selected['step'], selected_model=result['selected_model'])
    except Exception as exc:
        publish(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for process in reversed(processes):
            terminate(process)
        keep('on')
        publish(keepalive='restored on both GPUs')
        # Snapshot logs only after all writers have stopped; archive uses the same
        # verified devbox receipt + independent HDFS retry queue as model saves.
        evidence = args.work/'evidence'/'final'
        if not evidence.exists():
            shutil.copytree(args.output, evidence, ignore=shutil.ignore_patterns('.lock', 'keepalive'))
            persist(evidence)


if __name__ == '__main__':
    main()
