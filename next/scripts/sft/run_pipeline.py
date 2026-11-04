"""Run frozen SFT and paired data-reflow training with development-set selection."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from sft_data import NEXT
from supervise_sft import terminate
from shopping_agent.runs import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'work', 'output', 'keep-on', 'keep-off'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'starting', 'pid': os.getpid(), 'hostname': os.uname().nodename,
             'work': str(args.work), 'started_at': datetime.now(timezone.utc).isoformat(),
             'storage_queue': os.environ['SHOP_ARTIFACT_QUEUE'], 'test_evaluation': 'not scheduled'}
    child = None
    def publish(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output / 'status.json', state)
    def stop(*_):
        raise InterruptedError('stop requested')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    def run(stage, command):
        nonlocal child
        publish(status='running', stage=stage)
        with (args.output / f'{stage}.log').open('a') as stream:
            child = subprocess.Popen(list(map(str, command)), stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            publish(child_pid=child.pid)
            if child.wait():
                raise RuntimeError(f'{stage} stopped; inspect {stage}.log and stage/status.json')
        child = None
    common = ['--keep-on', args.keep_on, '--keep-off', args.keep_off, '--archive', args.archive]
    try:
        run('sft', [sys.executable, NEXT/'scripts/sft/supervise_sft.py', '--model', args.model,
                   '--work-dir', args.work/'s0', '--output', args.output/'sft', *common])
        summary_path = args.output/'sft/summary.json'
        selected = json.loads(summary_path.read_text())['selected']
        run('reflow', [sys.executable, NEXT/'scripts/reflow/run.py', '--model', selected['model'],
                      '--work', args.work/'reflow', '--output', args.output/'reflow',
                      '--s0-summary', summary_path, '--frozen-data', NEXT/'data/reflow/r1', *common])
        result = json.loads((args.output/'reflow/summary.json').read_text())
        publish(status='complete', selected_model=result['selected_model'],
                selected_for_next_stage=result['selected_for_next_stage'])
    except Exception as exc:
        publish(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        terminate(child)
        subprocess.run(['bash', str(args.keep_on), '0', '1'], check=True)
        publish(keepalive='restored on both GPUs')
    return 0 if state['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
