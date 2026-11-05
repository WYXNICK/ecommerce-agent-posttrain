"""Use the existing idle keeper; yield to useful GPU work during active stages."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def adaptive(gpu, directory):
    import torch
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream(priority=0)
    left = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
    right = torch.randn_like(left)
    result = torch.empty_like(left)
    while True:
        # Observe useful work with this process fully idle. This prevents our
        # own filler activity from being mistaken for training throughput.
        time.sleep(1.2)
        sample = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        useful = int(sample.strip())
        active = useful <= 35
        (directory/f'gpu-{gpu}-activity.json').write_text(json.dumps({
            'time':time.time(), 'useful_gpu_percent':useful, 'filling':active, 'pid':os.getpid()}))
        if active:
            deadline = time.monotonic() + 1.2
            with torch.inference_mode(), torch.cuda.stream(stream):
                while time.monotonic() < deadline:
                    torch.mm(left, right, out=result)
                    # A single small kernel at a time, never queue seconds of work.
                    stream.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['on', 'off', 'adaptive'])
    parser.add_argument('gpus', nargs='+', type=int)
    args = parser.parse_args()
    directory = Path(os.environ['SHOP_KEEP_STATE'])
    directory.mkdir(parents=True, exist_ok=True)
    if args.mode == 'adaptive':
        adaptive(args.gpus[0], directory)
        return
    for gpu in args.gpus:
        path = directory/f'gpu-{gpu}.pid'
        if path.exists():
            pid = int(path.read_text())
            # Withdraw the health-check registration before an intentional stop.
            path.unlink()
            # Only signal our own previously recorded adaptive keeper.
            command = Path(f'/proc/{pid}/cmdline')
            if command.exists() and b'keep_gpu.py\x00adaptive' in command.read_bytes():
                os.kill(pid, signal.SIGTERM)
                for _ in range(50):
                    if not command.exists():
                        break
                    time.sleep(.1)
    script = os.environ['SHOP_IDLE_KEEP_ON' if args.mode == 'on' else 'SHOP_IDLE_KEEP_OFF']
    subprocess.run(['bash', script, *map(str, args.gpus)], check=True)
    if args.mode == 'off':
        for gpu in args.gpus:
            with (directory/f'gpu-{gpu}.log').open('a') as log:
                proc = subprocess.Popen([sys.executable, __file__, 'adaptive', str(gpu)],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES':str(gpu)}, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (directory/f'gpu-{gpu}.pid').write_text(str(proc.pid))


if __name__ == '__main__':
    main()
