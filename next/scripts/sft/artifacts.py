"""Commit immutable training artifacts to devbox /tmp before acknowledging a save."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def inventory(path):
    return {str(p.relative_to(path)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(path.rglob('*')) if p.is_file()}


def persist(path):
    """Worker: publish a complete snapshot, then wait for verified devbox receipt."""
    if not os.environ.get('SHOP_ARTIFACT_QUEUE'):
        return
    queue = Path(os.environ['SHOP_ARTIFACT_QUEUE'])
    root = Path(os.environ['SHOP_ARTIFACT_ROOT']).resolve()
    path = Path(path).resolve()
    relative = str(path.relative_to(root))
    files = inventory(path)
    if not files:
        raise ValueError(f'cannot persist an empty artifact: {path}')
    request = {'source': str(path), 'relative': relative, 'files': files}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    write(queue / 'requests' / f'{key}.json', request)
    receipt = queue / 'receipts' / f'{key}.json'
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if receipt.exists():
            result = json.loads(receipt.read_text())
            if result['status'] != 'complete':
                raise RuntimeError(f'devbox artifact save failed: {result}')
            print(f'ARTIFACT_SAVED {relative} -> {result["destination"]}', flush=True)
            return
        time.sleep(2)
    raise TimeoutError(f'devbox artifact save timed out: {relative}')


def receive_one(request_path, queue, destination, worker, port, worker_root):
    request = json.loads(request_path.read_text())
    source = Path(request['source'])
    relative = source.relative_to(worker_root)
    if str(relative) != request['relative']:
        raise ValueError('artifact source does not match its relative destination')
    final = destination / relative
    if final.exists():
        if inventory(final) != request['files']:
            raise ValueError(f'immutable artifact already exists with different content: {final}')
    else:
        staging = final.with_name(final.name + '.incoming')
        staging.mkdir(parents=True, exist_ok=True)
        command = ['rsync', '-a', '--delete', '-e', f'ssh -p {port} -o BatchMode=yes -o ConnectTimeout=15',
                   f'{worker}:{source}/', str(staging) + '/']
        subprocess.run(command, check=True, timeout=600)
        if inventory(staging) != request['files']:
            raise ValueError('devbox artifact SHA256/size verification failed')
        staging.replace(final)
    manifest = {**request, 'destination': str(final), 'status': 'complete',
                'verified_at': time.time(), 'hostname': os.uname().nodename}
    write(destination / '.manifests' / request_path.name, manifest)
    write(queue / 'receipts' / request_path.name, manifest)


def upload_one(manifest_path, hdfs):
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest['destination'])
    if inventory(source) != manifest['files']:
        raise ValueError('local backup changed before HDFS upload')
    target = hdfs.rstrip('/') + '/' + manifest['relative']
    staging = target + '.uploading'
    def run(*args):
        return subprocess.run(['hdfs', 'dfs', *args], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800).stdout
    if subprocess.run(['hdfs','dfs','-test','-e', target + '/artifact-manifest.json']).returncode == 0:
        archived = json.loads(run('-cat', target + '/artifact-manifest.json'))
        if archived['files'] != manifest['files']:
            raise ValueError('HDFS target belongs to another artifact')
        return
    run('-mkdir', '-p', staging)
    for child in source.iterdir():
        run('-put', '-f', str(child), staging + '/')
    listing = run('-ls', '-R', staging)
    sizes = {line.split(maxsplit=7)[7][len(staging)+1:]: int(line.split()[4])
             for line in listing.splitlines() if line.startswith('-') and not line.endswith('/artifact-manifest.json')}
    if sizes != {name: item['bytes'] for name, item in manifest['files'].items()}:
        raise ValueError('HDFS artifact file inventory/size mismatch')
    run('-put', '-f', str(manifest_path), staging + '/artifact-manifest.json')
    run('-mv', staging, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['receive', 'archive'])
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--worker')
    parser.add_argument('--port', type=int)
    parser.add_argument('--worker-root', type=Path)
    parser.add_argument('--hdfs')
    args = parser.parse_args()
    args.queue.mkdir(parents=True, exist_ok=True)
    args.destination.mkdir(parents=True, exist_ok=True)
    lock = (args.queue / (args.mode + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        pending = False
        paths = sorted((args.queue / 'requests').glob('*.json')) if args.mode == 'receive' else sorted((args.destination / '.manifests').glob('*.json'))
        for path in paths:
            receipt = args.queue / ('receipts' if args.mode == 'receive' else 'hdfs') / path.name
            if receipt.exists():
                continue
            try:
                if args.mode == 'receive':
                    receive_one(path, args.queue, args.destination, args.worker, args.port, args.worker_root)
                else:
                    upload_one(path, args.hdfs)
                    write(receipt, {'status': 'complete', 'manifest': str(path), 'destination': args.hdfs, 'at': time.time()})
            except Exception as exc:
                pending = True
                error = str(exc)
                if isinstance(exc, subprocess.CalledProcessError):
                    error += '\n' + (exc.stderr or '')[-3000:]
                write(args.queue / (args.mode + '-status.json'), {'status': 'pending', 'error': error, 'at': time.time()})
                print(error, flush=True)
                if args.mode == 'receive':
                    write(receipt, {'status': 'failed', 'error': error})
                break
        if not pending:
            write(args.queue / (args.mode + '-status.json'), {'status': 'watching', 'at': time.time(), 'pid': os.getpid()})
        time.sleep(60 if args.mode == 'archive' and not pending else 300 if args.mode == 'archive' else 2)


if __name__ == '__main__':
    main()
