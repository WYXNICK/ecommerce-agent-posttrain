"""A save is acknowledged only after a complete, immutable devbox copy exists."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/sft'))
from artifacts import inventory, receive_one, write


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.worker = self.root/'worker'
        self.source = self.worker/'s0/checkpoints/checkpoint-125'
        self.source.mkdir(parents=True)
        for name in ['adapter_model.safetensors', 'optimizer.pt', 'rng_state_0.pth', 'rng_state_1.pth']:
            (self.source/name).write_bytes(name.encode())
        self.queue, self.destination = self.root/'queue', self.root/'master'
        self.request = self.queue/'requests/test.json'
        write(self.request, {'source':str(self.source), 'relative':'s0/checkpoints/checkpoint-125',
                            'files':inventory(self.source)})

    def receive(self, copy):
        with patch('artifacts.subprocess.run', side_effect=copy):
            receive_one(self.request, self.queue, self.destination, 'worker', 22, self.worker)

    def copy(self, command, **kwargs):
        shutil.copytree(self.source, Path(command[-1]), dirs_exist_ok=True)

    def test_complete_copy_is_verified_and_idempotent(self):
        self.receive(self.copy)
        self.assertEqual(json.loads((self.queue/'receipts/test.json').read_text())['status'],'complete')
        self.receive(lambda *a, **kw: self.fail('verified artifact should not be copied twice'))
        self.assertEqual(inventory(self.destination/'s0/checkpoints/checkpoint-125'),inventory(self.source))

    def test_incomplete_copy_has_no_receipt_or_final_directory(self):
        def incomplete(command, **kwargs):
            self.copy(command)
            (Path(command[-1])/'rng_state_1.pth').unlink()
        with self.assertRaisesRegex(ValueError, 'verification failed'):
            self.receive(incomplete)
        self.assertFalse((self.queue/'receipts/test.json').exists())
        self.assertFalse((self.destination/'s0/checkpoints/checkpoint-125').exists())

    def test_existing_artifact_cannot_be_overwritten(self):
        self.receive(self.copy)
        (self.destination/'s0/checkpoints/checkpoint-125/optimizer.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'immutable artifact'):
            self.receive(self.copy)


if __name__ == '__main__':
    unittest.main()
