"""CPU checks for append-only supervision and resumed evaluation evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from run import record_run
from train import record_evaluation


class ResumeTests(unittest.TestCase):
    def test_resume_preserves_provenance_and_rejects_changed_contract_or_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, checkpoint = root/'output', root/'checkpoint'
            output.mkdir()
            checkpoint.mkdir()
            original = {'settings': {'updates': 500}, 'resume': None,
                        'entry_files': {'run.py': 'old'}}
            (output/'run.json').write_text(json.dumps(original))
            prior = (output/'run.json').read_bytes()
            progress = {'step': 250, 'generation': 251}
            for path in [output/'progress.json', checkpoint/'training.json']:
                path.write_text(json.dumps(progress))
            spec = {**original, 'resume': str(checkpoint), 'entry_files': {'run.py': 'new'}}
            name = record_run(output, spec, checkpoint)
            self.assertEqual((output/'run.json').read_bytes(), prior)
            self.assertEqual(json.loads((output/(name+'.json')).read_text())['specification'], spec)
            with self.assertRaises(ValueError):
                record_run(output, spec, None)
            with self.assertRaises(ValueError):
                record_run(output, {**spec, 'settings': {'updates': 400}}, checkpoint)
            (output/'progress.json').write_text(json.dumps({'step': 251}))
            with self.assertRaises(ValueError):
                record_run(output, spec, checkpoint)

    def test_resumed_evaluation_writes_curves_selection_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, work = root/'output', root/'work'
            dev = output/'dev/step-250'
            dev.mkdir(parents=True)
            (work/'evidence').mkdir(parents=True)
            (dev/'raw.jsonl').write_text('{"task_id": 1}\n')
            summary = {'strict_success_rate': .75, 'mean_final_reward': .71,
                       'average_steps': 9., 'reward_type_counts': {}}
            candidate = {'step': 250, 'summary': summary,
                         'vs_starting_sft': {'gained': 2, 'lost': 3}}
            writer = Mock()
            with patch('train.persist') as persist, patch('train.selection_key', return_value=1):
                record_evaluation(candidate, [], SimpleNamespace(output=output, work=work),
                                  {'updates': 500}, writer)
            self.assertEqual(json.loads((output/'selection.json').read_text())['selected'], candidate)
            writer.add_scalar.assert_any_call('dev/strict_success_rate', .75, 250)
            self.assertEqual((work/'evidence/dev-250/raw.jsonl').read_bytes(), (dev/'raw.jsonl').read_bytes())
            persist.assert_called_once_with(work/'evidence/dev-250')


if __name__ == '__main__':
    unittest.main()
