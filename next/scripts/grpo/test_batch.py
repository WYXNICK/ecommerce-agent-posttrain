"""Check the larger effective batch without changing the terminal objective."""
import unittest

from settings import load_settings, starting_candidate
from test_stable import StableTests
from train import validate_resume_settings


class BatchTests(unittest.TestCase):
    sample = StableTests.sample

    def test_recipe_keeps_starting_policy_and_objective(self):
        baseline, batch = load_settings(), load_settings('batch')
        self.assertEqual({k for k in baseline if baseline[k] != batch[k]},
                         {'version', 'groups_per_update', 'max_generation_batches'})
        self.assertEqual(set(batch) - set(baseline), {'sampling_groups_per_batch'})
        self.assertEqual(starting_candidate('batch'), starting_candidate('outcome'))
        self.assertEqual(batch['updates']*batch['groups_per_update']*batch['rollouts_per_group'], 8000)
        self.assertEqual(250*batch['groups_per_update']*batch['rollouts_per_group'], 4000)
        with self.assertRaises(ValueError):
            validate_resume_settings(baseline, batch)

    def test_full_batch_preserves_groups_and_bounded_concurrency(self):
        rows, diag, concurrency, evidence = self.sample(
            {0:'equal', 1:'infrastructure'}, load_settings('batch'))
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r['task_id'] for r in rows}), 4)
        self.assertEqual(concurrency, [8]*3)
        self.assertEqual(len(evidence), 24)
        self.assertEqual(diag['counts']['filtered_equal'], 1)
        self.assertEqual(diag['counts']['filtered_invalid'], 1)
        self.assertEqual({r['policy_generation'] for r in rows}, {1})
        for i in range(0, 16, 4):
            self.assertEqual(len({r['task_id'] for r in rows[i:i+4]}), 1)
            self.assertAlmostEqual(sum(r['advantage'] for r in rows[i:i+4]), 0.)

    def test_incomplete_batch_never_updates(self):
        cfg={**load_settings('batch'),'max_generation_batches':2}
        rows,diag,concurrency,_=self.sample({0:'equal'}, cfg)
        self.assertEqual(rows, [])
        self.assertEqual(diag['counts']['discarded_partial_groups'], 3)
        self.assertEqual(concurrency, [8,8])


if __name__ == '__main__':
    unittest.main()
