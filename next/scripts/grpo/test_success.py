"""Success signs, frozen failure evidence, and the actual bounded sampling path."""
from copy import deepcopy
import gzip
from itertools import product
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from constraint_reward import constrained_return
from rollout import collect_update, group_advantages, training_signal
from settings import load_settings, starting_candidate
from success_reward import success_return, success_group_filter
from test_constraints import purchase
from train import validate_resume_settings


def failure():
    return {'failure_kind': 'model', 'done': False, 'status': 'response_budget_exhausted',
            'final_reward': 0, 'terminal_result': {}}


class SuccessTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings('success')

    def test_recipe_retains_constraints_and_training_contract(self):
        previous = load_settings('constraints')
        self.assertEqual({k for k in previous if previous[k] != self.settings[k]}, {'version'})
        self.assertEqual(set(self.settings)-set(previous), {'success_objective'})
        self.assertEqual(starting_candidate('success'), starting_candidate('constraints'))
        with self.assertRaises(ValueError):
            validate_resume_settings(previous, self.settings)

    def test_mixed_groups_have_correct_signs_at_failure_range_boundaries(self):
        # All arrangements, including one success versus the largest admissible
        # failure and two smallest failures: the difficult sign boundary.
        for size in (2, 4, 8):
            for successes in range(1, size):
                for failures in product([-.85, -.65, .25, 1.0], repeat=size-successes):
                    rows = [purchase() for _ in range(successes)] + [failure() for _ in failures]
                    old = [1.0]*successes + list(failures)
                    rewards = [success_return(r, v, size, -.85) for r, v in zip(rows, old)]
                    advantages = group_advantages(rewards)
                    self.assertTrue(all(a > 0 for a in advantages[:successes]))
                    self.assertTrue(all(a < 0 for a in advantages[successes:]))
                    self.assertAlmostEqual(sum(advantages), 0)

    def test_no_success_and_no_failure_are_not_update_groups(self):
        self.assertEqual(success_group_filter([failure(), purchase('黑色')]), 'no_success')
        self.assertEqual(success_group_filter([purchase(), purchase()]), 'all_success')
        self.assertIsNone(success_group_filter([purchase(), failure()]))

    def test_success_requires_complete_valid_terminal_not_reward_number(self):
        for field in ('done', 'over'):
            row = purchase();row['terminal_result'][field] = False
            self.assertLessEqual(success_return(row, 1, 4, -.85), 0)
        row = purchase();row['terminal_result']['reward_detail']['reward_valid'] = False
        self.assertLessEqual(success_return(row, 1, 4, -.85), 0)
        self.assertLessEqual(success_return(failure(), 1, 4, -.85), 0)

    def test_constraint_evidence_and_unverifiable_filter_are_preserved(self):
        for row in [purchase('黑色'), purchase('黑色', price_resolution={'status':'unverifiable','price':None}),
                    purchase(price_resolution={'status':'unverifiable','price':None})]:
            before = deepcopy(row)
            old = training_signal(row, -.65)
            constrained, _ = constrained_return(row, old, self.settings['constraint_reward'])
            value = success_return(row, constrained, 4, -.85)
            self.assertEqual(value, None if constrained is None else -.25)
            self.assertEqual(row, before)
        self.assertIsNone(success_return({**purchase(), 'failure_kind':'infrastructure'}, None, 4, -.85))
        self.assertLess(success_return(failure(), -.85, 4, -.85), success_return(failure(), -.65, 4, -.85))
        with self.assertRaises(ValueError):
            success_return(failure(), -1, 4, -.85)

    def test_real_sampler_filters_resamples_and_assigns_the_same_returns(self):
        scenarios = [
            [failure(), purchase('黑色'), failure(), failure()],
            [purchase()]*4,
            [purchase(), purchase('黑色'), failure(), purchase()],
            [{**purchase(), 'failure_kind':'infrastructure'}, purchase(), failure(), purchase()],
            [purchase(), purchase('黑色',price_resolution={'status':'unverifiable','price':None}), failure(), purchase()],
            [purchase()]*4,
        ]
        seen = {}
        def episode(task, **kwargs):
            index = seen.get(task['task_id'], 0);seen[task['task_id']] = index+1
            return {**deepcopy(scenarios[task['task_id']][index]), 'task_id':task['task_id']}
        class Executor:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def map(self, fn, jobs): return map(fn, jobs)
        with tempfile.TemporaryDirectory() as tmp, patch('rollout.run_episode', side_effect=episode), patch('rollout.ThreadPoolExecutor', Executor):
            rows, diag = collect_update(iter({'task_id':i} for i in range(6)),
                client_factory=lambda _:SimpleNamespace(turns=[{'response_ids':[1]}],cost={}),
                config={},tokenizer=None,env_url='',fingerprint='', settings=self.settings,
                generation=1,evidence=Path(tmp))
            with gzip.open(Path(tmp)/'generation-000001.jsonl.gz','rt') as f:
                logged = [json.loads(x) for x in f]
        self.assertEqual(diag['generation_batches'], 3)
        self.assertEqual(diag['counts']['filtered_no_success'], 1)
        self.assertEqual(diag['counts']['filtered_all_success'], 2)
        self.assertEqual(diag['counts']['filtered_invalid'], 1)
        self.assertEqual(diag['counts']['failure_positive_advantages'], 0)
        self.assertEqual(diag['counts']['success_nonpositive_advantages'], 0)
        self.assertEqual(len(logged), 24)
        self.assertEqual({r['task_id'] for r in rows}, {2,4})
        for start in (0,4):
            group = rows[start:start+4]
            self.assertEqual([r['advantage'] for r in group],group_advantages([r['training_reward'] for r in group]))
            self.assertEqual(group[1]['training_reward'], -.25)
            self.assertEqual(group[1]['constraint_training_reward'], -.85)
            self.assertLess(group[2]['advantage'], 0)
        self.assertIsNone(rows[5]['original_training_reward'])


if __name__ == '__main__':
    unittest.main()
