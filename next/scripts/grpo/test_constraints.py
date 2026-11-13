"""Constraint semantics and the real GRPO sampling path; no GPU or API calls."""
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from web_agent_site.engine.reward import evaluate_purchase
from web_agent_site.engine.reward_features import compile_reward_features

from constraint_reward import constrained_return
from rollout import collect_update, group_advantages, training_signal
from settings import load_settings, starting_candidate
from train import eval_dev, validate_resume_settings
from shopping_agent.evaluation.summary import _is_strict_success, summarize_trajectories


PRODUCT = {'asin': 'test-product', 'title': '亚克力保护壳', 'category': '配件›保护壳',
           'attribute': ['亚克力'], 'pricing': [10],
           'customization_options': {'颜色分类': [{'value': '白色', 'price': 10},
                                                {'value': '黑色', 'price': 10}]}}
INSTRUCTION = {'instruction': '买白色亚克力保护壳，预算15元',
               'attributes': ['亚克力'], 'instruction_options': ['白色']}


def purchase(selected='白色', *, product=None, instruction=None, target=PRODUCT, **kwargs):
    goal = {'asin': target['asin'], 'price_upper': 15,
            'instruction_text': (instruction or INSTRUCTION)['instruction'],
            **compile_reward_features(instruction or INSTRUCTION, target)}
    result = evaluate_purchase(product or PRODUCT, goal,
        selected_options={} if selected is None else {'颜色分类': selected},
        **(kwargs or {'price': 10})).to_dict()
    return {'status': 'done', 'done': True, 'final_reward': result['reward'],
        'failure_kind': None if result['reward_type'] == 'gold_purchase' else
                        'model' if result['reward_valid'] else 'unverifiable',
        'terminal_result': {'done': True, 'over': True, 'goal': goal, 'reward_detail': result}}


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings('constraints')

    def score(self, row):
        return constrained_return(row, training_signal(row, -.65), self.settings['constraint_reward'])

    def test_only_training_reward_changes_recipe(self):
        base = load_settings()
        self.assertEqual({k for k in base if base[k] != self.settings[k]}, {'version'})
        self.assertEqual(set(self.settings)-set(base), {'constraint_reward'})
        self.assertEqual(starting_candidate('constraints'), starting_candidate('outcome'))
        self.assertEqual(self.settings['constraint_reward']['wrong_purchase_reward'], -.85)
        with self.assertRaises(ValueError):
            validate_resume_settings(base, self.settings)

    def test_counterfactual_wrong_and_omitted_explicit_option(self):
        good = purchase()
        self.assertTrue(_is_strict_success(good))
        self.assertEqual(self.score(good)[0], 1.)
        for selected, reason in [('黑色', 'wrong_required_option'), (None, 'omitted_required_option')]:
            with self.subTest(selected=selected):
                row = purchase(selected)
                before = deepcopy(row)
                reward, evidence = self.score(row)
                self.assertEqual(row['terminal_result']['reward_detail']['reward_type'], 'partial_alternative_purchase')
                self.assertEqual(reward, -.85)
                self.assertEqual(evidence['violations'][0]['reason'], reason)
                self.assertEqual(evidence['violations'][0]['selected'], selected)
                self.assertFalse(_is_strict_success(row))
                self.assertEqual(row, before)  # The frozen evaluation evidence is immutable.

    def test_alternatives_and_other_failures_keep_original_returns(self):
        cases = [purchase(product={**PRODUCT, 'asin': 'alternative'}),
                 purchase(price=20), purchase(product={**PRODUCT, 'attribute': []})]
        for row in cases:
            with self.subTest(reward_type=row['terminal_result']['reward_detail']['reward_type']):
                self.assertEqual(self.score(row)[0], row['final_reward'])
                self.assertEqual(self.score(row)[1]['decision'], 'unchanged')
        row = {'done': False, 'failure_kind': 'model', 'final_reward': 0}
        self.assertEqual(self.score(row)[0], -.65)

    def test_uncertain_and_inferred_options_are_not_penalized(self):
        uncertain = purchase('黑色', product={**PRODUCT, 'customization_options': {}})
        ambiguous = purchase('黑色', product={**PRODUCT, 'customization_options': {
            **PRODUCT['customization_options'], '颜色': [{'value': '白色', 'price': 10}]}})
        inferred = purchase('黑色')
        results = inferred['terminal_result']['reward_detail']['evidence']['preference_scoring']['dimensions']['key_options']['results']
        results[0]['required']['color']['source'] = 'inferred_preference'
        for row in (uncertain, ambiguous, inferred):
            self.assertEqual(self.score(row), (row['final_reward'], {
                'version': self.settings['constraint_reward']['version'], 'decision': 'unchanged',
                'restored_model_failure': False, 'violations': []}))

    def test_unavailable_required_option_is_a_bad_product_choice(self):
        row = purchase('黑色', product={**PRODUCT, 'customization_options': {
            '颜色分类': [{'value': '黑色', 'price': 10}]}})
        reward, evidence = self.score(row)
        self.assertEqual(reward, -.85)
        self.assertEqual(evidence['violations'][0]['reason'], 'purchased_product_lacks_required_option')

    def test_infrastructure_is_filtered_but_proven_model_failure_can_be_restored(self):
        row = purchase('黑色')
        row['failure_kind'] = 'infrastructure'
        self.assertIsNone(self.score(row)[0])
        self.assertEqual(self.score(row)[1]['decision'], 'filtered')
        unresolved_price = purchase('黑色', price_resolution={'status': 'unverifiable', 'price': None})
        before = deepcopy(unresolved_price)
        self.assertEqual(self.score(unresolved_price)[0], -.85)
        self.assertTrue(self.score(unresolved_price)[1]['restored_model_failure'])
        self.assertEqual(unresolved_price, before)
        price_only = purchase(price_resolution={'status': 'unverifiable', 'price': None})
        self.assertIsNone(self.score(price_only)[0])

    def test_explicit_brand_and_model_contracts_are_gated(self):
        target = {**PRODUCT, 'brand': '石头', 'shop_name': '石头旗舰店', 'title': '石头 A20 亚克力保护壳'}
        instruction = {**INSTRUCTION, 'instruction': '购买石头 A20 白色亚克力保护壳，预算15元'}
        for dimension, product in [
            ('brand', {**target, 'brand': '美的', 'shop_name': '美的旗舰店'}),
            ('model', {**target, 'title': '石头 B20 亚克力保护壳'})]:
            row = purchase(product=product, target=target, instruction=instruction)
            terminal = row['terminal_result']
            source = terminal['goal']['feature_sources'][dimension]
            dims = terminal['reward_detail']['evidence']['preference_scoring']['dimensions']
            self.assertEqual(self.score(row)[0], -.85)
            self.assertEqual(self.score(row)[1]['violations'][0]['constraint'], dimension)
            terminal['goal']['feature_sources'][dimension] = 'inferred_preference'
            self.assertEqual(self.score(row)[0], row['final_reward'])
            terminal['goal']['feature_sources'][dimension] = source
            dims[dimension]['results'][0]['actual'] = ''
            self.assertEqual(self.score(row)[0], row['final_reward'])

    def test_function_contradiction_is_different_from_missing_keywords(self):
        row = purchase(product={**PRODUCT, 'attribute': ['非亚克力']})
        self.assertEqual(self.score(row)[0], -.85)
        self.assertEqual(self.score(row)[1]['violations'][0]['reason'], 'required_function_negated')
        missing = purchase(product={**PRODUCT, 'attribute': ['塑料'], 'title': '保护壳'})
        self.assertEqual(self.score(missing)[0], missing['final_reward'])

    def test_wrong_option_no_longer_wins_against_loops(self):
        row = purchase('黑色')
        old = [row['final_reward'], -.65, -.65, -.65]
        new = [self.score(row)[0], -.65, -.65, -.65]
        self.assertGreater(group_advantages(old)[0], 0)
        self.assertLess(group_advantages(new)[0], 0)
        # This intentionally exposes the competing incentive to loop; monitor it during training.
        self.assertGreater(group_advantages(new)[1], 0)

    def test_dev_diagnostics_do_not_replace_official_scores(self):
        rows = [{**purchase('黑色' if i == 300 else '白色'), 'task_id': i} for i in range(1, 301)]
        summary = {**summarize_trajectories(range(1, 301), rows), 'status': 'complete',
                   'unresolved_infrastructure_tasks': []}
        before = deepcopy(summary)
        class Store:
            def __init__(self, output, spec):
                output.mkdir(parents=True)
                self.rows = rows
            def __enter__(self): return self
            def __exit__(self, *args): pass
        with tempfile.TemporaryDirectory() as tmp, patch('train.RunStore', Store), \
                patch('train.TaskPool') as pool, patch('train.evaluate', return_value=summary), \
                patch('train.inventory', return_value={}), patch('train.sha256', return_value='test-hash'):
            pool.return_value.tasks.return_value = [{'task_id': i} for i in range(1, 301)]
            args = SimpleNamespace(variant='constraints', output=Path(tmp), model=Path(tmp),
                                   model_url='http://unused', environment_url='http://unused')
            item = eval_dev(50, 'test', args, {}, None, 'test', Path(tmp), before)
            diagnostic = json.loads((Path(tmp)/'dev/step-50/constraints.json').read_text())
        self.assertEqual(item['summary'], before)
        self.assertEqual(item['summary']['strict_successes'], 299)
        self.assertEqual(item['constraint_counts']['violating_purchases'], 1)
        self.assertEqual(diagnostic['decisions'][0]['task_id'], 300)
        self.assertIn('constraints.json', item['evidence'])

    def test_sampling_filter_advantage_and_logs_share_the_new_reward(self):
        class Executor:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def map(self, fn, items): return map(fn, items)
        def factory(seed):
            return SimpleNamespace(seed=seed, cost=Counter(), turns=[
                {'prompt_ids': [1], 'response_ids': [2], 'old_logprobs': [-.5]}])
        def episode(task, client, **kwargs):
            index = (client.seed-self.settings['seed']) % 10
            row = purchase('黑色' if task['task_id'] == 0 or index == 0 else '白色',
                product={**PRODUCT, 'attribute': []} if task['task_id'] == 0 and index == 1 else PRODUCT)
            if task['task_id'] == 2 and index == 0:
                row['failure_kind'] = 'infrastructure'
            if task['task_id'] == 3 and index == 0:
                row = purchase('黑色', price_resolution={'status': 'unverifiable', 'price': None})
            return {**row, 'task_id': task['task_id']}
        with tempfile.TemporaryDirectory() as tmp, patch('rollout.run_episode', side_effect=episode), \
                patch('rollout.ThreadPoolExecutor', Executor):
            rows, diagnostic = collect_update(iter({'task_id': i} for i in range(4)),
                client_factory=factory, config={}, tokenizer=None, env_url='', fingerprint='',
                settings=self.settings, generation=1, evidence=Path(tmp))
            with gzip.open(Path(tmp)/'generation-000001.jsonl.gz', 'rt') as f:
                logged = [json.loads(line) for line in f]
        self.assertEqual(diagnostic['counts']['filtered_equal'], 1)
        self.assertEqual(diagnostic['counts']['filtered_invalid'], 1)
        self.assertEqual({row['task_id'] for row in rows}, {1, 3})
        self.assertEqual(len(logged), 16)
        self.assertEqual(diagnostic['counts']['constraint_accepted_trajectories'], 2)
        self.assertEqual(diagnostic['counts']['constraint_positive_advantages'], 0)
        self.assertEqual(diagnostic['counts']['constraint_restored_model_failures'], 1)
        for start in (0, 4):
            group = rows[start:start+4]
            self.assertEqual([row['training_reward'] for row in group], [-.85, 1, 1, 1])
            self.assertEqual([row['advantage'] for row in group], group_advantages([-.85, 1, 1, 1]))
        self.assertTrue(all(row['original_training_reward'] == row['final_reward']
                            for row in rows if row['original_training_reward'] is not None))
        self.assertIsNone(rows[4]['original_training_reward'])
        self.assertFalse(rows[4]['terminal_result']['reward_detail']['reward_valid'])


if __name__ == '__main__':
    unittest.main()
