"""Behavioral verification of replay, budgets, estimator and action-only gradients."""
from collections import Counter
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tests'))
from test_execution import Client, Environment, Tokenizer, message
from local_credit import (BranchClient, PrefixEnvironment, ReplayMismatch, apply_local_credit, comparison,
                          decision_for, estimate_decision, seed_for)
from rollout import TrainingClient
from settings import load_settings
from train import balance_turns
from shopping_agent.clients import ModelReply, ModelServiceError
from shopping_agent.protocol import load_config, validate_execution
from shopping_agent.runtime import run_episode


class RecordedClient(Client):
    def __init__(self, replies):
        super().__init__(replies)
        self.turns, self.cost = [], Counter()

    def complete(self, messages, tools, max_tokens):
        self.cost.update(requests=1, prompt_tokens=10, completion_tokens=2)
        reply = super().complete(messages, tools, max_tokens)
        self.turns.append({'prompt_ids': [1, 2], 'response_ids': [3, 4], 'old_logprobs': [-.1, -.2],
            'replay': {'messages': deepcopy(messages), 'max_tokens': max_tokens, 'message': deepcopy(reply.message)}})
        return reply


class LocalCreditTests(unittest.TestCase):
    def source(self):
        client = RecordedClient([message('search_products', {'query': 'product'}, 'search'),
                                 message('buy_now', call_id='buy')])
        row = run_episode({'task_id': 1}, client=client, config=load_config(), tokenizer=Tokenizer(),
                          base_url='unused', fingerprint='test', env_factory=Environment)
        row.update(rollout_turns=client.turns, policy_generation=3, advantage=.2, training_reward=1.)
        return row

    def estimate(self, source=None, env_factory=Environment, alternative=None):
        source = source or self.source()
        alternative = alternative or {'role': 'assistant', 'content': 'stop'}
        clients = []
        def factory(seed):
            client = RecordedClient([alternative] if seed == seed_for(123, 'alternative') else [])
            clients.append((seed, client))
            return client
        log = io.StringIO()
        settings = load_settings('local')
        result = estimate_decision(source, 1, seed=123, client_factory=factory, config=load_config(),
            tokenizer=Tokenizer(), env_url='unused', fingerprint='test', settings=settings,
            local=settings['local_credit'], log=log, env_factory=env_factory)
        return result, [json.loads(line) for line in log.getvalue().splitlines()], clients

    def test_shared_runtime_replays_prefix_and_retains_model_failure_returns(self):
        result, evidence, clients = self.estimate()
        n = load_settings('local')['local_credit']['continuations']
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(result['returns'], [[1.]*n, [-.65]*n])
        self.assertEqual(result['verified_branches'], 2*n)
        self.assertEqual(result['model_failures'], n)
        self.assertEqual(result['increment'], .25)
        branches = [r for r in evidence if r['kind'] == 'local_branch']
        self.assertEqual([len(r['steps']) for r in branches], [2]*n+[1]*n)
        self.assertEqual([s for s, _ in clients[1:n+1]], [s for s, _ in clients[n+1:2*n+1]])
        self.assertEqual(result['cost']['requests'], 1)  # Fixed actions need no model calls.

    def test_hidden_reset_change_invalidates_pair_even_if_observation_matches(self):
        class Changed(Environment):
            def reset(self, task_id):
                result = super().reset(task_id)
                result['goal_options'] = {'secret': 'changed'}
                return result
        result, _, _ = self.estimate(env_factory=Changed)
        self.assertEqual(result['status'], 'invalid_pair')
        self.assertEqual(result['verified_branches'], 0)
        self.assertNotIn('increment', result)

    def test_lease_metadata_does_not_change_state_but_real_task_and_price_do(self):
        source = {'env_idx': 1, 'idx': 'slot-1-42', 'observation_state': {'price': 15}}
        replay = {**source, 'env_idx': 7, 'idx': 'slot-7-42'}
        self.assertEqual(PrefixEnvironment.stable(source), PrefixEnvironment.stable(replay))
        self.assertNotEqual(PrefixEnvironment.stable(source), PrefixEnvironment.stable({**replay, 'idx': 'slot-7-43'}))
        self.assertNotEqual(PrefixEnvironment.stable(source),
                            PrefixEnvironment.stable({**replay, 'observation_state': {'price': 16}}))
        initial = {'idx': 42, 'goal_options': ['size M'], 'instruction': 'buy'}
        cached = {**initial, 'user_persona': {'unused': True}, 'reason_key': 'unused'}
        self.assertEqual(PrefixEnvironment.stable(initial, reset=True), PrefixEnvironment.stable(cached, reset=True))

    def test_alternative_action_is_followed_by_same_policy_continuations(self):
        source, clients, log = self.source(), [], io.StringIO()
        class QueryReward(Environment):
            def step(self, action):
                result = super().step(action)
                if result['done'] and self.actions[0] == 'search[alternative]':
                    result['reward'] = -.4
                    result['reward_detail']['reward_type'] = 'wrong_purchase'
                return result
        def factory(seed):
            reply = (message('search_products', {'query': 'alternative'})
                     if seed == seed_for(321, 'alternative') else message('buy_now'))
            client = RecordedClient([reply])
            clients.append(client)
            return client
        settings = load_settings('local')
        result = estimate_decision(source, 0, seed=321, client_factory=factory, config=load_config(),
            tokenizer=Tokenizer(), env_url='unused', fingerprint='test', settings=settings,
            local=settings['local_credit'], log=log, env_factory=QueryReward)
        self.assertEqual(result['status'], 'valid')
        n = settings['local_credit']['continuations']
        self.assertEqual(result['returns'], [[1.]*n, [-.4]*n])
        self.assertEqual(result['cost']['requests'], 2*n+1)
        self.assertTrue(all(len(c.seen) == 1 for c in clients))
        self.assertTrue(all('alternative' in c.seen[0][2]['tool_calls'][0]['function']['arguments']
                            for c in clients[n+1:]))

    def test_alternative_service_failure_is_bounded_and_does_not_mutate_source(self):
        source = self.source()
        before = deepcopy(source)
        settings = load_settings('local')
        result = estimate_decision(source, 1, seed=1,
            client_factory=lambda _: RecordedClient([ModelServiceError('unavailable')]),
            config=load_config(), tokenizer=Tokenizer(), env_url='unused', fingerprint='test',
            settings=settings, local=settings['local_credit'], log=io.StringIO(), env_factory=Environment)
        self.assertEqual(result['status'], 'alternative_unavailable')
        self.assertEqual(result['cost']['requests'], 1)
        self.assertEqual(source, before)

    def test_unverifiable_branch_cannot_be_used_as_a_negative_reward(self):
        class Unverifiable(Environment):
            def step(self, action):
                result = super().step(action)
                if result['done']:
                    result['reward_detail']['reward_valid'] = False
                return result
        result, _, _ = self.estimate(env_factory=Unverifiable)
        self.assertEqual(result['status'], 'invalid_pair')
        self.assertEqual(result['unverifiable'], load_settings('local')['local_credit']['continuations'])
        self.assertNotIn('increment', result)

    def test_equal_actions_skip_without_resampling_or_branches(self):
        result, evidence, clients = self.estimate(alternative=message('buy_now', call_id='different-id'))
        self.assertEqual(result['status'], 'same_action')
        self.assertEqual(len(clients), 1)
        self.assertFalse(any(r['kind'] == 'local_branch' for r in evidence))

    def test_replay_checks_exact_budget_and_preserves_step_limit(self):
        source = self.source()
        saved = source['rollout_turns'][1]['replay']
        client = BranchClient(source['rollout_turns'], 1, saved['message'], Client([]))
        client.cursor = 1
        with self.assertRaises(ReplayMismatch):
            client.complete(saved['messages'], [], saved['max_tokens']+1)
        config = load_config()
        config['budgets']['max_steps'] = 2
        client = BranchClient(source['rollout_turns'], 1,
                             message('think', {'note': 'wait'}), Client([]))
        row = run_episode({'task_id': 1}, client=client, config=config, tokenizer=Tokenizer(),
                          base_url='unused', fingerprint='test', env_factory=Environment)
        self.assertEqual(row['status'], 'max_steps')
        self.assertEqual(len(row['steps']), 2)  # Prefix is counted; no budget reset.

    def test_q_uses_all_replicates_and_reports_paired_noise(self):
        local = load_settings('local')['local_credit']
        result = comparison([[1, 0, 1, 0], [0, 1, 0, 1]], local)
        self.assertEqual(result['increment'], 0)
        self.assertGreater(result['paired_standard_error'], 0)
        self.assertIsNone(comparison([[1, None, 1, 1], [0, 0, 0, 0]], local))

    def test_lambda_zero_is_exact_noop_including_compute(self):
        rows = [self.source()]
        before = deepcopy(rows)
        settings = load_settings('local')
        settings['local_credit']['weight'] = 0
        with tempfile.TemporaryDirectory() as directory:
            result = apply_local_credit(rows, client_factory=None, config=None, tokenizer=None,
                env_url=None, fingerprint=None, settings=settings, generation=3, evidence=Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(result, {})
        self.assertEqual(rows, before)

    def test_only_selected_original_action_receives_increment(self):
        rows = [self.source()]
        rows[0]['rollout_turns'][1]['local_increment'] = -.25
        partitions, _ = balance_turns(rows, 2)
        actual = {id(t): a for p in partitions for t, a in p}
        self.assertEqual(actual[id(rows[0]['rollout_turns'][0])], .2)
        self.assertAlmostEqual(actual[id(rows[0]['rollout_turns'][1])], -.05)
        self.assertEqual(sum(len(t['response_ids']) for p in partitions for t, _ in p), 4)

    def test_decision_selection_does_not_depend_on_terminal_outcome(self):
        a = self.source()
        b = {**a, 'failure_kind': 'model', 'training_reward': -.65}
        self.assertEqual(decision_for(a, random.Random(42)), decision_for(b, random.Random(42)))

    def test_real_serving_reply_records_predecision_history_without_aliasing(self):
        class Tokens:
            def apply_chat_template(self, *args, **kwargs):
                return [1, 2]
        client = TrainingClient('unused', 'policy', Tokens(), 42, load_settings('local'))
        response = {'prompt_token_ids': [1, 2], 'choices': [{'token_ids': [3],
            'logprobs': {'content': [{'token': 'token_id:3', 'logprob': -.1}]},
            'finish_reason': 'tool_calls', 'message': message('buy_now')}]}
        history = [{'role': 'user', 'content': 'buy'}]
        with patch('rollout.request_json', return_value=response):
            client.complete(history, [], 512)
        history.append(message('buy_now'))
        self.assertEqual(len(client.turns[0]['replay']['messages']), 1)
        self.assertEqual(client.turns[0]['replay']['max_tokens'], 512)
        self.assertEqual(client.cost, {'requests': 1, 'prompt_tokens': 2, 'completion_tokens': 1})

    def test_overlay_changes_no_baseline_hyperparameters_or_execution(self):
        base, local = load_settings(), load_settings('local')
        self.assertEqual({k: v for k, v in local.items() if k not in {'local_credit', 'version'}},
                         {k: v for k, v in base.items() if k != 'version'})
        self.assertEqual(validate_execution(), 'e66a82e73e641dc14460fec97c46038f75ee77866e8cbec6ed382fc9e5422227')


if __name__ == '__main__':
    unittest.main()
