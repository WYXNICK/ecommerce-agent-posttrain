"""Fixed-action parity through real veRL tool processing and the evaluation collector."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from omegaconf import OmegaConf
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from shopping_grpo.evaluation.rollout import collect_for_task
from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
from shopping_grpo.training.grpo.adapter.runtime import reward_breakdown
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
from test_rollout import GuardRecoveryEnv, MockClient, assistant_tool
from test_verl_adapter import make_tool


class ProtocolEnv(GuardRecoveryEnv):
    def step(self, action):
        result = super().step(action)
        if result.get('done'):
            result['reward_detail'] = {
                'reward_version': 'shopsimulator-reward-v3', 'reward_type': 'gold_purchase',
                'termination_reason': 'gold_purchase', 'reward_valid': True,
                'sampling_invalid': False, 'terminal_utility': 1.0,
                'purchase_success': True, 'hard_gates': {},
                'weighted_score': 1.0, 'evidence_coverage': 1.0,
            }
        return result


async def training_sequence(sequence, max_steps=35):
    session = ShopSimulatorSession(max_steps=max_steps, env_factory=ProtocolEnv)
    state = await session.start(1)
    loop = object.__new__(ShoppingToolAgentLoop)
    loop.tools = {s['function']['name']: make_tool(s['function']['name']) for s in SHOP_TOOL_SCHEMAS}
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 16384
    loop.tool_response_truncate_side = 'middle'
    loop.response_length = 20480
    loop.processor = None
    loop.tool_parser_name = 'qwen3_coder'
    loop.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text.split()))))
    loop.observation_token_budget = 1536
    loop.observation_detail_token_budget = 4096
    loop.observation_generic_token_budget = 768
    loop.observation_search_top_k = 20
    captured = []

    async def apply_template(messages, **kwargs):
        captured.extend(m['content'] for m in messages if m['role'] == 'tool')
        return [1]

    loop.apply_chat_template = apply_template
    data = SimpleNamespace(prompt_ids=[], response_ids=[], response_mask=[], response_logprobs=[],
                           tools_kwargs={}, messages=[], image_data=None, video_data=None,
                           user_turns=0, metrics={}, tool_rewards=[])
    try:
        for assistant in sequence:
            calls = assistant.get('tool_calls') or []
            if not calls:
                state.update(terminate=True, termination_reason='assistant_final', error='assistant_final')
                break
            data.tool_calls = [SimpleNamespace(name=c['function']['name'],
                                              arguments=c['function']['arguments']) for c in calls]
            result = await loop._handle_processing_tools_state(data)
            if result == AgentState.TERMINATED:
                break
        return dict(state), list(session.env.actions), captured
    finally:
        await session.close()


class GrpoProtocolTest(unittest.TestCase):
    def test_native_turn_counter_does_not_preempt_last_legal_tool(self):
        config = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/grpo.yaml')
        multi = config.actor_rollout_ref.rollout.multi_turn
        loop = object.__new__(ShoppingToolAgentLoop)
        loop.max_assistant_turns = multi.max_assistant_turns
        loop.max_user_turns = multi.max_user_turns
        loop.context_compaction_enable = False
        loop.context_window_tokens = 24576
        loop.context_generation_reserve_tokens = 512
        loop.context_safety_margin_tokens = 512
        loop.response_length = 20480
        loop.tools = {'buy_now': make_tool('buy_now')}
        call = SimpleNamespace(name='buy_now', arguments='{}')
        loop.tool_parser = SimpleNamespace(stop_token_ids=None,
                                          extract_tool_calls=AsyncMock(return_value=('', [call])))
        output = SimpleNamespace(token_ids=[1], log_probs=[], num_preempted=None,
                                 extra_fields={}, routed_experts=None)
        loop.server_manager = SimpleNamespace(generate=AsyncMock(return_value=output))
        data = AgentData([], None, None, None, None, {}, 'fixed-action', {})
        data.assistant_turns = data.user_turns = 104
        result = asyncio.run(loop._handle_generating_state(data, {}))
        self.assertEqual(result, AgentState.PROCESSING_TOOLS)
        self.assertEqual(data.tool_calls, [call])
        # Removing redundant turn limits must preserve the response-token limit.
        loop.response_length = 2
        self.assertEqual(asyncio.run(loop._handle_generating_state(data, {})), AgentState.TERMINATED)

    def compare(self, sequence, max_steps=35):
        trajectory = collect_for_task({'task_id': 1}, client=MockClient(sequence),
                                      env_factory=ProtocolEnv, base_url='http://test', max_steps=max_steps)
        state, actions, feedback = asyncio.run(training_sequence(sequence, max_steps))
        self.assertEqual(actions, [s['env_action'] for s in trajectory['steps'] if s['env_action'] is not None])
        self.assertEqual(state['guard_rejection_count'], len(trajectory['blocked_tool_calls']))
        expected = [m['content'] for m in trajectory['messages'] if m.get('runtime_action_guard')]
        actual = [s for s in feedback if '本地动作守卫拒绝' in s]
        self.assertEqual(actual, expected)
        self.assertEqual(len(state['steps']), len(trajectory['steps']))
        self.assertEqual(state['done'], trajectory['done'])
        return state, trajectory

    def test_illegal_subpage_purchase_recovers_with_identical_feedback(self):
        seq = [assistant_tool('search_products', {'query': '乳胶枕'}),
               assistant_tool('open_product', {'asin': '100000000001'}),
               assistant_tool('view_features', {}), assistant_tool('buy_now', {}),
               assistant_tool('prev_page', {}), assistant_tool('buy_now', {})]
        state, trajectory = self.compare(seq)
        self.assertEqual(trajectory['status'], 'done')
        self.assertEqual(reward_breakdown(state)['training_reward'], 1.0)

    def test_serial_multiple_calls_execute_only_first(self):
        first = assistant_tool('search_products', {'query': '乳胶枕'})
        first['tool_calls'] += assistant_tool('buy_now', {})['tool_calls']
        state, trajectory = self.compare([first, assistant_tool('open_product', {'asin': '100000000001'}),
                                          assistant_tool('buy_now', {})])
        self.assertEqual(state['dropped_tool_calls'], 1)
        self.assertEqual(len(trajectory['tool_call_truncations']), 1)

    def test_three_rejections_are_a_trainable_model_failure(self):
        state, trajectory = self.compare([assistant_tool('buy_now', {})] * 3)
        self.assertEqual(state['termination_reason'], trajectory['status'])
        reward = reward_breakdown(state)
        self.assertEqual(reward['training_reward'], -0.65)
        self.assertEqual(reward['terminal_utility'], 0.0)
        self.assertFalse(reward['sampling_invalid'])
        self.assertEqual(reward['failure_kind'], 'model')

    def test_malformed_json_and_unknown_tool_use_same_rejection_path(self):
        bad = assistant_tool('search_products', {})
        bad['tool_calls'][0]['function']['arguments'] = '{'
        state, _ = self.compare([bad, assistant_tool('invented_tool', {}), assistant_tool('buy_now', {})])
        self.assertEqual(state['termination_reason'], 'invalid_action_limit')
        self.assertFalse(reward_breakdown(state)['sampling_invalid'])

    def test_missing_query_is_model_failure_not_environment_exception(self):
        state, _ = self.compare([assistant_tool('search_products', {})] * 3)
        self.assertFalse(state['infrastructure_invalid'])
        self.assertFalse(reward_breakdown(state)['sampling_invalid'])

    def test_early_final_and_think_budget_have_same_termination(self):
        final = {'role': 'assistant', 'content': 'done'}
        state, trajectory = self.compare([final])
        self.assertEqual(state['termination_reason'], trajectory['status'])
        self.assertLess(reward_breakdown(state)['training_reward'], 0)
        state, trajectory = self.compare([assistant_tool('think', {'note': 'plan'})], max_steps=1)
        self.assertEqual(state['termination_reason'], trajectory['status'])
        self.assertEqual(state['latest_observation'], 'plan')
