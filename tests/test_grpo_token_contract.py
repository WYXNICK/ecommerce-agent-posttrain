"""Real Qwen tokenizer regression for request limits, prefixes and learning masks."""

import asyncio
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from shopping_grpo.environment.context import (
    ContextBudgetError, ResponseBudgetError, generation_token_limit,
)
from shopping_grpo.environment.tokenization import tool_response_tokens


class GenerationBudgetTest(unittest.TestCase):
    def limit(self, **overrides):
        options = dict(input_tokens=1800, response_tokens=0, context_window=24576,
                       response_length=20480, max_tokens=512, safety_margin=512)
        return generation_token_limit(**(options | overrides))

    def test_default_native_request_and_explicit_limits(self):
        self.assertEqual(self.limit(), 512)
        self.assertEqual(self.limit(requested_max_tokens=256), 256)
        self.assertEqual(self.limit(requested_max_tokens=4096), 512)
        self.assertEqual(self.limit(response_tokens=20479), 1)

    def test_context_and_response_boundaries(self):
        self.assertEqual(self.limit(input_tokens=23552), 512)
        with self.assertRaises(ContextBudgetError):
            self.limit(input_tokens=23553)
        with self.assertRaises(ResponseBudgetError):
            self.limit(response_tokens=20480)


TOKENIZER = Path(os.environ.get(
    "SHOPPING_TOKENIZER_PATH",
    Path(__file__).resolve().parents[1] / "outputs/models/sft-merged",
))


@unittest.skipUnless((TOKENIZER / "tokenizer.json").is_file(), "local Qwen tokenizer required")
class RealTokenProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor, AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        cls.processor = AutoProcessor.from_pretrained(TOKENIZER, local_files_only=True)

    def client(self, texts, **overrides):
        from shopping_grpo.evaluation.token_client import VllmTokenClient
        tokens = [self.tokenizer.encode(t, add_special_tokens=False) for t in texts]
        requests = []

        def transport(url, payload, headers, timeout):
            requests.append(copy.deepcopy(payload))
            return {"choices": [{"token_ids": tokens.pop(0)}]}

        options = dict(model="shopping-agent", base_url="http://test/v1", api_key="EMPTY",
                       tokenizer=self.tokenizer, context_window=24576, max_tokens=512,
                       observation_token_budget=1536, transport=transport)
        return VllmTokenClient(**(options | overrides)), requests

    def test_real_prefix_and_mask_match_native_multi_turn_execution(self):
        from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.tools.schemas import OpenAIFunctionToolSchema
        from verl.utils.chat_template import initialize_system_prompt
        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
        from shopping_grpo.evaluation.rollout import collect_for_task, _initial_messages
        from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
        from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
        from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
        from test_grpo_protocol import ProtocolEnv

        # Noncanonical whitespace must survive exactly; parsing/re-rendering would lose it.
        texts = [
            '\n<tool_call>\n<function=search_products>\n<parameter=query>乳胶枕</parameter>\n</function>\n</tool_call><|im_end|>',
            '<tool_call>\n<function=open_product>\n<parameter=asin>100000000001</parameter>\n</function>\n</tool_call><|im_end|>',
            '<tool_call>\n<function=view_features>\n</function>\n</tool_call><|im_end|>',
            '<tool_call>\n<function=buy_now>\n</function>\n</tool_call><|im_end|>',
            '<tool_call>\n<function=prev_page>\n</function>\n</tool_call><|im_end|>',
            '<tool_call>\n<function=buy_now>\n</function>\n</tool_call><|im_end|>',
        ]

        async def native():
            session = ShopSimulatorSession(max_steps=35, env_factory=ProtocolEnv)
            state = await session.start(1)
            loop = object.__new__(ShoppingToolAgentLoop)
            loop.tokenizer = self.tokenizer
            loop.processor = self.processor
            loop.loop = asyncio.get_running_loop()
            loop.apply_chat_template_kwargs = {"enable_thinking": False}
            loop.system_prompt = initialize_system_prompt(self.tokenizer, enable_thinking=False)
            loop.rollout_config = SimpleNamespace(prompt_length=4096)
            loop.tools = {s['function']['name']: ShopSimulatorTool({}, OpenAIFunctionToolSchema.model_validate(s))
                          for s in SHOP_TOOL_SCHEMAS}
            loop.tool_schemas = [t.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                                 for t in loop.tools.values()]
            loop.tool_parser = ToolParser.get_tool_parser('qwen3_coder', self.tokenizer)
            loop.tool_parser_name = 'qwen3_coder'
            loop.max_assistant_turns = loop.max_user_turns = None
            loop.max_parallel_calls = 1
            loop.max_tool_response_length = 16384
            loop.tool_response_truncate_side = 'middle'
            loop.response_length = 20480
            loop.context_compaction_enable = False
            loop.context_window_tokens = 24576
            loop.context_generation_reserve_tokens = loop.context_safety_margin_tokens = 512
            loop.observation_token_budget = 1536
            loop.observation_detail_token_budget = 4096
            loop.observation_generic_token_budget = 768
            loop.observation_search_top_k = 20
            outputs = [SimpleNamespace(token_ids=self.tokenizer.encode(t, add_special_tokens=False),
                       log_probs=[], num_preempted=None, extra_fields={}, routed_experts=None)
                       for t in texts]
            requests = []

            async def generate(**kwargs):
                requests.append(copy.deepcopy(kwargs))
                return outputs.pop(0)

            loop.server_manager = SimpleNamespace(generate=AsyncMock(side_effect=generate))
            messages = _initial_messages({'task_id': 1}, {'instruction': state['latest_observation']})
            data = AgentData(messages, None, None, None, None, {}, 'token-regression', {})
            try:
                await loop._handle_pending_state(data, {})
                while not state['terminate']:
                    result = await loop._handle_generating_state(data, {})
                    if result == AgentState.TERMINATED:
                        break
                    result = await loop._handle_processing_tools_state(data)
                    if result == AgentState.TERMINATED:
                        break
                return requests, data, state
            finally:
                await session.close()

        native_requests, data, state = asyncio.run(native())
        client, eval_requests = self.client(texts)
        trajectory = collect_for_task({'task_id': 1}, client=client, env_factory=ProtocolEnv,
                                      base_url='http://test', max_steps=35)
        self.assertTrue(trajectory['done'])
        self.assertEqual(state['guard_rejection_count'], 1)
        self.assertEqual(len(trajectory['blocked_tool_calls']), 1)
        self.assertEqual(len(native_requests), len(eval_requests))
        for native_request, eval_request in zip(native_requests, eval_requests, strict=True):
            self.assertEqual(native_request['prompt_ids'], eval_request['prompt'])
            self.assertEqual(native_request['sampling_params']['max_tokens'], eval_request['max_tokens'])
        assistant_count = sum(len(self.tokenizer.encode(t, add_special_tokens=False)) for t in texts)
        self.assertEqual(sum(data.response_mask), assistant_count)
        self.assertEqual(len(data.response_mask), len(data.prompt_ids) - len(native_requests[0]['prompt_ids']))
        self.assertIn(0, data.response_mask)

    def test_tool_wrapper_survives_chunks_larger_than_initial_prompt_limit(self):
        from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
        content = '规格 ' * 4500
        messages = [{'role': 'tool', 'content': content}]
        expected = tool_response_tokens(self.tokenizer, messages, template_kwargs={'enable_thinking': False})
        self.assertGreater(len(expected), 4096)
        loop = object.__new__(ShoppingToolAgentLoop)
        loop.tokenizer = self.tokenizer
        loop.apply_chat_template_kwargs = {'enable_thinking': False}
        actual = asyncio.run(loop.apply_chat_template(messages, remove_system_prompt=True))
        self.assertEqual(actual, expected)
        self.assertIn('<tool_response>', self.tokenizer.decode(actual))

    def test_generation_reaching_response_limit_never_executes_the_tool(self):
        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
        text = '<tool_call>\n<function=buy_now>\n</function>\n</tool_call><|im_end|>'
        count = len(self.tokenizer.encode(text, add_special_tokens=False))
        client, requests = self.client([text], response_length=count)
        with self.assertRaises(ResponseBudgetError):
            client.complete([{'role': 'system', 'content': 'rules'},
                             {'role': 'user', 'content': 'task'}], SHOP_TOOL_SCHEMAS)
        self.assertEqual(requests[0]['max_tokens'], count)

    def test_tool_feedback_reaching_response_limit_prevents_next_generation(self):
        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
        text = '<tool_call>\n<function=search_products>\n<parameter=query>乳胶枕</parameter>\n</function>\n</tool_call><|im_end|>'
        client, requests = self.client([text])
        messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'task'}]
        assistant = client.complete(messages, SHOP_TOOL_SCHEMAS)
        messages += [assistant, {'role': 'tool', 'content': 'feedback'}]
        feedback = tool_response_tokens(self.tokenizer, [messages[-1]], template_kwargs={'enable_thinking': False})
        client.response_length = client.response_tokens + len(feedback)
        with self.assertRaises(ResponseBudgetError):
            client.complete(messages, SHOP_TOOL_SCHEMAS)
        self.assertEqual(len(requests), 1)
