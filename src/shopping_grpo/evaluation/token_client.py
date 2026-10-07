"""Local vLLM evaluation using the same raw token trajectory as veRL training."""

import asyncio
from uuid import uuid4

from shopping_grpo.environment.context import ResponseBudgetError, generation_token_limit
from shopping_grpo.environment.tokenization import token_fingerprint, tool_response_tokens
from shopping_grpo.evaluation.rollout import OpenAIChatClient


class VllmTokenClient(OpenAIChatClient):
    """Keep generated tokens verbatim; never reserialize a sampled tool call."""

    def __init__(self, *, tokenizer, response_length=20480, prompt_length=4096, **kwargs):
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.tools.schemas import OpenAIFunctionToolSchema
        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.response_length = int(response_length)
        self.prompt_length = int(prompt_length)
        self.template_kwargs = {"enable_thinking": False}
        self.tool_parser = ToolParser.get_tool_parser("qwen3_coder", tokenizer)
        self.tool_schemas = [OpenAIFunctionToolSchema.model_validate(s) for s in SHOP_TOOL_SCHEMAS]
        self.generation_turns = []
        self.prefix_ids = []
        self.response_tokens = 0
        if self.response_length < 1 or self.prompt_length < 1:
            raise ValueError("response_length and prompt_length must be positive")
        if self.context_window is None or self.context_compaction_enable:
            raise ValueError("token trajectory evaluation requires a fixed context without compaction")
        # Observation projection uses this same tokenizer without an HTTP round trip.
        self.observation_token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))

    def complete(self, messages, tools):
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import normalize_token_ids

        self.last_context_event = None
        if len(messages) == 2:
            self.generation_turns = []
            self.prefix_ids = normalize_token_ids(apply_chat_template(
                self.tokenizer, messages,
                tools=[s.model_dump(exclude_unset=True, exclude_none=True) for s in self.tool_schemas],
                tokenize=True,
                add_generation_prompt=True, **self.template_kwargs,
            ))
            if len(self.prefix_ids) > self.prompt_length:
                raise ValueError("initial evaluation prompt exceeds the training prompt limit")
            self.response_tokens = 0
        else:
            tokens = tool_response_tokens(
                self.tokenizer, [messages[-1]], template_kwargs=self.template_kwargs,
            )
            self.generation_turns[-1]["tool_tokens_attempted"] = len(tokens)
            if self.response_tokens + len(tokens) >= self.response_length:
                raise ResponseBudgetError("response_budget_exhausted")
            self.prefix_ids.extend(tokens)
            self.response_tokens += len(tokens)
            self.generation_turns[-1].update(
                tool_tokens=len(tokens), response_tokens_after=self.response_tokens,
            )
        self.last_context_tokens = len(self.prefix_ids)
        limit = generation_token_limit(
            input_tokens=len(self.prefix_ids), response_tokens=self.response_tokens,
            context_window=self.context_window, response_length=self.response_length,
            max_tokens=self.max_tokens, safety_margin=self.context_safety_margin,
        )
        turn = {
            "input_tokens": len(self.prefix_ids),
            "input_token_sha256": token_fingerprint(self.prefix_ids),
            "response_tokens_before": self.response_tokens,
            "effective_max_tokens": limit,
            "remaining_response_tokens": self.response_length - self.response_tokens,
            "remaining_context_tokens": self.context_window - len(self.prefix_ids)
                                        - self.context_safety_margin,
        }
        payload = {
            "model": self.model, "prompt": self.prefix_ids,
            "max_tokens": limit, "temperature": self.temperature, "top_p": self.top_p,
            "return_token_ids": True, "add_special_tokens": False,
            "skip_special_tokens": False,
        }
        response = self._request(f"{self.base_url}/completions", payload)
        choice = response["choices"][0]
        tokens = choice.get("token_ids")
        if not isinstance(tokens, list) or not tokens or len(tokens) > limit:
            raise ValueError("vLLM completion must return token_ids within the requested limit")
        self.prefix_ids.extend(tokens)
        self.response_tokens += len(tokens)
        turn.update(assistant_tokens=len(tokens), tool_tokens=0,
                    response_tokens_after=self.response_tokens)
        self.generation_turns.append(turn)
        if self.response_tokens >= self.response_length:
            raise ResponseBudgetError("response_budget_exhausted")
        content, calls = asyncio.run(self.tool_parser.extract_tool_calls(tokens, self.tool_schemas))
        return {
            "role": "assistant", "content": content or None,
            "tool_calls": [{
                "id": f"call-{uuid4().hex}", "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            } for call in calls],
        }
