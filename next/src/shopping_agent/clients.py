"""Wire adapters return the same tool-call message; execution stays provider independent."""

from dataclasses import dataclass
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

MAX_TRANSPORT_ATTEMPTS = 3
RETRY_DELAYS = (5, 10)
logger = logging.getLogger(__name__)


class ModelServiceError(RuntimeError):
    def __init__(self, message, audit=None):
        super().__init__(message)
        self.audit = audit or {}


class ModelOutputError(ModelServiceError):
    """A completed model response was refused, truncated or not a valid action."""


@dataclass(frozen=True)
class ModelReply:
    message: dict
    audit: dict


def read_modelhub_key():
    key = os.environ.get("MODELHUB_AK")
    if not key and os.environ.get("MODELHUB_AK_FILE"):
        key = json.loads(Path(os.environ["MODELHUB_AK_FILE"]).read_text())["ak"]
    if not key:
        raise ValueError("set MODELHUB_AK or MODELHUB_AK_FILE; never put keys in the run config")
    return key


def request_json(endpoint, payload, *, headers, timeout, secret="", transport=None,
                 rate_limit_attempts=MAX_TRANSPORT_ATTEMPTS,
                 rate_limit_initial_delay_seconds=10, rate_limit_max_delay_seconds=120):
    """Bounded transport retries; credential-bearing URLs never enter exceptions."""
    errors = []
    for attempt in range(max(MAX_TRANSPORT_ATTEMPTS, rate_limit_attempts)):
        try:
            if transport:
                result = transport(endpoint, payload, headers, timeout)
            else:
                request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
                with urlopen(request, timeout=timeout) as raw:
                    result = json.loads(raw.read().decode())
            if isinstance(result, dict):
                result["_transport_attempts"] = attempt + 1
                result["_transport_errors"] = errors
            return result
        except HTTPError as exc:
            body = exc.read().decode(errors="replace")
            if secret:
                body = body.replace(secret, "[REDACTED]")
            gateway_timeout = (exc.code == 403 and "Compliance Gateway HTTP" in body
                               and any(stage in body for stage in ("Forward to backend error", "check response failed"))
                               and "i/o timeout" in body)
            retryable = exc.code in {429, 500, 502, 503, 504} or gateway_timeout
            kind = "gateway_timeout" if gateway_timeout else f"http_{exc.code}"
            message = f"model service HTTP {exc.code}: {body[:1500]}"
        except (URLError, OSError) as exc:
            retryable, kind = True, "connection_error"
            message = f"model connection failed: {type(exc).__name__}"
        rate_limited = kind == "http_429" and rate_limit_attempts > MAX_TRANSPORT_ATTEMPTS
        maximum = rate_limit_attempts if rate_limited else MAX_TRANSPORT_ATTEMPTS
        retry = retryable and attempt + 1 < maximum
        delay = (min(rate_limit_initial_delay_seconds * 2 ** attempt, rate_limit_max_delay_seconds)
                 if rate_limited else RETRY_DELAYS[attempt]) if retry else 0
        errors.append({"attempt": attempt + 1, "kind": kind, "retry_delay_seconds": delay})
        if not retry:
            raise ModelServiceError(message, {"transport_attempts": attempt + 1, "transport_errors": errors}) from None
        logger.warning("model transport retry request_id=%s kind=%s attempt=%s/%s delay=%ss",
                       headers.get("X-TT-LOGID", "local"), kind, attempt + 1, maximum, delay)
        time.sleep(delay)


def decode_completion(response, request_id, elapsed):
    if not isinstance(response, dict) or not response.get("choices"):
        raise ModelServiceError("model service returned no completion choices")
    choice = response["choices"][0]
    audit = {
        "request_id": request_id, "response_id": response.get("id"),
        "response_model": response.get("model"), "usage": response.get("usage"),
        "finish_reason": choice.get("finish_reason"), "latency_seconds": elapsed,
        "transport_attempts": response.get("_transport_attempts", 1),
        "transport_errors": response.get("_transport_errors", []),
    }
    if choice.get("finish_reason") in {"length", "content_filter"}:
        raise ModelOutputError(f"model output ended with {choice['finish_reason']}", audit)
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ModelServiceError("model service returned no assistant message", audit)
    if message.get("refusal"):
        raise ModelOutputError("model refused the action request", audit)
    return message, audit


def action_schema(tools):
    variants = []
    for tool in tools:
        function = tool["function"]
        variants.append({
            "type": "object",
            "properties": {"name": {"type": "string", "enum": [function["name"]]},
                           "arguments": deepcopy(function["parameters"])},
            "required": ["name", "arguments"], "additionalProperties": False,
        })
    return {
        "type": "object", "properties": {"action": {"anyOf": variants}},
        "required": ["action"], "additionalProperties": False,
    }


def wire_messages(messages):
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    return [{key: deepcopy(value) for key, value in message.items() if key in allowed}
            for message in messages]


class ModelHubClient:
    """Use high reasoning + Structured Outputs through the Crawl gateway.

    The JSON is an action proposal, converted to the same canonical function call
    that the shared executor validates and executes against the real environment.
    No action outcomes or reasoning text are fabricated or used as SFT targets.
    """

    def __init__(self, settings, *, key=None, transport=None):
        self.settings = settings
        self.key = key or read_modelhub_key()
        self.transport = transport
        if urlsplit(settings["endpoint"]).query:
            raise ValueError("ModelHub endpoint must not contain credentials or query parameters")

    def complete(self, messages, tools, max_tokens):
        request_id = uuid4().hex
        instructions = (
            "本接口使用严格 JSON 动作输出。根据下列工具说明选择一个动作，"
            "只输出 action 对象，不输出推理文本。动作结果由外部真实环境返回。工具说明：\n"
            + json.dumps(tools, ensure_ascii=False)
        )
        normalized = wire_messages(messages)
        request_messages = [*normalized[:1], {"role": "system", "content": instructions},
                            *normalized[1:]]
        payload = {
            "model": self.settings["model"], "messages": request_messages,
            "reasoning_effort": self.settings["reasoning_effort"],
            "max_completion_tokens": self.settings["max_completion_tokens"],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "shopping_next_action", "strict": True, "schema": action_schema(tools),
            }},
        }
        started = time.monotonic()
        response = request_json(
            self.settings["endpoint"] + "?" + urlencode({"ak": self.key}), payload,
            headers={"Content-Type": "application/json", "X-TT-LOGID": request_id},
            timeout=self.settings["timeout_seconds"], secret=self.key, transport=self.transport,
            rate_limit_attempts=self.settings.get("rate_limit_attempts", MAX_TRANSPORT_ATTEMPTS),
            rate_limit_initial_delay_seconds=self.settings.get("rate_limit_initial_delay_seconds", 10),
            rate_limit_max_delay_seconds=self.settings.get("rate_limit_max_delay_seconds", 120),
        )
        raw, audit = decode_completion(response, request_id, time.monotonic() - started)
        audit["wire_format"] = "structured_action"
        audit["requested_reasoning_effort"] = self.settings["reasoning_effort"]
        if audit["response_model"] != self.settings["expected_response_model"]:
            raise ModelServiceError("teacher snapshot changed; review the teacher configuration", audit)
        try:
            action = json.loads(raw["content"])["action"]
            name, arguments = action["name"], action["arguments"]
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("action must have a string name and object arguments")
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelOutputError(f"invalid structured action: {type(exc).__name__}", audit) from None
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_" + request_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }]}
        return ModelReply(message, audit)


class LocalChatClient:
    def __init__(self, base_url, model="shopping-agent", *, api_key="EMPTY", timeout=180, transport=None):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key, self.timeout, self.transport = api_key, timeout, transport

    def complete(self, messages, tools, max_tokens):
        request_id = uuid4().hex
        payload = {
            "model": self.model, "messages": wire_messages(messages), "tools": tools, "tool_choice": "auto",
            "parallel_tool_calls": False, "temperature": 0, "top_p": 1,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.monotonic()
        response = request_json(
            self.base_url + "/chat/completions", payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key},
            timeout=self.timeout, secret=self.api_key, transport=self.transport,
        )
        raw, audit = decode_completion(response, request_id, time.monotonic() - started)
        message = {key: raw[key] for key in ("role", "content", "tool_calls") if key in raw}
        message["role"] = "assistant"
        audit["wire_format"] = "chat_function_call"
        return ModelReply(message, audit)
