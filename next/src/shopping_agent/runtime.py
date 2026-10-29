"""The shared actor loop: visible state -> guarded action -> real environment."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from uuid import uuid4

from shopping_agent.clients import ModelOutputError, ModelServiceError
from shopping_agent.environment.actions import (
    MAX_BLOCKED_TOOL_CALLS, action_guard_tool_message, action_reject_reason, serial_tool_calls,
)
from shopping_agent.environment.client import ShopAgentEnv
from shopping_agent.environment.observation import render_structured_observation
from shopping_agent.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action
from shopping_agent.evaluation.summary import _is_strict_success
from shopping_agent.prompts import SYSTEM_PROMPT
from shopping_agent.protocol import BudgetExceeded, EpisodeBudget, visible_observation


def observation_text(result):
    if result.get("observation_state") is not None:
        return render_structured_observation(result["observation_state"])
    return result.get("instruction", result.get("observation", ""))


def prepare_action(assistant, observation):
    """Used by every entry; extra calls never execute against an outdated page."""
    assistant = deepcopy(assistant)
    calls, dropped = serial_tool_calls(assistant.get("tool_calls") or [])
    if not calls:
        return assistant, None, None, dropped
    assistant["tool_calls"] = calls
    function = calls[0].get("function") or {}
    name = function.get("name")
    try:
        raw = function.get("arguments") or "{}"
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return assistant, (name, {}), "invalid_tool_arguments", dropped
    return assistant, (name, arguments), action_reject_reason(name, arguments, observation), dropped


def execute_action(env, tool_call, name, arguments, index):
    action = tool_call_to_action(name, arguments)
    result = env.step(action) if action is not None else {
        "instruction": arguments.get("note", ""), "reward": 0.0, "done": False,
    }
    return {
        "step_index": index, "tool_call": tool_call, "tool_name": name,
        "parameters": arguments, "env_action": action,
        "observation": observation_text(result), "reward": float(result.get("reward", 0)),
        "done": bool(result.get("done")), "result": result,
    }


def run_episode(task, *, client, config, tokenizer, base_url, fingerprint, env_factory=ShopAgentEnv,
                attempt_index=0):
    trajectory = {
        "trajectory_id": str(uuid4()), "task_id": int(task["task_id"]),
        "attempt_index": attempt_index, "execution_fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "messages": [], "steps": [], "blocked_tool_calls": [], "tool_call_truncations": [],
        "model_calls": [], "initial_result": {}, "terminal_result": {},
        "final_reward": 0.0, "done": False, "error": None, "release_error": None,
        "failure_kind": None,
    }
    env = env_factory(base_url=base_url)
    try:
        initial = env.reset(task["task_id"])
        trajectory["initial_result"] = initial
        if initial.get("environment_version") != config["environment"]:
            raise ValueError("server environment version differs from frozen execution contract")
        latest = observation_text(initial)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial["instruction"]},
        ]
        trajectory["messages"] = messages
        budget = EpisodeBudget(config, tokenizer, tokenizer.count(messages))
        blocked = 0
        latest_truncated = False
        while len(trajectory["steps"]) < config["budgets"]["max_steps"]:
            limit, state = budget.next_limit(messages)
            reply = client.complete(messages, SHOP_TOOL_SCHEMAS, limit)
            trajectory["model_calls"].append({**reply.audit, "visible_budget": state})
            assistant, parsed, reason, dropped = prepare_action(reply.message, latest)
            if reason == "invalid_tool_arguments":
                trajectory["status"] = "model_output_invalid"
                break
            if dropped:
                trajectory["tool_call_truncations"].append({"dropped_tool_calls": dropped})
            before = list(messages)
            messages.append(assistant)
            budget.validate_assistant(before, messages, limit)
            if parsed is None:
                trajectory["status"] = "assistant_final"
                break
            call = assistant["tool_calls"][0]
            if reason:
                blocked += 1
                trajectory["blocked_tool_calls"].append({
                    "step_index": len(trajectory["steps"]), "tool_call": call, "reason": reason,
                    "consecutive_count": blocked, "latest_observation_truncated": latest_truncated,
                })
                messages.append(action_guard_tool_message(call, reason, latest))
                if blocked >= MAX_BLOCKED_TOOL_CALLS:
                    trajectory["status"] = "invalid_action_limit"
                    break
                continue
            name, arguments = parsed
            step = execute_action(env, call, name, arguments, len(trajectory["steps"]))
            # Record the real transition before projecting; a projector failure must
            # not erase an action that already reached the environment.
            trajectory["steps"].append(step)
            raw = step["observation"]
            if step["done"]:
                trajectory.update(status="done", done=True, terminal_result=step["result"],
                                  final_reward=step["reward"])
                step["raw_observation"] = raw
                step["observation"] = "购买已完成。" if _is_strict_success(trajectory) else "任务已结束。"
            else:
                text, projection = visible_observation(config, tokenizer, name, raw, arguments)
                step.update(raw_observation=raw, observation=text, projection=projection)
            latest = step["observation"]
            latest_truncated = bool((step.get("projection") or {}).get("truncated"))
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": latest})
            blocked = 0
            if step["done"]:
                # A completed real terminal wins over the budget of the next turn.
                break
        else:
            trajectory["status"] = "max_steps"
        trajectory["token_budget"] = budget.state(messages)
        if trajectory["steps"] and not trajectory["done"]:
            trajectory["final_reward"] = trajectory["steps"][-1]["reward"]
        detail = trajectory["terminal_result"].get("reward_detail") or {}
        trajectory["failure_kind"] = (
            None if _is_strict_success(trajectory) else
            "unverifiable" if trajectory["done"] and detail.get("reward_valid") is not True else "model"
        )
    except BudgetExceeded as exc:
        trajectory.update(status=exc.status, failure_kind="model")
    except ModelOutputError as exc:
        trajectory["model_calls"].append(exc.audit)
        trajectory.update(status="model_output_invalid", failure_kind="model",
                          error={"type": type(exc).__name__, "message": str(exc)})
    except Exception as exc:
        if isinstance(exc, ModelServiceError):
            trajectory["model_calls"].append(exc.audit)
        trajectory.update(status="infrastructure_error", failure_kind="infrastructure",
                          error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        if trajectory["steps"]:
            trajectory["final_reward"] = trajectory["steps"][-1]["reward"]
        try:
            env.release()
        except Exception as exc:
            trajectory.update(status="environment_release_failed", failure_kind="infrastructure",
                              release_error={"type": type(exc).__name__, "message": str(exc)})
    return trajectory
