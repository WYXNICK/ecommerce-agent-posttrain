"""Behavioral checks for shared execution and collection acceptance boundaries."""

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopping_agent.clients import ModelHubClient, ModelOutputError, ModelReply, ModelServiceError, request_json
from shopping_agent.cli import selected_tasks
from shopping_agent.collection.sft import acceptance_reasons, build_sft_row
from shopping_agent.environment.observation import render_structured_observation
from shopping_agent.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_agent.protocol import EpisodeBudget, load_config, sha256
from shopping_agent.runs import CollectionProgress, RunStore, collect, evaluate
from shopping_agent.runtime import run_episode
from shopping_agent.tasks import TaskPool, domain_quotas, read_jsonl


def message(name, arguments=None, call_id="call1"):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments or {})}}]}


class Tokenizer:
    def count(self, messages, generation_prompt=False):
        return len(json.dumps(messages, ensure_ascii=False)) // 10

    def text_tokens(self, text):
        return len(text) // 10


class Environment:
    def __init__(self, **_):
        self.actions = []
        self.released = False

    def reset(self, task_id):
        return {"instruction": '购买一个商品。\n可点击的按钮: ["Buy Now"]', "goal_options": {"secret": "DO_NOT_SHOW"},
                "environment_version": "shopsimulator-environment-v2.1"}

    def step(self, action):
        self.actions.append(action)
        if action.startswith("search["):
            return {"done": False, "reward": 0, "instruction": '商品详情\n搜索功能是否可用: False\n可点击的按钮: ["Buy Now"]'}
        return {"done": True, "over": True, "reward": 1.0, "instruction": "private reward text",
                "reward_detail": {"reward_version": "shopsimulator-reward-v3", "reward_type": "gold_purchase",
                                  "reward_valid": True, "purchase_success": True, "termination_reason": "gold_purchase"}}

    def release(self):
        self.released = True


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []

    def complete(self, messages, tools, max_tokens):
        self.seen.append(deepcopy(messages))
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return ModelReply(value, {"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})


class ExecutionTests(unittest.TestCase):
    def episode(self, replies, *, config=None):
        env = Environment()
        client = Client(replies)
        result = run_episode({"task_id": 1}, client=client, config=config or load_config(), tokenizer=Tokenizer(),
                             base_url="unused", fingerprint="frozen", env_factory=lambda **_: env)
        return result, env, client

    def test_guard_recovery_and_sft_remove_failed_attempt(self):
        row, env, client = self.episode([message("open_product", {"asin": "B000000001"}),
                                        message("search_products", {"query": "product"}, call_id="call2"),
                                        message("buy_now", call_id="call3")])
        self.assertTrue(env.released)
        self.assertEqual(env.actions, ["search[product]", "click[Buy Now]"])
        self.assertIn("可点击按钮: Buy Now", client.seen[1][-1]["content"])
        self.assertNotIn("DO_NOT_SHOW", json.dumps(client.seen))
        self.assertTrue(acceptance_reasons(row)[0])
        clean = build_sft_row(row)
        self.assertEqual(sum(item["role"] == "assistant" for item in clean["messages"]), 2)
        self.assertNotIn("private reward text", json.dumps(clean))

    def test_three_illegal_calls_are_model_failure_and_do_not_execute(self):
        row, env, _ = self.episode([message("open_product", {"asin": "B000000001"})] * 3)
        self.assertEqual(row["status"], "invalid_action_limit")
        self.assertEqual(row["failure_kind"], "model")
        self.assertEqual(env.actions, [])
        self.assertTrue(env.released)

    def test_multi_call_only_first_executes(self):
        reply = message("buy_now")
        reply["tool_calls"] += message("buy_now", call_id="second")["tool_calls"]
        row, env, _ = self.episode([reply])
        self.assertEqual(env.actions, ["click[Buy Now]"])
        self.assertEqual(len(row["tool_call_truncations"]), 1)

    def test_terminal_does_not_check_next_turn_budget(self):
        with patch.object(EpisodeBudget, "next_limit", side_effect=[(512, {})]):
            row, _, _ = self.episode([message("buy_now")])
        self.assertEqual(row["status"], "done")

    def test_truncated_model_output_not_sent_to_environment(self):
        row, env, _ = self.episode([ModelOutputError("length", {"finish_reason": "length"})])
        self.assertEqual(row["failure_kind"], "model")
        self.assertEqual(env.actions, [])
        self.assertTrue(env.released)

    def test_service_failure_is_infrastructure(self):
        row, env, _ = self.episode([ModelServiceError("HTTP 503")])
        self.assertEqual(row["failure_kind"], "infrastructure")
        self.assertTrue(env.released)

    def test_unknown_public_state_gold_is_rejected(self):
        with self.assertRaises(ValueError):
            render_structured_observation({"observation_version": "shopping-observation-v2", "goal": "secret"})


class ClientTests(unittest.TestCase):
    def test_teacher_profile_preserves_student_execution_contract(self):
        astra, sol = load_config(), load_config(teacher_profile="sol")
        self.assertEqual(sol["teacher"]["model"], "GPT-6.1-Sol")
        self.assertNotEqual(astra["teacher"]["expected_response_model"], sol["teacher"]["expected_response_model"])
        for name in ("student", "budgets", "observation", "collection", "reward", "tool_version"):
            self.assertEqual(astra[name], sol[name])

    def test_short_429_policy_is_bounded_at_twelve_attempts(self):
        calls = []
        def transport(*_):
            calls.append(1)
            raise HTTPError("https://example.test", 429, "capacity", {}, io.BytesIO(b"pool full"))
        with patch("shopping_agent.clients.time.sleep") as sleep, self.assertRaises(ModelServiceError) as caught:
            request_json("https://example.test", {}, headers={}, timeout=1, transport=transport,
                         rate_limit_attempts=12, rate_limit_initial_delay_seconds=2,
                         rate_limit_max_delay_seconds=20)
        self.assertEqual(len(calls), 12)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4, 8, 16] + [20] * 7)
        self.assertEqual(caught.exception.audit["transport_attempts"], 12)

    def test_extended_429_retries_recover_without_changing_payload(self):
        calls = []
        payload = {"messages": [{"role": "user", "content": "public task"}]}
        def transport(_url, body, *_):
            calls.append(body)
            if len(calls) < 6:
                raise HTTPError("https://example.test", 429, "capacity", {}, io.BytesIO(b"pool full"))
            return {"ok": True}
        with patch("shopping_agent.clients.time.sleep") as sleep:
            result = request_json("https://example.test", payload, headers={}, timeout=1,
                                  transport=transport, rate_limit_attempts=10)
        self.assertEqual(result["_transport_attempts"], 6)
        self.assertTrue(all(body is payload for body in calls))
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [10, 20, 40, 80, 120])

    def test_extended_429_budget_remains_bounded_and_does_not_extend_auth_or_timeout(self):
        for code, body, expected in ((429, b"pool full", 10), (403, b"invalid key", 1),
                                      (503, b"unavailable", 3)):
            calls = []
            def transport(*_):
                calls.append(1)
                raise HTTPError("https://example.test", code, "failure", {}, io.BytesIO(body))
            with patch("shopping_agent.clients.time.sleep"), self.assertRaises(ModelServiceError) as caught:
                request_json("https://example.test", {}, headers={}, timeout=1,
                             transport=transport, rate_limit_attempts=10)
            self.assertEqual(len(calls), expected)
            self.assertEqual(caught.exception.audit["transport_attempts"], expected)
            self.assertEqual(caught.exception.audit["transport_errors"][-1]["retry_delay_seconds"], 0)

    def test_gateway_response_check_timeout_recovers_and_audits_retries(self):
        calls = []
        def transport(*_):
            calls.append(1)
            if len(calls) < 3:
                body = b'[Compliance Gateway HTTP] check response failed: dial tcp 127.0.0.1:1233: i/o timeout'
                raise HTTPError("https://example.test", 403, "forbidden", {}, io.BytesIO(body))
            return {"ok": True}
        with patch("shopping_agent.clients.time.sleep") as sleep:
            result = request_json("https://example.test", {}, headers={}, timeout=1, transport=transport)
        self.assertTrue(result["ok"])
        self.assertEqual(result["_transport_attempts"], 3)
        self.assertEqual([error["kind"] for error in result["_transport_errors"]], ["gateway_timeout"] * 2)
        self.assertEqual(sleep.call_args_list, [((5,),), ((10,),)])

    def test_persistent_gateway_timeout_stops_after_three_attempts(self):
        calls = []
        def transport(*_):
            calls.append(1)
            body = b'[Compliance Gateway HTTP] check response failed: i/o timeout'
            raise HTTPError("https://example.test", 403, "forbidden", {}, io.BytesIO(body))
        with patch("shopping_agent.clients.time.sleep"), self.assertRaises(ModelServiceError) as caught:
            request_json("https://example.test", {}, headers={}, timeout=1, transport=transport)
        self.assertEqual(len(calls), 3)
        self.assertEqual(caught.exception.audit["transport_attempts"], 3)
        self.assertEqual(len(caught.exception.audit["transport_errors"]), 3)
        self.assertEqual(caught.exception.audit["transport_errors"][-1]["retry_delay_seconds"], 0)

    def test_observed_gateway_dns_403_retries_but_auth_403_does_not(self):
        calls = []
        def transient(*_):
            calls.append(1)
            if len(calls) < 3:
                body = b'[Compliance Gateway HTTP] Forward to backend error: dial tcp: lookup host: i/o timeout'
                raise HTTPError("https://example.test", 403, "forbidden", {}, io.BytesIO(body))
            return {"ok": True}
        with patch("shopping_agent.clients.time.sleep"):
            result = request_json("https://example.test", {}, headers={}, timeout=1, transport=transient)
        self.assertEqual(result["_transport_attempts"], 3)
        self.assertEqual(len(calls), 3)
        calls.clear()
        def unauthorized(*_):
            calls.append(1)
            raise HTTPError("https://example.test", 403, "forbidden", {}, io.BytesIO(b"invalid API key"))
        with patch("shopping_agent.clients.time.sleep"), self.assertRaises(ModelServiceError):
            request_json("https://example.test", {}, headers={}, timeout=1, transport=unauthorized)
        self.assertEqual(len(calls), 1)

    def test_http_failure_redacts_key_and_credential_url(self):
        error = HTTPError("https://example.test?ak=SECRET_KEY", 400, "bad request", {}, io.BytesIO(b"error: SECRET_KEY"))
        with patch("shopping_agent.clients.urlopen", side_effect=error):
            with self.assertRaises(ModelServiceError) as caught:
                request_json("https://example.test?ak=SECRET_KEY", {}, headers={}, timeout=1, secret="SECRET_KEY")
        self.assertNotIn("SECRET_KEY", str(caught.exception))
        self.assertNotIn("https://", str(caught.exception))

    def test_high_reasoning_structured_adapter_uses_same_schema_and_strips_private_fields(self):
        settings = load_config()["teacher"]
        seen = {}

        def transport(url, body, headers, timeout):
            seen.update(body=body, headers=headers, url=url)
            return {"model": settings["expected_response_model"], "choices": [{"finish_reason": "stop", "message": {
                "content": '{"action":{"name":"buy_now","arguments":{}}}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}}

        client = ModelHubClient(settings, key="TEST_PRIVATE_KEY", transport=transport)
        reply = client.complete([{"role": "system", "content": "rules"},
                                 {"role": "tool", "content": "public", "runtime_action_guard": True}], SHOP_TOOL_SCHEMAS, 512)
        self.assertNotIn("tools", seen["body"])
        self.assertEqual(seen["body"]["reasoning_effort"], "high")
        self.assertIn("X-TT-LOGID", seen["headers"])
        self.assertNotIn("runtime_action_guard", json.dumps(seen["body"]))
        self.assertNotIn("TEST_PRIVATE_KEY", json.dumps(reply.audit))
        self.assertEqual(reply.message["tool_calls"][0]["function"]["name"], "buy_now")
        schema = seen["body"]["response_format"]["json_schema"]["schema"]
        self.assertEqual(len(schema["properties"]["action"]["anyOf"]), len(SHOP_TOOL_SCHEMAS))

    def test_changed_snapshot_is_not_silently_accepted(self):
        def transport(*_):
            return {"model": "different", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
        with self.assertRaises(ModelServiceError):
            ModelHubClient(load_config()["teacher"], key="private", transport=transport).complete([], SHOP_TOOL_SCHEMAS, 512)


class ArtifactTests(unittest.TestCase):
    def seed_specification(self):
        return {**self.specification([1, 2]), "kind": "collect", "workers": 4,
                "environment_url": "http://localhost:5701", "teacher": {"model": "teacher", "rate_limit_attempts": 3}}

    def test_inherited_evidence_is_read_only_and_new_raw_contains_only_new_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / "old", Path(temporary) / "new"
            original = self.seed_specification()
            with RunStore(source, original) as store:
                store.append(self.row())
            original_raw = (source / "raw.jsonl").read_bytes()
            updated = {**original, "execution_fingerprint": "updated", "workers": 6,
                       "environment_url": "http://localhost:5702",
                       "teacher": {"model": "teacher", "rate_limit_attempts": 12,
                                   "rate_limit_initial_delay_seconds": 2, "rate_limit_max_delay_seconds": 20}}
            with RunStore(destination, updated, seed_from=source) as store:
                self.assertEqual(len(store.rows), 1)
                row = self.row(2); row["execution_fingerprint"] = "updated"; store.append(row)
            with RunStore(destination, updated) as store:
                self.assertEqual([row["task_id"] for row in store.rows], [1, 2])
            self.assertEqual((source / "raw.jsonl").read_bytes(), original_raw)
            self.assertEqual(len(read_jsonl(destination / "raw.jsonl")), 1)
            (source / "raw.jsonl").write_bytes(original_raw + b'\n')
            with self.assertRaises(ValueError), RunStore(destination, updated):
                pass

    def test_seed_cannot_mix_teachers_or_change_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "old"
            original = self.seed_specification()
            with RunStore(source, original) as store:
                store.append(self.row())
            for name, changed in (("teacher", {**original, "teacher": {"model": "other"}}),
                                  ("tasks", {**original, "candidate_task_ids": [1]})):
                with self.assertRaises(ValueError), RunStore(Path(temporary) / name, changed, seed_from=source):
                    pass

    def transport_receipt(self, directory, original, updated):
        raw = directory / "raw.jsonl"
        receipt = {"scope": "transport-only", "from_execution_fingerprint": original,
                   "to_execution_fingerprint": updated, "runs": {directory.name: {
                       "run_sha256": sha256(directory / "run.json"),
                       "raw_prefix": {"bytes": raw.stat().st_size, "sha256": sha256(raw), "rows": 1}}}}
        (directory.parent / "transport-update.json").write_text(json.dumps(receipt))

    def test_reviewed_transport_resume_preserves_original_rows_and_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "train"
            original = self.specification([1, 2])
            with RunStore(directory, original) as store:
                store.append(self.row())
            original_run = (directory / "run.json").read_bytes()
            original_raw = (directory / "raw.jsonl").read_bytes()
            self.transport_receipt(directory, "frozen", "updated")
            updated = {**original, "execution_fingerprint": "updated"}
            with RunStore(directory, updated) as store:
                row = self.row(2); row["execution_fingerprint"] = "updated"
                store.append(row)
            with RunStore(directory, updated) as store:
                self.assertEqual([row["execution_fingerprint"] for row in store.rows], ["frozen", "updated"])
            self.assertEqual((directory / "run.json").read_bytes(), original_run)
            self.assertTrue((directory / "raw.jsonl").read_bytes().startswith(original_raw))

    def test_transport_receipt_cannot_change_tasks_or_hide_modified_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "train"
            original = self.specification([1])
            with RunStore(directory, original) as store:
                store.append(self.row())
            self.transport_receipt(directory, "frozen", "updated")
            updated = {**original, "execution_fingerprint": "updated", "candidate_task_ids": [2]}
            with self.assertRaises(ValueError), RunStore(directory, updated):
                pass
            raw = directory / "raw.jsonl"
            raw.write_text(raw.read_text().replace('"task_id": 1', '"task_id": 9'))
            with self.assertRaises(ValueError), RunStore(directory, {**original, "execution_fingerprint": "updated"}):
                pass

    def test_frozen_roles_protect_dev_test_and_validation(self):
        pool = TaskPool()
        train = {row["task_id"] for row in pool.tasks("teacher_candidates")}
        validation = {row["task_id"] for row in pool.tasks("sft_validation_candidates")}
        self.assertFalse(train & pool.protected_ids("train"))
        self.assertTrue(validation <= pool.protected_ids("train"))
        self.assertFalse(validation & pool.protected_ids("validation"))
        with tempfile.TemporaryDirectory() as directory:
            subset = Path(directory) / "wrong-role.jsonl"
            subset.write_text(json.dumps({"task_id": next(iter(validation))}) + "\n")
            with self.assertRaises(ValueError):
                selected_tasks(pool, "teacher_candidates", subset)

    def row(self, task_id=1, attempt=0, kind=None):
        row, _, _ = ExecutionTests().episode([message("search_products", {"query": "product"}), message("buy_now", call_id="call2")])
        row.update(task_id=task_id, attempt_index=attempt, failure_kind=kind)
        if kind:
            row.update(status="infrastructure_error", done=False, error={"type": "ServiceError"})
        return row

    def specification(self, ids):
        return {"candidate_task_ids": ids, "execution_fingerprint": "frozen", "role": "train", "split": "test", "formal": True}

    @patch("shopping_agent.runs.build_supervised_example", return_value={"input_ids": [1, 2], "labels": [-100, 2]})
    def test_unique_trainable_task_count_and_resume(self, _):
        with tempfile.TemporaryDirectory() as directory, RunStore(directory, self.specification([1, 2])) as store:
            store.append(self.row())
            store.append(self.row(attempt=1))
            progress = CollectionProgress(store, domains={1: "x", 2: "x"}, quotas={"x": 2}, tokenizer=Tokenizer(), max_length=24576)
            self.assertEqual(progress.publish("incomplete")["accepted_unique_tasks"], 1)
            self.assertIn("duplicate_accepted_task", progress.rejected[0]["reasons"])

    @patch("shopping_agent.runs.build_supervised_example", return_value=None)
    def test_nontrainable_success_does_not_satisfy_target(self, _):
        with tempfile.TemporaryDirectory() as directory, RunStore(directory, self.specification([1])) as store:
            store.append(self.row())
            progress = CollectionProgress(store, domains={1: "x"}, quotas={"x": 1}, tokenizer=Tokenizer(), max_length=2)
            self.assertEqual(progress.publish("incomplete")["accepted_unique_tasks"], 0)

    @patch("shopping_agent.runs.build_supervised_example", return_value={"input_ids": [1], "labels": [1]})
    def test_bounded_collection_does_not_overshoot_target(self, _):
        with tempfile.TemporaryDirectory() as directory, RunStore(directory, self.specification([1, 2, 3])) as store:
            calls = []
            def episode(task, attempt_index):
                calls.append(task["task_id"])
                return self.row(task["task_id"], attempt_index)
            with patch("sys.stdout", new=io.StringIO()):
                summary = collect(store, tasks=[{"task_id": n} for n in [1, 2, 3]], domains={n: "x" for n in [1, 2, 3]},
                                  quotas={"x": 1}, tokenizer=Tokenizer(), config=load_config(), episode=episode, workers=4)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(len(calls), 1)

    def test_evaluation_keeps_fixed_denominator_on_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory, RunStore(directory, self.specification([1, 2])) as store:
            with patch("sys.stdout", new=io.StringIO()):
                result = evaluate(store, tasks=[{"task_id": 1}, {"task_id": 2}],
                                  episode=lambda task, **_: self.row(task["task_id"], kind="infrastructure"), workers=1)
            self.assertEqual(result["expected_tasks"], 2)
            self.assertEqual(result["status"], "infrastructure_error")
            self.assertEqual(result["strict_success_rate"], 0)
            self.assertGreater(result["context_projection"]["max_context_input_tokens"], 0)

    def test_resume_rejects_changed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with RunStore(directory, self.specification([1])):
                pass
            with self.assertRaises(ValueError), RunStore(directory, self.specification([2])):
                pass

    def test_domain_quota_largest_remainder_and_limit(self):
        self.assertEqual(domain_quotas([{"task_id": n} for n in range(4)], {0: "a", 1: "a", 2: "a", 3: "b"}, 2), {"a": 2, "b": 0})
        with self.assertRaises(ValueError):
            domain_quotas([{"task_id": 1}], {1: "x"}, 2)


if __name__ == "__main__":
    unittest.main()
