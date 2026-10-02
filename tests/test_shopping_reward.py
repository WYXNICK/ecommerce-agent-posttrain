"""Shopping GRPO 终局奖励的纯函数测试。"""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    reward_breakdown,
    validate_reward,
)


def terminal_state(*, steps=8, reward_type="gold_purchase", native_reward=1.0):
    valid = reward_type != "reward_unverifiable"
    detail = {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": reward_type, "termination_reason": reward_type,
        "reward_valid": valid, "sampling_invalid": not valid,
        "terminal_utility": native_reward,
        "purchase_success": reward_type in {"gold_purchase", "valid_alternative_purchase"},
        "hard_gates": {}, "weighted_score": 1.0, "evidence_coverage": 1.0,
        "dimension_scores": {},
    }
    state = make_runtime_state(task_id=1, max_steps=35)
    state.update(validate_reward(detail))
    state.update({
        "steps": [{"index": index} for index in range(steps)],
        "done": True, "terminal_result": {"done": True, "over": True},
        "final_reward": native_reward, "reward_detail": detail,
    })
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_model_nonterminals_remain_trainable_and_environment_score_unchanged(self):
        for reason in ('invalid_action_limit', 'assistant_final', 'context_hard_limit_exceeded',
                       'response_budget_exhausted', 'max_steps'):
            state = make_runtime_state(1, 35)
            state.update(termination_reason=reason, error=reason, terminate=True)
            reward = reward_breakdown(state)
            self.assertEqual(reward['training_reward'], -0.65)
            self.assertEqual(reward['terminal_utility'], 0.0)
            self.assertEqual(reward['native'], 0.0)
            self.assertFalse(reward['sampling_invalid'])
            self.assertFalse(reward['reward_unverifiable'])
            state['infrastructure_invalid'] = True
            invalid = reward_breakdown(state)
            self.assertTrue(invalid['sampling_invalid'])
            self.assertEqual(invalid['training_reward'], 0.0)

    def test_gold_purchase_preserves_terminal_reward_without_efficiency_bonus(self):
        for steps in (8, 35):
            with self.subTest(steps=steps):
                result = reward_breakdown(terminal_state(steps=steps))
                self.assertEqual(result["full"], 1.0)
                self.assertEqual(result["strict"], 1.0)
                self.assertEqual(result["total"], 1.0)
                self.assertEqual(result["efficiency"], 0.0)
                self.assertFalse(result["sampling_invalid"])

    def test_alternative_purchase_is_not_strict_gold_success(self):
        result = reward_breakdown(terminal_state(
            reward_type="valid_alternative_purchase", native_reward=0.55,
        ))
        self.assertEqual(result["strict"], 0.0)
        self.assertEqual(result["purchase_success"], 1.0)
        self.assertEqual(result["total"], 0.55)

    def test_nonpurchase_terminal_preserves_negative_utility(self):
        result = reward_breakdown(terminal_state(
            reward_type="early_abstain", native_reward=-0.35,
        ))
        self.assertEqual(result["strict"], 0.0)
        self.assertEqual(result["purchase_success"], 0.0)
        self.assertEqual(result["total"], -0.35)

    def test_unverifiable_terminal_has_no_training_signal(self):
        result = reward_breakdown(terminal_state(
            reward_type="reward_unverifiable", native_reward=0.0,
        ))
        self.assertTrue(result["sampling_invalid"])
        self.assertTrue(result["reward_unverifiable"])
        self.assertFalse(result["infrastructure_invalid"])
        self.assertEqual(result["total"], 0.0)

    def test_missing_reward_contract_fails_closed(self):
        result = reward_breakdown(make_runtime_state(task_id=1, max_steps=35))
        self.assertTrue(result["infrastructure_invalid"])
        self.assertTrue(result["sampling_invalid"])
        self.assertEqual(result["total"], 0.0)

    def test_malformed_reward_contract_is_rejected(self):
        detail = terminal_state()["reward_detail"]
        for updates, error in [
            ({"reward_version": "old"}, "unsupported"),
            ({"terminal_utility": float("nan")}, "finite"),
            ({"weighted_score": 1.1}, "finite and in"),
            ({"reward_valid": False}, "only reward_unverifiable"),
            ({"sampling_invalid": True}, "sampling_invalid"),
        ]:
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, error):
                validate_reward({**detail, **updates})

    def test_same_action_on_same_page_within_three_attempts_is_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")
        record_action_attempt(state, "open_product", {"asin": "123"}, "search page")
        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")

        self.assertEqual(state["action_attempt_count"], 3)
        self.assertEqual(state["repeat_action_count"], 1)
        self.assertAlmostEqual(reward_breakdown(state)["repeat_action_rate"], 1 / 3)

    def test_different_parameters_or_page_are_not_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "cup"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "mug"}, "page 2")

        self.assertEqual(state["repeat_action_count"], 0)

    def test_think_is_not_an_environment_action_attempt(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "think", {"note": "plan"}, "page")

        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["recent_action_signatures"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
