"""Necessary tests for terminal returns, exact action tokens and gradient scaling."""
import unittest
from unittest.mock import patch

from rollout import TrainingClient, group_advantages, training_signal
from shopping_agent.clients import ModelOutputError


class ContractTests(unittest.TestCase):
    def test_failure_returns_and_group_baseline(self):
        row = {'failure_kind': 'model', 'done': False, 'final_reward': 0}
        self.assertEqual(training_signal(row, -.65), -.65)
        self.assertEqual(training_signal({**row, 'done': True, 'final_reward': -.4}, -.65), -.4)
        self.assertIsNone(training_signal({**row, 'failure_kind': 'infrastructure'}, -.65))
        self.assertIsNone(training_signal({**row, 'failure_kind': 'unverifiable'}, -.65))
        self.assertEqual(group_advantages([1, 1, 1, 1]), [0, 0, 0, 0])
        a = group_advantages([1, 1, 0, -.65])
        self.assertAlmostEqual(sum(a), 0)
        self.assertLess(a[-1], 0)

    def test_invalid_generated_tokens_are_retained(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [1, 2] if kwargs.get('return_dict') is False else {'input_ids': [1, 2]}
        client = TrainingClient('http://unused', 'policy', Tokenizer(), 42,
                                {'temperature': 1, 'top_p': 1})
        response = {'prompt_token_ids': [1, 2], 'choices': [{'token_ids': [3, 4],
            'logprobs': {'content': [{'token': 'token_id:3', 'logprob': -.1},
                                    {'token': 'token_id:4', 'logprob': -.2}]},
            'finish_reason': 'length', 'message': {'role': 'assistant', 'content': 'bad'}}]}
        with patch('rollout.request_json', return_value=response), self.assertRaises(ModelOutputError):
            client.complete([{'role': 'user', 'content': 'test'}], [], 2)
        self.assertEqual(client.turns[0]['response_ids'], [3, 4])
        response['prompt_token_ids'] = [9]
        with patch('rollout.request_json', return_value=response), self.assertRaises(ValueError):
            client.complete([{'role': 'user', 'content': 'test'}], [], 2)

    def test_rank_sum_matches_global_token_objective(self):
        import torch
        from omegaconf import OmegaConf
        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
        cfg = OmegaConf.create({'clip_ratio': .2, 'clip_ratio_low': None,
                                'clip_ratio_high': None, 'global_batch_info': {}})
        old = torch.tensor([-.5, -.7, -.2, -.4, -.8])
        adv = torch.tensor([.3, .3, -.6, -.6, -.6])
        base = old + .01
        weight = torch.tensor(0., requires_grad=True)
        loss, _ = compute_policy_loss_vanilla(old[None], (base+weight)[None], adv[None],
                                             torch.ones_like(old)[None], config=cfg)
        loss.backward()
        expected = weight.grad.item()
        gradients = []
        for a,b in [(0,2),(2,5)]:
            w = torch.tensor(0., requires_grad=True)
            local, _ = compute_policy_loss_vanilla(old[None,a:b], (base[a:b]+w)[None],
                adv[None,a:b], torch.ones(1,b-a), config=cfg)
            (local*(b-a)/5).backward()
            gradients.append(w.grad.item())
        self.assertAlmostEqual(sum(gradients), expected, places=6)


if __name__ == '__main__':
    unittest.main()
