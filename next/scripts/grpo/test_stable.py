"""CPU acceptance of bounded sampling, frozen reference and actual KL gradients."""
from collections import Counter
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

from regularization import action_entropy, reference_kl, reference_logprobs
from rollout import collect_update
from settings import load_settings
from train import compare_successes


class StableTests(unittest.TestCase):
    def test_overlay_preserves_reward_and_baseline(self):
        base, stable = load_settings(), load_settings('stable')
        changed = {k for k in base if base[k] != stable[k]}
        self.assertEqual(changed, {'version', 'updates', 'groups_per_update',
            'max_generation_batches', 'kl_coefficient', 'save_every', 'evaluate_every'})
        self.assertEqual(stable['updates'] * stable['groups_per_update'] * stable['rollouts_per_group'], 8000)
        self.assertEqual(base, load_settings())
        self.assertNotIn('local_credit', stable)

    def sample(self, outcomes, settings=None):
        settings = settings or load_settings('stable')
        clients, concurrency = [], []
        class Executor:
            def __init__(self, max_workers):
                concurrency.append(max_workers)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def map(self, fn, jobs):
                return map(fn, jobs)
        def factory(seed):
            client = SimpleNamespace(seed=seed, cost=Counter(requests=1), turns=[
                {'prompt_ids': [1], 'response_ids': [2], 'old_logprobs': [-.5]}])
            clients.append(client)
            return client
        def episode(task, client, **kwargs):
            kind = outcomes.get(task['task_id'], 'varying')
            return {'task_id': task['task_id'], 'done': kind != 'model',
                'failure_kind': kind if kind in ('infrastructure', 'model') else None,
                'final_reward': 1. if kind == 'equal' else float((client.seed-settings['seed']) % 10 == 0)}
        with tempfile.TemporaryDirectory() as tmp, patch('rollout.run_episode', side_effect=episode), \
                patch('rollout.ThreadPoolExecutor', Executor):
            rows, diag = collect_update(iter({'task_id': i} for i in range(100)),
                client_factory=factory, config={}, tokenizer=None, env_url='', fingerprint='',
                settings=settings, generation=1, evidence=Path(tmp))
            with gzip.open(Path(tmp)/'generation-000001.jsonl.gz', 'rt') as f:
                evidence = [json.loads(line) for line in f]
        return rows, diag, concurrency, evidence

    def test_32_trajectories_in_waves_filtering_and_evidence(self):
        rows, diag, concurrency, evidence = self.sample({0: 'equal', 1: 'infrastructure'})
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({r['task_id'] for r in rows}), 8)
        self.assertEqual(concurrency, [8]*5)
        self.assertEqual(len(evidence), 40)
        self.assertEqual(diag['counts']['filtered_equal'], 1)
        self.assertEqual(diag['counts']['filtered_invalid'], 1)
        self.assertEqual({r['policy_generation'] for r in rows}, {1})
        for i in range(0, 32, 4):
            self.assertAlmostEqual(sum(r['advantage'] for r in rows[i:i+4]), 0.)

    def test_insufficient_batch_does_not_make_partial_update(self):
        settings = {**load_settings('stable'), 'max_generation_batches': 4}
        rows, diag, concurrency, evidence = self.sample({0: 'equal'}, settings)
        self.assertEqual(rows, [])
        self.assertEqual(len(concurrency), 4)
        self.assertEqual(diag['counts']['discarded_partial_groups'], 7)
        self.assertEqual(len(evidence), 32)

    def test_k3_plus_expected_gradient_matches_forward_kl(self):
        logits = torch.tensor([.2, -.1, .5], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(-1)
        ref = torch.tensor([.4, -.3, .1], dtype=torch.float64).log_softmax(-1).requires_grad_()
        estimator = (logp.detach().exp() * reference_kl(logp, ref)).sum()
        expected = (logp.exp() * (logp-ref.detach())).sum()
        actual_grad = torch.autograd.grad(estimator, logits, retain_graph=True)[0]
        correct_grad = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(estimator, expected)
        torch.testing.assert_close(actual_grad, correct_grad)
        self.assertIsNone(ref.grad)
        self.assertGreater(actual_grad.abs().sum().item(), 0)

    def test_frozen_reference_survives_actor_update_and_checkpoint_backward(self):
        torch.manual_seed(42)
        model = get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=17, hidden_size=16,
            intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, attention_dropout=0., use_cache=False)),
            LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4, lora_dropout=0.,
                       target_modules=['q_proj', 'v_proj'], bias='none'))
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        ids, pos, targets = torch.tensor([[1, 2, 3, 4]]), torch.tensor([1, 2]), torch.tensor([3, 4])
        reference = reference_logprobs(model, ids, pos, targets)
        self.assertTrue(model.training)
        self.assertTrue(model.get_model_status().enabled)
        self.assertFalse(reference.requires_grad)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.fill_(.1)
        output = model(input_ids=ids, logits_to_keep=pos).logits[0]
        logp = -torch.nn.functional.cross_entropy(output, targets, reduction='none')
        reference_kl(logp, reference).sum().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if 'lora_' not in n))
        after = reference_logprobs(model, ids, pos, targets)
        torch.testing.assert_close(reference, after, rtol=0, atol=0)
        with patch.object(model.base_model, 'forward', side_effect=RuntimeError('forward failed')):
            with self.assertRaises(RuntimeError):
                reference_logprobs(model, ids, pos, targets)
        self.assertTrue(model.training)
        self.assertTrue(model.get_model_status().enabled)

    def test_entropy_is_categorical_and_has_no_gradient(self):
        logits = torch.zeros(70, 7, requires_grad=True)
        entropy = action_entropy(logits)
        torch.testing.assert_close(entropy, torch.tensor(70.)*torch.tensor(7.).log())
        self.assertFalse(entropy.requires_grad)
        concentrated = torch.tensor([[30., -30., -30.]])
        self.assertLess(action_entropy(concentrated).item(), 1e-20)

    def test_combined_loss_rank_sum_equals_single_global_objective(self):
        cfg = OmegaConf.create({'clip_ratio': .2, 'clip_ratio_low': None,
                                'clip_ratio_high': None, 'global_batch_info': {}})
        logits = torch.tensor([[.1, .3, -.2]]*5)
        old = logits.log_softmax(-1)[:, 0]
        ref = torch.tensor([-.7]*5)
        adv = torch.tensor([.3, .3, -.6, -.6, -.6])
        def gradient(parts, beta):
            w = torch.tensor([.02, .01, -.01], requires_grad=True)
            for a, b in parts:
                logp = (logits[a:b]+w).log_softmax(-1)[:, 0]
                loss, _ = compute_policy_loss_vanilla(old[None,a:b], logp[None],
                    adv[None,a:b], torch.ones(1,b-a), config=cfg)
                ((loss+beta*reference_kl(logp, ref[a:b]).mean())*(b-a)/5).backward()
            return w.grad
        expected = gradient([(0,5)], .001)
        actual = gradient([(0,2)], .001)+gradient([(2,5)], .001)
        torch.testing.assert_close(expected, actual)
        self.assertGreater((expected-gradient([(0,5)], 0)).abs().sum().item(), 0.)

    def test_development_gain_and_regression_are_separate(self):
        comparison = compare_successes({'strict_success_task_ids': [2, 3, 4]},
                                       {'strict_success_task_ids': [1, 2, 3]})
        self.assertEqual(comparison, {'gained_task_ids': [4], 'lost_task_ids': [1],
                                      'gained': 1, 'lost': 1})


if __name__ == '__main__':
    unittest.main()
