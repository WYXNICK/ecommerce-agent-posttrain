"""SDPO acceptance: feedback provenance, causal alignment, loss and EMA gradients."""
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from rollout import collect_update
from settings import load_settings, starting_candidate
from shopping_agent.training.sft.dataset import normalize_messages_for_chat_template
from self_distillation import (EMATeacher, distillation_loss, execution_feedback,
                               hindsight_logits, prepare_group)


class Tokenizer:
    def text_tokens(self, text):
        return len(text)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        # Content-sensitive length makes alignment errors visible.
        return [1] * len(json.dumps(messages))


def trajectory(index, success=False, failure='model'):
    messages = [{'role': 'system', 'content': 'act'}, {'role': 'user', 'content': 'buy 42'},
                {'role': 'assistant', 'tool_calls': [{'id': 'c', 'type': 'function',
                  'function': {'name': 'inspect', 'arguments': '{"id":"a"}'}}]},
                {'role': 'tool', 'tool_call_id': 'c', 'content': 'current page; size 42 available'}]
    detail = {'reward_version': 'shopsimulator-reward-v3',
              'reward_type': 'gold_purchase' if success else 'wrong_purchase',
              'reward_valid': True, 'purchase_success': success,
              'termination_reason': 'gold_purchase' if success else 'wrong_purchase'}
    return {'trajectory_id': str(index), 'task_id': 1, 'status': 'done', 'done': True,
            'terminal_result': {'done': True, 'over': True, 'reward_detail': detail},
            'failure_kind': None if success else failure, 'training_reward': 1. if success else -.65,
            'final_reward': 1. if success else -.65,
            'steps': [{'tool_name': 'buy', 'parameters': {'button': 'buy now'}}],
            'blocked_tool_calls': [], 'messages': messages,
            'rollout_turns': [{'messages': messages, 'prompt_ids': [1, 2],
                               'response_ids': [3, 4], 'old_logprobs': [-.1, -.2]}]}


class SDPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.settings = load_settings('sdpo')
        self.cfg = self.settings['self_distillation']

    def test_settings_start_from_selected_repair_and_increase_group_exploration(self):
        self.assertEqual(starting_candidate('sdpo'), starting_candidate('outcome'))
        self.assertEqual(self.settings['base_weights_sha256'], load_settings()['base_weights_sha256'])
        self.assertEqual(self.settings['rollouts_per_group'], 8)
        self.assertEqual(self.settings['sampling_groups_per_batch'], 1)
        self.assertEqual(self.settings['groups_per_update'], 2)
        self.assertNotIn('self_distillation', load_settings())

    def test_verified_feedback_excludes_hidden_target_and_unverifiable_checks(self):
        row = trajectory(0)
        row['terminal_result']['reward_detail'].update(target_asin='SECRET',
            hard_gates={'budget': {'passed': False, 'verifiable': True, 'required': 100, 'actual': 120},
                        'unknown': {'passed': False, 'verifiable': False, 'required': 'SECRET'}})
        text = json.dumps(execution_feedback(row))
        self.assertIn('120', text)
        self.assertNotIn('SECRET', text)
        self.assertIsNone(execution_feedback(trajectory(1, success=True)))
        for kind in ('infrastructure', 'unverifiable'):
            self.assertIsNone(execution_feedback(trajectory(2, failure=kind)))

    def test_leave_one_out_and_student_context_and_targets_unchanged(self):
        tok, counts = Tokenizer(), Counter()
        group = [trajectory(0, True), trajectory(1)]
        before = deepcopy(group)
        rows = prepare_group(group, tok, self.settings, counts)
        # The only successful trajectory cannot teach itself.
        self.assertEqual([r['trajectory_id'] for r in rows], ['1'])
        self.assertEqual(rows[0]['distillation']['demonstration_id'], '0')
        self.assertEqual(tok.messages[2:], normalize_messages_for_chat_template(before[1]['messages'])[2:])
        self.assertNotEqual(tok.messages[1], before[1]['messages'][1])
        for actual, original in zip(group, before):
            self.assertEqual(actual['messages'], original['messages'])
            self.assertEqual(actual['rollout_turns'], original['rollout_turns'])
        for field in ('prompt_ids', 'response_ids', 'old_logprobs'):
            self.assertEqual(rows[0]['rollout_turns'][0][field], before[1]['rollout_turns'][0][field])

    def test_equal_groups_and_valid_siblings_are_not_discarded(self):
        group = [trajectory(i, True) for i in range(8)]
        group[0].update(failure_kind='infrastructure', training_reward=None)
        rows = prepare_group(group, Tokenizer(), self.settings, Counter())
        self.assertEqual(len(rows), 7)
        for row in rows:
            self.assertNotEqual(row['distillation']['demonstration_id'], row['trajectory_id'])
            self.assertNotEqual(row['distillation']['demonstration_id'], '0')
        failures = [trajectory(i) for i in range(8)]
        rows = prepare_group(failures, Tokenizer(), self.settings, Counter())
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r['distillation']['environment_feedback'] for r in rows))

    def test_bounded_sampler_uses_16_trajectories_and_archives_feedback(self):
        def factory(seed):
            return SimpleNamespace(seed=seed, cost=Counter(requests=1), turns=trajectory(seed)['rollout_turns'])
        def episode(task, client, **kwargs):
            return {**trajectory(client.seed, True), 'task_id': task['task_id']}
        with tempfile.TemporaryDirectory() as tmp, patch('rollout.run_episode', side_effect=episode):
            rows, diag = collect_update(iter({'task_id': i} for i in range(30)),
                client_factory=factory, config={}, tokenizer=Tokenizer(), env_url='', fingerprint='',
                settings=self.settings, generation=1, evidence=Path(tmp), group_transform=prepare_group)
            with gzip.open(Path(tmp)/'generation-000001.jsonl.gz', 'rt') as f:
                evidence = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 16)
        self.assertEqual(diag['generation_batches'], 2)
        self.assertEqual(diag['counts']['accepted_groups'], 2)
        self.assertTrue(all(r['distillation']['demonstration_id'] for r in evidence))

    def test_context_overflow_is_masked_not_silently_truncated(self):
        settings = deepcopy(self.settings)
        settings['self_distillation']['teacher_context_tokens'] = 3
        counts = Counter()
        self.assertEqual(prepare_group([trajectory(0)], Tokenizer(), settings, counts), [])
        self.assertEqual(counts['sdpo_context_overflow_turns'], 1)

    def test_chunked_full_kl_matches_direct_value_and_gradient_with_capped_weights(self):
        torch.manual_seed(6)
        logits = torch.randn(7, 17, requires_grad=True)
        teacher = torch.randn(7, 17, requires_grad=True)
        targets = torch.arange(7)
        old = F.log_softmax(logits.detach(), -1)[torch.arange(7), targets] - 2
        actual, stats = distillation_loss(logits, teacher, targets, old,
                                          {**self.cfg, 'logit_chunk_tokens': 2})
        p, q = logits.log_softmax(-1), teacher.detach().log_softmax(-1)
        expected = F.kl_div(q, p, reduction='none', log_target=True).sum(-1).mean()*2
        torch.testing.assert_close(actual, expected)
        correct = torch.autograd.grad(expected, logits, retain_graph=True)[0]
        actual.backward()
        torch.testing.assert_close(logits.grad, correct)
        self.assertIsNone(teacher.grad)
        self.assertEqual(stats['importance_clipped_fraction'], 1.)

    def test_zero_divergence_and_gradient_direction(self):
        logits = torch.tensor([[0., 0., 0.]], requires_grad=True)
        target, old = torch.tensor([0]), torch.tensor([-1.0986123])
        loss, _ = distillation_loss(logits, logits.detach(), target, old, self.cfg)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        logits.grad = None
        loss, _ = distillation_loss(logits, torch.tensor([[4., 0., 0.]]), target, old, self.cfg)
        loss.backward()
        self.assertLess(logits.grad[0, 0], 0.)  # Gradient descent increases teacher-favoured action.
        self.assertGreater(logits.grad[0, 1], 0.)

    def test_two_rank_sum_matches_global_token_normalization(self):
        torch.manual_seed(3)
        logits, teacher = torch.randn(7, 11), torch.randn(7, 11)
        targets = torch.arange(7)
        old = logits.log_softmax(-1)[torch.arange(7), targets]
        def gradient(a, b):
            weight = torch.zeros(11, requires_grad=True)
            loss, _ = distillation_loss(logits[a:b]+weight, teacher[a:b], targets[a:b], old[a:b], self.cfg)
            (loss*(b-a)/7).backward()
            return weight.grad
        torch.testing.assert_close(gradient(0, 7), gradient(0, 2)+gradient(2, 7))

    def test_real_causal_model_ema_restoration_backward_and_resume(self):
        torch.manual_seed(9)
        model = get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=17, hidden_size=16,
            intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, attention_dropout=0., use_cache=False)),
            LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4, lora_dropout=0.,
                       target_modules=['q_proj', 'v_proj'], bias='none'))
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        teacher = EMATeacher(model, .1)
        initial = teacher.state_dict()
        with torch.no_grad():
            for name, parameter in teacher.parameters.items():
                if 'lora_B' in name:
                    parameter.fill_(.2)
        student = {n: p.detach().clone() for n, p in teacher.parameters.items()}
        turn = {'teacher_prompt_ids': [1, 2, 5, 6], 'response_ids': [3, 4]}
        q = hindsight_logits(model, teacher, turn, 'cpu')
        self.assertEqual(q.shape, (2, 17))
        with teacher.activate(model), torch.no_grad():
            full = model(input_ids=torch.tensor([[1, 2, 5, 6, 3, 4]])).logits[0]
        torch.testing.assert_close(q, full[3:5])
        self.assertFalse(q.requires_grad)
        self.assertTrue(model.training)
        for name, parameter in teacher.parameters.items():
            torch.testing.assert_close(parameter, student[name], rtol=0, atol=0)
        p = model(input_ids=torch.tensor([[1, 2, 3, 4]]), logits_to_keep=torch.tensor([1, 2])).logits[0]
        targets = torch.tensor([3, 4])
        old = p.detach().log_softmax(-1)[torch.arange(2), targets]
        loss, _ = distillation_loss(p, q, targets, old, self.cfg)
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in teacher.parameters.values()), 0)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if n not in teacher.parameters))
        teacher.update()
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, initial[name]*.9+student[name]*.1)
        restored = EMATeacher(model, .1, teacher.state_dict())
        torch.testing.assert_close(hindsight_logits(model, teacher, turn, 'cpu'),
                                   hindsight_logits(model, restored, turn, 'cpu'), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
