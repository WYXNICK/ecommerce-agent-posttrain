"""Shopping SDPO: verified hindsight, EMA self-teacher, on-policy logit loss.

Objective follows lasgroup/SDPO (commit in configs/sdpo.json). This module
implements the equations independently; the environment and actor stay shared.
"""
from contextlib import contextmanager
from copy import deepcopy
import json

from shopping_agent.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_agent.evaluation.summary import _is_strict_success
from shopping_agent.training.sft.dataset import normalize_messages_for_chat_template


def action_trace(row):
    return [{'tool': s['tool_name'], 'arguments': s['parameters']} for s in row['steps']]


def execution_feedback(row):
    """Only returned, verifiable outcomes; never hidden target IDs or guessed causes."""
    if row['failure_kind'] in {'infrastructure', 'unverifiable'} or _is_strict_success(row):
        return None
    detail = row['terminal_result'].get('reward_detail') or {}
    checks = dict(detail.get('hard_gates') or {})
    dimensions = (detail.get('evidence') or {}).get('preference_scoring', {}).get('dimensions', {})
    failures = []
    for dimension, item in dimensions.items():
        for index, check in enumerate(item.get('results', [])):
            checks[f'{dimension}/{index}'] = check
    for name, check in checks.items():
        if check.get('verifiable') is True and check.get('passed') is False:
            failures.append({'constraint': name, 'required': check.get('required'),
                             'actual': check.get('actual')})
    return {'status': row['status'], 'outcome': detail.get('reward_type'),
            'strict_success': False, 'failed_checks': failures,
            'rejected_actions': [{'action': b['tool_call']['function'], 'reason': b['reason']}
                                 for b in row['blocked_tool_calls']],
            'executed_actions': action_trace(row)}


def feedback_text(feedback, demonstration):
    parts = []
    if demonstration is not None:
        parts.append('同一任务另一条已实际执行、通过严格购买验收的轨迹：\n' +
                     json.dumps(action_trace(demonstration), ensure_ascii=False, separators=(',', ':')))
    if feedback is not None:
        parts.append('本次尝试执行完毕后的真实反馈：\n' +
                     json.dumps(feedback, ensure_ascii=False, separators=(',', ':')))
    if not parts:
        return None
    return ('\n\n[训练时的回顾信息]\n' + '\n'.join(parts) +
            '\n这些是参考执行记录，不是当前页面。请结合原任务纠正失败原因、保留正确行为。'
            '当前状态以接下来对话的最后一次观察为准；只选择该页面合法的下一步动作，'
            '不要直接重演参考轨迹或复制当前页面没有的按钮。不要解释回顾信息，继续调用工具。'
            '\n[回顾信息结束]\n')


def prepare_group(group, tokenizer, settings, counts):
    """Keep informative individual trajectories, including equal-reward groups.

    Invalid siblings do not poison valid trajectories: SDPO has no group-return
    baseline. A demonstration must be another complete verified success.
    """
    config = settings['self_distillation']
    valid = [r for r in group if r['training_reward'] is not None and r['rollout_turns']]
    successes = [r for r in valid if _is_strict_success(r)]
    prepared = []
    for row in valid:
        demo = next((r for r in successes if r['trajectory_id'] != row['trajectory_id']), None)
        feedback = execution_feedback(row)
        text = feedback_text(feedback, demo)
        if text is None:
            counts['sdpo_no_feedback_trajectories'] += 1
            continue
        # Drop whole optional sections, never truncate a JSON value or a live
        # observation. An overlong feedback is explicitly masked and counted.
        if tokenizer.text_tokens(text) > config['feedback_tokens'] and demo is not None and feedback is not None:
            demo = None
            text = feedback_text(feedback, None)
            counts['sdpo_demo_omitted_for_length'] += 1
        if tokenizer.text_tokens(text) > config['feedback_tokens']:
            counts['sdpo_feedback_overflow_trajectories'] += 1
            continue
        turns = []
        for turn in row['rollout_turns']:
            messages = deepcopy(turn['messages'])
            # Initial user request, before the actual interleaved tool history.
            messages[1]['content'] += text
            prompt = tokenizer.apply_chat_template(normalize_messages_for_chat_template(messages),
                tools=SHOP_TOOL_SCHEMAS, tokenize=True, return_dict=False, add_generation_prompt=True)
            if len(prompt) + len(turn['response_ids']) > config['teacher_context_tokens']:
                counts['sdpo_context_overflow_turns'] += 1
                continue
            turns.append({**turn, 'teacher_prompt_ids': prompt,
                          'sdpo_strict_success': _is_strict_success(row)})
        if not turns:
            continue
        metadata = {'demonstration_id': demo['trajectory_id'] if demo else None,
                    'environment_feedback': feedback is not None, 'text': text,
                    'trained_turns': len(turns), 'strict_success': _is_strict_success(row)}
        row['distillation'] = metadata  # Saved with the original, unabridged rollout.
        prepared.append({**row, 'rollout_turns': turns, 'advantage': 0.0})
        counts['sdpo_demonstration_trajectories'] += demo is not None
        counts['sdpo_environment_feedback_trajectories'] += feedback is not None
        counts['sdpo_success_trajectories'] += _is_strict_success(row)
        counts['sdpo_failure_trajectories'] += not _is_strict_success(row)
        counts['sdpo_action_tokens'] += sum(len(t['response_ids']) for t in turns)
    return prepared


class EMATeacher:
    """EMA in trainable LoRA parameter space; frozen base is shared, not copied.

    Teacher forward must finish BEFORE the student autograd graph is created.
    This keeps gradient-checkpoint recomputation on the student's parameters.
    """
    def __init__(self, model, rate, state=None):
        self.parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.rate = rate
        if state is not None and self.parameters.keys() != state.keys():
            raise ValueError('EMA checkpoint parameters do not match the student adapter')
        self.shadow = {n: (state[n].to(p.device).float().clone() if state is not None
                           else p.detach().float().clone()) for n, p in self.parameters.items()}

    def update(self):
        import torch
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                self.shadow[name].lerp_(parameter.detach().float(), self.rate)

    def state_dict(self):
        return {n: p.detach().cpu().clone() for n, p in self.shadow.items()}

    @contextmanager
    def activate(self, model):
        import torch
        saved = {n: p.detach().clone() for n, p in self.parameters.items()}
        was_training = model.training
        try:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(self.shadow[name])
            model.eval()
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(saved[name])
            model.train(was_training)


def hindsight_logits(model, teacher, turn, device):
    import torch
    prompt, response = turn['teacher_prompt_ids'], turn['response_ids']
    ids = torch.tensor([prompt + response], device=device)
    positions = torch.arange(len(prompt)-1, len(prompt)+len(response)-1, device=device)
    with teacher.activate(model), torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids),
                     logits_to_keep=positions, use_cache=False).logits[0].detach()


def distillation_loss(student_logits, teacher_logits, targets, old_logprobs, config):
    """Exact full-vocabulary reverse KL, with detached capped rollout correction.

    Vocabulary reductions use FP32 and checkpointing in action-token chunks.
    No graph/gradient through the teacher, correction weight, or sampled tokens.
    Returns a token mean plus detached diagnostics; caller supplies global scale.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    def chunk_loss(student, teacher, target, old):
        logp, logq = student.float().log_softmax(-1), teacher.detach().float().log_softmax(-1)
        sampled = logp.gather(-1, target[:, None]).squeeze(-1)
        weight = (sampled.detach()-old).clamp(-20, 20).exp().clamp(max=config['importance_clip'])
        per_token = (logp.exp() * (logp-logq)).sum(-1)
        return (weight * per_token).sum()
    losses = []
    for start in range(0, len(targets), config['logit_chunk_tokens']):
        sl = slice(start, start+config['logit_chunk_tokens'])
        losses.append(checkpoint(chunk_loss, student_logits[sl], teacher_logits[sl],
                                 targets[sl], old_logprobs[sl], use_reentrant=False))
    loss = torch.stack(losses).sum() / len(targets)
    with torch.no_grad():
        def sampled_logprobs(logits):
            return torch.cat([-torch.nn.functional.cross_entropy(chunk.float(), target, reduction='none')
                for chunk, target in zip(logits.split(config['logit_chunk_tokens']),
                                         targets.split(config['logit_chunk_tokens']))])
        logp, logq = sampled_logprobs(student_logits), sampled_logprobs(teacher_logits)
        ratio = (logp-old_logprobs).clamp(-20, 20).exp()
        stats = {'importance_clipped_fraction': (ratio > config['importance_clip']).float().mean().item(),
                 'teacher_sampled_logprob_delta': (logq-logp).mean().item()}
    return loss, stats
