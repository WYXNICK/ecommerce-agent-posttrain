"""One-round failure replay and suffix-only supervision, using the frozen actor."""
from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

NEXT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NEXT / 'src'))
sys.path.insert(0, str(NEXT / 'scripts/sft'))
from shopping_agent.clients import ModelReply, wire_messages
from shopping_agent.collection.sft import acceptance_reasons, build_sft_row
from shopping_agent.evaluation.summary import _is_strict_success
from shopping_agent.training.sft.dataset import build_supervised_example, normalize_messages_for_chat_template, _common_prefix_length


def ordered(rows, seed):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['task_id']}".encode()).hexdigest())


def repair_point(row):
    """Find an evidenced intervention point, not a claimed first causal error."""
    if _is_strict_success(row) or row.get('failure_kind') in {'infrastructure', 'unverifiable'}:
        return None
    messages = row['messages']
    indices = {c['id']: i for i,m in enumerate(messages) for c in m.get('tool_calls', [])}
    marks = [(indices[b['tool_call']['id']], 'guard_rejection') for b in row['blocked_tool_calls']
             if b['tool_call']['id'] in indices]
    seen = set()
    for step in row['steps']:
        i = indices.get(step['tool_call']['id'])
        if i is None:
            continue
        # Same visible state + same action is an evidenced revisit, not merely a repeated tool.
        state = next((m.get('content') for m in reversed(messages[:i]) if m['role'] in {'tool','user'}), '')
        key = (state, step['env_action'])
        if key in seen:
            marks.append((i, 'repeated_state_action'))
        seen.add(key)
    if marks:
        i, reason = min(marks)
    else:
        terminal = [s for s in row['steps'] if s['done'] and s['tool_name'] == 'buy_now']
        if not terminal:
            return None  # No reliable intervention point; do not invent a diagnosis.
        i, reason = indices[terminal[-1]['tool_call']['id']], 'before_failed_purchase'
    if not any(m['role']=='assistant' for m in messages[:i]):
        return {'message_index': i, 'reason': reason, 'prefix_actions': 0}
    return {'message_index': i, 'reason': reason,
            'prefix_actions': sum(m['role']=='assistant' for m in messages[:i])}


class ReplayClient:
    """Re-execute student actions and verify visible history before teacher takeover."""
    def __init__(self, source, point, teacher):
        self.source, self.point, self.teacher = source, point, teacher
        self.teacher_call_ids = []
        self.verified = False

    def complete(self, messages, tools, max_tokens):
        boundary = self.point['message_index']
        if not self.verified:
            if wire_messages(messages) != wire_messages(self.source['messages'][:len(messages)]):
                raise ValueError('replay_observation_mismatch')
            if len(messages) < boundary:
                message = self.source['messages'][len(messages)]
                if message['role'] != 'assistant':
                    raise ValueError('replay_boundary_not_assistant')
                return ModelReply(deepcopy(message), {'replayed': True})
            if len(messages) != boundary:
                raise ValueError('replay_passed_takeover_boundary')
            self.verified = True
        reply = self.teacher.complete(messages, tools, max_tokens)
        self.teacher_call_ids.extend(c['id'] for c in reply.message.get('tool_calls', []))
        return reply


def repaired_row(trajectory, replay):
    _, reasons = acceptance_reasons(trajectory)
    if reasons or not replay.verified:
        return None, reasons or ['prefix_not_verified']
    row = build_sft_row(trajectory)
    teacher_ids = set(replay.teacher_call_ids)
    starts = [i for i,m in enumerate(row['messages']) if any(c['id'] in teacher_ids for c in m.get('tool_calls', []))]
    if not starts:
        return None, ['no_teacher_actions']
    row.update(supervise_from_message=min(starts), source_trajectory_id=replay.source['trajectory_id'],
               intervention=replay.point, replay_verified=True)
    return row, []


def example_for(row, tokenizer, max_length):
    example = build_supervised_example(row['messages'], row['tools'], tokenizer, max_length=max_length)
    if example is None:
        return None
    start = row.get('supervise_from_message', 0)
    if start:
        messages = normalize_messages_for_chat_template(row['messages'])
        if messages[start]['role'] != 'assistant':
            raise ValueError('supervision boundary must be an assistant action')
        before = tokenizer.apply_chat_template(messages[:start], tools=row['tools'], tokenize=False, add_generation_prompt=True)
        through = tokenizer.apply_chat_template(messages[:start+1], tools=row['tools'], tokenize=False, add_generation_prompt=False)
        left = tokenizer(before, add_special_tokens=False)['input_ids']
        right = tokenizer(through, add_special_tokens=False)['input_ids']
        boundary = _common_prefix_length(left, right)
        if example['input_ids'][:len(right)] != list(right):
            raise ValueError('teacher suffix template boundary changed')
        example['labels'][:boundary] = [-100] * boundary
    return example if any(x != -100 for x in example['labels']) else None


def annotate(row, tokenizer, max_length):
    example = example_for(row, tokenizer, max_length)
    if example is None:
        return None
    return {**row, 'sequence_tokens': len(example['input_ids']),
            'supervised_tokens': sum(x != -100 for x in example['labels'])}
