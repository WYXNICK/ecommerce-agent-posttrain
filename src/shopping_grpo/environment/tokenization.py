"""Token boundaries shared by Shopping's veRL and local evaluation executors."""

import hashlib
import json


def token_fingerprint(token_ids):
    return hashlib.sha256(json.dumps(list(token_ids), separators=(",", ":")).encode()).hexdigest()


def tool_response_tokens(tokenizer, messages, *, template_kwargs):
    """Render only the new tool feedback, preserving the complete template boundary.

    Qwen3.5 needs a user message to render tool-only input. veRL's helper removes
    that dummy message; its system-prefix removal must also be mirrored exactly.
    Tool chunks must not be left-truncated by the initial-prompt length limit.
    """
    from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
    from verl.utils.tokenizer import normalize_token_ids

    tokens = normalize_token_ids(apply_chat_template(
        tokenizer, messages, add_generation_prompt=True, tokenize=True,
        **template_kwargs,
    ))
    system = initialize_system_prompt(tokenizer, **template_kwargs)
    return tokens[len(system):]
