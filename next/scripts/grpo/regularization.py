"""Frozen SFT reference and action-position diagnostics; no teacher API."""
import torch
import torch.nn.functional as F
from verl.trainer.ppo.core_algos import kl_penalty


@torch.no_grad()
def reference_logprobs(model, ids, positions, targets):
    """Merged S-repair is frozen beneath the trainable, bias-free RL LoRA.

    Run before the actor forward, so checkpoint recomputation always sees the
    enabled actor adapter. Only sampled-token log probabilities survive here.
    """
    was_training = model.training
    try:
        model.eval()
        with model.disable_adapter():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           logits_to_keep=positions, use_cache=False).logits[0]
            return torch.cat([-F.cross_entropy(chunk.float(), target, reduction='none')
                              for chunk, target in zip(logits.split(64), targets.split(64))])
    finally:
        model.train(was_training)


def reference_kl(logprob, ref_logprob):
    # veRL k3+ reports k3 but uses k2's score gradient. With our single on-policy
    # update this estimates the forward KL gradient, unlike differentiating k3
    # on fixed sampled tokens. The reference never receives gradients.
    return kl_penalty(logprob, ref_logprob.detach(), 'k3+')


@torch.no_grad()
def action_entropy(logits):
    """True categorical entropy, summed over generated positions, in nats.

    Chunk only the diagnostic vocabulary reduction, avoiding a second full FP32
    logits buffer. This is not sampled NLL and is not added to the objective.
    """
    total = logits.new_zeros((), dtype=torch.float32)
    for chunk in logits.detach().split(64):
        logp = chunk.float().log_softmax(-1)
        total += -(logp.exp() * logp).sum()
    return total
