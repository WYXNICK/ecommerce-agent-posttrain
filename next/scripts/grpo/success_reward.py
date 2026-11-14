"""Strict success dominates bounded failure quality in mixed-success GRPO groups."""
from shopping_agent.evaluation.summary import _is_strict_success


def success_return(row, constrained_reward, group_size, failure_floor):
    if constrained_reward is None:
        return None
    if not failure_floor <= constrained_reward <= 1:
        raise ValueError('constraint return outside the frozen reward range')
    if _is_strict_success(row):
        return 1.0
    # Preserve failure ordering without allowing a failed trajectory to outrank
    # the group mean when at least one of G trajectories strictly succeeds.
    return (constrained_reward - 1) / (1 - failure_floor) / group_size


def success_group_filter(group):
    successes = sum(_is_strict_success(row) for row in group)
    if successes == 0:
        return 'no_success'
    if successes == len(group):
        return 'all_success'
    return None
