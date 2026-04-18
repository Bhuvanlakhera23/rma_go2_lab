"""Compatibility shim for removed legacy PPO adaptation code."""

from rsl_rl.algorithms import PPO


class PPOWithRMAAdaptation(PPO):
    """Placeholder kept only so stale eager imports do not fail."""


__all__ = ["PPOWithRMAAdaptation"]
