"""Compatibility aliases for stale flat-expert PPO imports."""

from rma_go2_lab.models.blind.frozen_flat_expert import FrozenFlatExpert
from rma_go2_lab.models.blind.ppo_with_flat_expert import BlindPPOWithFlatExpert

PPOWithFlatExpert = BlindPPOWithFlatExpert

__all__ = ["FrozenFlatExpert", "PPOWithFlatExpert"]
