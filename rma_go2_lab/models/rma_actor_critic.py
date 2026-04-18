"""Minimal compatibility shims for stale RMA actor-critic imports.

The active repo no longer carries the old teacher/student architecture, but the
local IsaacLab / rsl_rl install still imports these symbols eagerly at module
import time. Keep small placeholder classes here so active baseline tasks can
launch without restoring the removed legacy tree.
"""

from rsl_rl.modules import ActorCritic


class RMAActorCritic(ActorCritic):
    """Compatibility placeholder for removed legacy RMA actor-critic."""


class RMAStudentActorCritic(ActorCritic):
    """Compatibility placeholder for removed legacy student actor-critic."""


__all__ = ["RMAActorCritic", "RMAStudentActorCritic"]
