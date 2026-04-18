from __future__ import annotations

from collections import OrderedDict

import torch

from rsl_rl.modules import ActorCritic


class WarmStartActorCritic(ActorCritic):
    """ActorCritic with optional actor-only checkpoint initialization."""

    def __init__(self, *args, actor_init_path: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if actor_init_path:
            self.load_actor_only(actor_init_path)

    def load_actor_only(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        actor_state = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("actor."):
                actor_state[key[len("actor.") :]] = value

        if not actor_state:
            raise RuntimeError(f"No actor weights found in checkpoint: {checkpoint_path}")

        missing_keys, unexpected_keys = self.actor.load_state_dict(actor_state, strict=False)
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected actor keys while warm-starting from {checkpoint_path}: {unexpected_keys}"
            )

        print(
            f"[INFO] Warm-started actor from {checkpoint_path}. "
            f"Ignored missing actor keys: {list(missing_keys)}"
        )
