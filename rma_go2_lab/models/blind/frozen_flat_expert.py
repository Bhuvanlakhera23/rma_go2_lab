from __future__ import annotations

import re
from collections import OrderedDict

import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization


def _activation_from_name(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation for flat expert: {name}")


class FrozenFlatExpert(nn.Module):
    """Read-only actor rebuilt from a saved flat-prior checkpoint."""

    def __init__(self, checkpoint_path: str, activation: str = "elu", device: str = "cpu") -> None:
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        layer_ids = []
        for key in state_dict:
            match = re.fullmatch(r"actor\.(\d+)\.weight", key)
            if match:
                layer_ids.append(int(match.group(1)))
        if not layer_ids:
            raise RuntimeError(f"No actor weights found in flat checkpoint: {checkpoint_path}")
        layer_ids = sorted(layer_ids)

        modules = []
        for index, layer_id in enumerate(layer_ids):
            weight = state_dict[f"actor.{layer_id}.weight"]
            bias = state_dict[f"actor.{layer_id}.bias"]
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            modules.append((str(layer_id), linear))
            if index != len(layer_ids) - 1:
                modules.append((f"act_{layer_id}", _activation_from_name(activation)))

        actor_input_dim = state_dict[f"actor.{layer_ids[0]}.weight"].shape[1]
        normalizer_state = {
            key[len("actor_obs_normalizer.") :]: value
            for key, value in state_dict.items()
            if key.startswith("actor_obs_normalizer.")
        }
        if normalizer_state:
            self.actor_obs_normalizer = EmpiricalNormalization(actor_input_dim)
            self.actor_obs_normalizer.load_state_dict(normalizer_state, strict=True)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.actor = nn.Sequential(OrderedDict(modules))
        self.actor_obs_normalizer.eval()
        self.actor.eval()
        for param in self.actor_obs_normalizer.parameters():
            param.requires_grad_(False)
        for param in self.actor.parameters():
            param.requires_grad_(False)

    def forward(self, policy_obs: torch.Tensor) -> torch.Tensor:
        policy_obs = self.actor_obs_normalizer(policy_obs)
        return self.actor(policy_obs)
