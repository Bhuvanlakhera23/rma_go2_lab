import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_l2(
    env: "ManagerBasedRLEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from desired base height using L2 loss."""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return torch.square(base_height - target_height)
