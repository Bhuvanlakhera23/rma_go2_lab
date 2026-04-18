import gymnasium as gym

from rma_go2_lab.models.blind.actor_critic import WarmStartActorCritic
from rma_go2_lab.models.blind.ppo_with_flat_expert import BlindPPOWithFlatExpert
import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner

# Inject custom classes so rsl_rl's eval() can find them
_rsl_on_policy_runner.WarmStartActorCritic = WarmStartActorCritic
_rsl_on_policy_runner.BlindPPOWithFlatExpert = BlindPPOWithFlatExpert

#Teacher Flat
gym.register(
    id="RMA-Go2-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.priors.flat_cfg:Go2FlatPriorEnvCfg",

        "rsl_rl_cfg_entry_point":
            "rma_go2_lab.models.priors.flat_ppo_cfg:Go2FlatPriorPPORunnerCfg",
    },
)

# Active pipeline only:
# flat prior plus the three blind baseline variants.
gym.register(
    id="RMA-Go2-Blind-Baseline-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.blind.rough_cfg:Go2BlindBaselineRoughEnvCfg",
        "rsl_rl_cfg_entry_point":
            "rma_go2_lab.models.blind.variants_ppo_cfg:Go2BlindBaselineScratchPPORunnerCfg",
    },
)

gym.register(
    id="RMA-Go2-Blind-Baseline-Rough-WarmStart",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.blind.rough_cfg:Go2BlindBaselineRoughEnvCfg",
        "rsl_rl_cfg_entry_point":
            "rma_go2_lab.models.blind.variants_ppo_cfg:Go2BlindBaselineWarmStartPPORunnerCfg",
    },
)

gym.register(
    id="RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.blind.rough_cfg:Go2BlindBaselineRoughEnvCfg",
        "rsl_rl_cfg_entry_point":
            "rma_go2_lab.models.blind.variants_ppo_cfg:Go2BlindBaselineWarmStartImitationPPORunnerCfg",
    },
)
