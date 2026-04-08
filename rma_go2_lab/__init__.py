import gymnasium as gym

from rma_go2_lab.models.ppo_with_flat_expert import PPOWithFlatExpert
from rma_go2_lab.models.rma_actor_critic import RMAActorCritic, RMAStudentActorCritic
from rma_go2_lab.models.ppo_with_rma_adaptation import PPOWithRMAAdaptation
import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner

# Inject custom classes so rsl_rl's eval() can find them
_rsl_on_policy_runner.PPOWithFlatExpert = PPOWithFlatExpert
_rsl_on_policy_runner.RMAActorCritic = RMAActorCritic
_rsl_on_policy_runner.RMAStudentActorCritic = RMAStudentActorCritic
_rsl_on_policy_runner.PPOWithRMAAdaptation = PPOWithRMAAdaptation

#Teacher Flat
gym.register(
    id="RMA-Go2-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.flat_expert_cfg:Go2RMATeacherEnvCfg",

        "rsl_rl_cfg_entry_point":
            "rma_go2_lab.models.flat_ppo_cfg:Go2RMAFlatPPORunnerCfg",
    },
)

#Teacher Rough V1
gym.register(
    id="RMA-Go2-Teacher-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.teacher_cfg_rough:Go2RMATeacherRoughEnvCfg",
        "rsl_rl_cfg_entry_point":
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)

#Teacher Rough Non-Adaptive Baseline
gym.register(
    id="RMA-Go2-Teacher-Rough-NA",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.teacher_cfg_rough_na:Go2RMATeacherRoughEnvCfg",
        "rsl_rl_cfg_entry_point":
            "rma_go2_lab.models.teacher_ppo_cfg_na:Go2RMATeacherPPORunnerCfg",
    },
)

# Student / Adaptation Phase (Stage F)
gym.register(
    id="RMA-Go2-Student-Rough-NA",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.student_cfg_rough_na:Go2RMAStudentRoughEnvCfg",
        "rsl_rl_cfg_entry_point":
            "rma_go2_lab.models.student_ppo_cfg_na:Go2RMAStudentPPORunnerCfg",
    },
)

# ------------------------------------------------
# Benchmarking Baselines
# ------------------------------------------------

# Baseline: Robust Domain Randomization (Blind PPO)
# Trained on the same Level 9 Rough/Stairs environment but without RMA heads.
gym.register(
    id="RMA-Go2-Baseline-Robust",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.teacher_cfg_rough_na:Go2RMATeacherRoughEnvCfg",
        "rsl_rl_cfg_entry_point":
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)
