import gymnasium as gym

#Teacher Flat
gym.register(
    id="RMA-Go2-Teacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.teacher_cfg_flat:Go2RMATeacherEnvCfg",
        "rsl_rl_cfg_entry_point":
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
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

#Teacher Rough Adaptive
gym.register(
    id="RMA-Go2-Teacher-Rough-Adaptive",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":
            "rma_go2_lab.envs.adaptive_teacher_cfg_rough:Go2RMATeacherRoughEnvCfg",
        "rsl_rl_cfg_entry_point":
            "rma_go2_lab.envs.adaptive_teacher_cfg_rough:Go2RMATeacherPPORunnerCfg",
    },
)

