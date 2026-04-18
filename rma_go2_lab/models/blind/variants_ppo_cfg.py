"""PPO variants for the blind rough-terrain baseline."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


FLAT_EXPERT_CKPT = "/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/flat1499.pt"


@configclass
class BlindWarmStartPolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "WarmStartActorCritic"
    actor_init_path: str | None = None


@configclass
class BlindImitationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "BlindPPOWithFlatExpert"
    flat_expert_path: str | None = None
    flat_expert_activation: str = "elu"
    flat_imitation_command_threshold: float = 0.1
    flat_imitation_coef_stage0: float = 0.3
    flat_imitation_coef_stage1: float = 0.1
    flat_imitation_stage0_end: int = 300
    flat_imitation_stage1_end: int = 800


@configclass
class Go2BlindBaselineScratchPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_blind_baseline_rough_scratch"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
    )


@configclass
class Go2BlindBaselineWarmStartPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_blind_baseline_rough_warmstart"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy = BlindWarmStartPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=FLAT_EXPERT_CKPT,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
    )


@configclass
class Go2BlindBaselineWarmStartImitationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_blind_baseline_rough_warmstart_imitation"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy = BlindWarmStartPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=FLAT_EXPERT_CKPT,
    )

    algorithm = BlindImitationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        flat_expert_path=FLAT_EXPERT_CKPT,
        flat_expert_activation="elu",
        flat_imitation_command_threshold=0.1,
        flat_imitation_coef_stage0=0.3,
        flat_imitation_coef_stage1=0.1,
        flat_imitation_stage0_end=300,
        flat_imitation_stage1_end=800,
    )
