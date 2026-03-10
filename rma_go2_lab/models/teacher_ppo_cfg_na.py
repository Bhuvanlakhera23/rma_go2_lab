from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class Go2RMATeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "go2_rma_teacher_na"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(

        value_loss_coef = 1.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,

        entropy_coef = 0.003,

        num_learning_epochs = 3,
        num_mini_batches = 4,

        learning_rate = 1e-4,

        schedule = "adaptive",
        desired_kl = 0.003,

        gamma = 0.99,
        lam = 0.95,
        max_grad_norm = 1.0,
    )