#rma_go2_lab/models/teacher_ppo_cfg_na.py
from isaaclab.utils import configclass
from rma_go2_lab.models.rma_actor_critic import RMAActorCritic
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class RMAPolicyCfg(RslRlPpoActorCriticCfg):
    pretrained_path: str = None
    num_actor_obs: int = None
    num_critic_obs: int = None

@configclass
class Go2RMATeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    num_envs = None # Allow CLI override (e.g. num_envs=1)
    num_steps_per_env = 32
    max_iterations = 2000  # fresh teacher run budget
    save_interval = 20

    experiment_name = "go2_rma_teacher_na"

    obs_groups = {
        "policy": ["policy"],  # Actor sees deployable policy observations only.
        "critic": ["policy", "privileged"],  # Asymmetric critic keeps privileged context.
    }

    policy = RMAPolicyCfg(
        class_name="RMAActorCritic",
        init_noise_std=0.5, # 🟢 Lower noise to preserve bootstrap stability
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_actor_obs=56,
        num_critic_obs=252,
        pretrained_path="/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_teacher_na/2026-04-02_18-41-09/model_1999.pt",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-5, # even gentler LR for late-stage refinement
        schedule="adaptive",
        desired_kl=0.02, # loosened KL to allow adaptation to steep obstacles
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
    )