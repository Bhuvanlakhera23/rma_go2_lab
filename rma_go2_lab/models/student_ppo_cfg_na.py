# rma_go2_lab/models/student_ppo_cfg_na.py
from isaaclab.utils import configclass
from rma_go2_lab.models.rma_actor_critic import RMAStudentActorCritic
from rma_go2_lab.models.ppo_with_rma_adaptation import PPOWithRMAAdaptation
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class Go2RMAStudentPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env = 32
    max_iterations = 1000 # Student usually converges faster than teacher
    save_interval = 50

    experiment_name = "go2_rma_student_na"

    obs_groups = {
        "policy": ["policy", "history"],
        "critic": ["policy", "privileged"], # Critic still sees privileged info during training
    }

    policy = dict(
        class_name="RMAStudentActorCritic",
        init_noise_std=0.2, # Start with low noise as we bootstrap from a qualified expert
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # TEACHER SUPERVISOR: Refined Rough Terrain Expert (Iteration 240)
        pretrained_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/rough_teacher_v2_refined_240.pt",
    )

    algorithm = dict(
        class_name="PPOWithRMAAdaptation",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001, # Low entropy (focus on imitation)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-5,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        adaptation_loss_coef=15.0, # 🔴 SUPERVISED DISTILLATION WEIGHT
    )
