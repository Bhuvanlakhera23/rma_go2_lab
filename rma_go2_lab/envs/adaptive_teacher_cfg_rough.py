"""
RMA Teacher – Rough Terrain
Heavily domain randomized locomotion teacher.
Stores true extrinsics and exposes them as privileged observations.
"""

import torch
from isaaclab.utils import configclass

import rsl_rl.runners.on_policy_runner as opr
from rma_go2_lab.models.actorcritic import RMAActorCritic
opr.RMAActorCritic = RMAActorCritic

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

# =========================================================
# -----------  PRIVILEGED OBSERVATION TERM ----------------
# =========================================================

def get_extrinsics(env):
    if not hasattr(env, "extrinsics"):
        device = env.device
        num_envs = env.scene.num_envs
        env.extrinsics = torch.zeros(num_envs, 5, device=device)
    return env.extrinsics


@configclass
class PrivilegedObsCfg(ObsGroup):
    extrinsics = ObsTerm(func=get_extrinsics)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


# =========================================================
# -----------  RMA RANDOMIZATION (RESET SAFE) -------------
# =========================================================

def rma_randomization(env, env_ids):

    device = env.device
    num_envs = env.scene.num_envs

    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device)

    robot = env.scene["robot"]
    base_id = robot.data.body_names.index("base")

    # ------------------------------------------------------
    # Initialize extrinsics buffer once
    # ------------------------------------------------------
    if not hasattr(env, "extrinsics"):
        env.extrinsics = torch.zeros(num_envs, 5, device=device)

    # ------------------------------------------------------
    # Cache original physical parameters (once)
    # ------------------------------------------------------
    if not hasattr(robot, "_original_masses"):
        robot._original_masses = robot.root_physx_view.get_masses().clone()
        robot._original_inertias = robot.root_physx_view.get_inertias().clone()
        robot._original_coms = robot.root_physx_view.get_coms().clone()

    # ------------------------------------------------------
    # Sample extrinsics
    # ------------------------------------------------------
    friction = torch.rand(len(env_ids), device=device) * 0.6 + 0.6
    base_mass_scale = torch.rand(len(env_ids), device=device) * 0.4 + 0.8
    payload_offset = torch.rand(len(env_ids), 3, device=device) * 0.04 - 0.02

    env.extrinsics[env_ids, 0] = friction
    env.extrinsics[env_ids, 1] = base_mass_scale
    env.extrinsics[env_ids, 2:5] = payload_offset

    # ------------------------------------------------------
    # Friction (feet only)
    # ------------------------------------------------------
    materials = robot.root_physx_view.get_material_properties()

    env_ids_cpu = env_ids.to("cpu")
    friction_cpu = friction.to("cpu")

    materials[env_ids_cpu, :, 0] = friction_cpu.unsqueeze(1)
    materials[env_ids_cpu, :, 1] = friction_cpu.unsqueeze(1)

    robot.root_physx_view.set_material_properties(
        materials,
        env_ids_cpu,
    )

    # ------------------------------------------------------
    # Mass + inertia (NO compounding)
    # ------------------------------------------------------
    masses = robot._original_masses.clone()
    inertias = robot._original_inertias.clone()

    env_ids_cpu = env_ids.to("cpu")
    base_mass_scale_cpu = base_mass_scale.to("cpu")

    masses[env_ids_cpu, base_id] *= base_mass_scale_cpu
    inertias[env_ids_cpu, base_id] *= base_mass_scale_cpu.unsqueeze(-1)

    robot.root_physx_view.set_masses(masses, env_ids_cpu)
    robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

    # ------------------------------------------------------
    # COM shift (relative to original COM)
    # ------------------------------------------------------
    coms = robot._original_coms.clone()

    payload_offset_cpu = payload_offset.to("cpu")

    base_mass = robot._original_masses[env_ids_cpu, base_id]
    payload_mass = base_mass * 0.1

    # --- extract original COM position (only xyz)
    original_com_pos = coms[env_ids_cpu, base_id, :3]

    # --- compute new COM position
    new_com_pos = (
        base_mass.unsqueeze(-1) * original_com_pos
        + payload_mass.unsqueeze(-1) * payload_offset_cpu
    ) / (base_mass + payload_mass).unsqueeze(-1)

    # --- write back ONLY position
    coms[env_ids_cpu, base_id, :3] = new_com_pos

    robot.root_physx_view.set_coms(coms, env_ids_cpu)

# =========================================================
# -----------  TEACHER ENV CONFIG -------------------------
# =========================================================

@configclass
class Go2RMATeacherRoughEnvCfg(UnitreeGo2RoughEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        # Attach privileged observation group
        self.observations.privileged = PrivilegedObsCfg()

        # Physics tuning
        self.sim.physx.gpu_max_rigid_patch_count = 600000

        # Disable default randomizers
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None

        # Register RMA randomization (RESET MODE)
        self.events.rma_randomization = EventTerm(
            func=rma_randomization,
            mode="reset",
        )

        # Reward shaping
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.2
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.dof_pos_limits.weight = 0.0


# =========================================================
# -----------  PPO RUNNER CONFIG --------------------------
# =========================================================

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
    experiment_name = "go2_rma_teacher_adaptive"

    # Actor sees only policy obs
    # Critic sees policy + privileged
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="RMAActorCritic",
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )