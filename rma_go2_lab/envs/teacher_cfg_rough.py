"""
V1 - Non Adaptive Locomotion Policy
This is a heavily domain-randomised, rough-terrain robust locomotion policy.
This is not an adaptive policy.
No privileged observations as input.
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

from rma_go2_lab.envs.teacher_rewards import base_height_l2


@configclass
class Go2RMATeacherRoughEnvCfg(UnitreeGo2RoughEnvCfg):

    def __post_init__(self):

        super().__post_init__()

        # =========================================================
        # Robot Asset
        # =========================================================
        self.scene.robot.spawn.usd_path = "/home/bhuvan/assets/go2/go2.usd"

        # Increase GPU rigid patch count for large parallel runs
        self.sim.physx.gpu_max_rigid_patch_count = 600000

        # =========================================================
        # Parallelization (use your A6000 properly)
        # =========================================================
        self.scene.num_envs = 8192

        # =========================================================
        # Command Space (full locomotion capability)
        # =========================================================
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_standing_envs = 0.05

        # =========================================================
        # Domain Randomization (moderate, realistic)
        # =========================================================

        # Friction variation
        self.events.physics_material.static_friction_range = (0.25, 1.2)
        self.events.physics_material.dynamic_friction_range = (0.2, 1.1)

        # Mass variation
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 3.0)

        # External pushes (not too aggressive early)
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["force_range"] = (-30.0, 30.0)

        # =========================================================
        # Reward Shaping (balanced, terrain-aware)
        # =========================================================

        # Velocity tracking (keep strong)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # Stability penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # Energy + smoothness
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01

        # Encourage real stepping (important on rough)
        self.rewards.feet_air_time.weight = 0.2

        # Penalize slipping
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.2,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )

        # Orientation penalty (moderate, not stiff)
        self.rewards.flat_orientation_l2.weight = -2.0

        # Keep joint limit penalty zero
        self.rewards.dof_pos_limits.weight = 0.0
        