"""
Final Robust DR Baseline (Non-Adaptive)

Environment:
heavy domain randomization
rough terrain curriculum

Actor:
proprio + height scan

Critic:
proprio + height scan + privileged env parameters

Algorithm:
PPO
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from isaaclab.sensors import RayCaster
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

# ----------------------
# Base Height
# ----------------------        

def base_height_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
):

    asset: RigidObject = env.scene[asset_cfg.name]

    base_height = asset.data.root_pos_w[:,2]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        terrain_height = torch.mean(sensor.data.ray_hits_w[...,2], dim=1)

        height_error = (base_height - terrain_height) - target_height
    else:
        height_error = base_height - target_height

    return torch.square(height_error)

@configclass
class Go2RMATeacherRoughEnvCfg(UnitreeGo2RoughEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        # =========================================================
        # Core Setup
        # =========================================================
        self.scene.robot.spawn.usd_path = "/home/bhuvan/assets/go2/go2.usd"
        self.sim.physx.gpu_max_rigid_patch_count = 600000
        self.scene.num_envs = 4096

        # =========================================================
        # Command Space
        # =========================================================
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_standing_envs = 0.05

        # =========================================================
        # Domain Randomization
        # =========================================================

        self.events.physics_material.params["static_friction_range"] = (0.25, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.1)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)

        if self.events.add_base_mass is not None:
            self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)

        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.mode = "interval"
            self.events.base_external_force_torque.interval_range_s = (5.0, 10.0)
            self.events.base_external_force_torque.params["force_range"] = (-30.0, 30.0)

        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com, 
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {
                    "x": (-0.02, 0.02),
                    "y": (-0.02, 0.02),
                    "z": (-0.01, 0.01),
                },
            },
        )

        # =========================================================
        # Terrain Difficulty
        # =========================================================
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.04, 0.15)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.02, 0.08)

        # =========================================================
        # Reward
        # =========================================================

        # ----------------------
        # Tracking
        # ----------------------
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # ----------------------
        # Stability
        # ----------------------
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -2.0

        # ----------------------
        # Base Height Anchoring (CRITICAL)
        # ----------------------
        self.rewards.base_height = RewTerm(
            func=base_height_l2,
            weight=-1.0,  # start conservative
            params={
                "target_height": 0.32,  # <-- SET THIS to your nominal root height
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),  # must match your raycaster name
            },
        )

        # ----------------------
        # Energy + Smoothness
        # ----------------------
        self.rewards.dof_torques_l2.weight = -1.5e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.0075

        # ----------------------
        # Stepping
        # ----------------------
        self.rewards.feet_air_time.weight = 2.5

        # ----------------------
        # Slip penalty
        # ----------------------
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )

        # ----------------------
        # Undesired contacts
        # ----------------------
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
                "threshold": 1.0,
            },
        )

        self.rewards.dof_pos_limits.weight = 0.0

        # ---------------------------------------------------------
        # Privileged observations (critic only)
        # ---------------------------------------------------------

        self.observations.privileged = ObsGroup()

        self.observations.privileged.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        self.observations.privileged.base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        self.observations.privileged.projected_gravity = ObsTerm(func=mdp.projected_gravity)

        self.observations.privileged.joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.observations.privileged.joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )