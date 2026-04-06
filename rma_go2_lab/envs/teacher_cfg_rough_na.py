"""
rma_go2_lab/envs/teacher_cfg_rough_na.py
RMA True Teacher (V3 Stabilization)

Architecture:
    Actor:  Proprioception (48) + Latent Z (8) (Blind to terrain)
    Encoder: Privileged (252) -> Latent Z (8) (Processes terrain/dynamics)
    Critic:  Asymmetric (Policy + Privileged) (Full visibility)
"""

import torch
import torch.nn as nn
import numpy as np
from isaaclab.envs import mdp as base_mdp
import warp as wp
import isaaclab.utils.warp as wp_utils
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils.math import quat_apply, normalize

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)


def terrain_levels_vel_monotonic(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Promote terrains when the robot succeeds, but never collapse difficulty downward."""
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = torch.zeros_like(move_up, dtype=torch.bool)
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())

# ------------------------------------------------
# 🛠️ Fast RMA Utilities (Zero-Sensor Raycasting)
# ------------------------------------------------

def warp_height_scan(env, grid_size=(17, 11), grid_res=0.1):
    """
    Fast RMA-style terrain profile without USD sensors.
    Uses Warp to query the terrain mesh directly on the GPU.
    """
    # 1. Initialize Warp Mesh if not cached
    if not hasattr(env, "_warp_terrain_mesh"):
        # If the importer doesn't have the mesh directly, we re-generate it once (deterministic)
        if hasattr(env.scene.terrain, "terrain_mesh"):
            mesh = env.scene.terrain.terrain_mesh
        else:
            gen_cfg = env.scene.terrain.cfg.terrain_generator
            generator = gen_cfg.class_type(gen_cfg, device="cpu")
            mesh = generator.terrain_mesh

        env._warp_terrain_mesh = wp_utils.convert_to_warp_mesh(
            mesh.vertices, mesh.faces, device=env.device
        )

    # 2. Setup Ray Grid
    num_envs = env.num_envs
    num_points = grid_size[0] * grid_size[1]

    # Grid offsets relative to base
    x_range = torch.linspace(-(grid_size[0]-1)*grid_res/2, (grid_size[0]-1)*grid_res/2, grid_size[0], device=env.device)
    y_range = torch.linspace(-(grid_size[1]-1)*grid_res/2, (grid_size[1]-1)*grid_res/2, grid_size[1], device=env.device)
    xv, yv = torch.meshgrid(x_range, y_range, indexing='ij')
    offsets = torch.stack([xv, yv, torch.zeros_like(xv)], dim=-1).view(-1, 3) # [187, 3]

    # 3. Transform rays to World Frame (EGOCENTRIC FIX)
    robot = env.scene["robot"]

    # During manager initialization, robot data might not be ready yet.
    if robot.data.root_pos_w is None or robot.data.root_pos_w.shape[0] == 0:
        return torch.zeros((num_envs, num_points), device=env.device)

    base_pos = robot.data.root_pos_w # [N, 3]
    # NORMALIZE QUATERNION: Prevent NaNs from numerical drift in orientation
    base_quat = normalize(robot.data.root_quat_w) # [N, 4]

    # ROTATE the grid to be robot-aligned (Egocentric)
    # offsets: [187, 3], base_quat: [N, 4] -> output: [N, 187, 3]
    # We repeat/expand quat to match offsets for parallel rotation
    repeated_quat = base_quat.unsqueeze(1).repeat(1, num_points, 1).view(-1, 4)
    repeated_offsets = offsets.unsqueeze(0).repeat(num_envs, 1, 1).view(-1, 3)
    rotated_offsets = quat_apply(repeated_quat, repeated_offsets).view(num_envs, num_points, 3)

    points = base_pos.unsqueeze(1) + rotated_offsets # [N, 187, 3]

    # Raycast starts from 1.0m above
    origins = points.clone()
    origins[..., 2] += 1.0
    directions = torch.zeros_like(origins)
    directions[..., 2] = -1.0

    # 4. Warp Raycast (GPU Parallel)
    try:
        ray_hits = wp_utils.raycast_mesh(
            origins.view(-1, 3), directions.view(-1, 3), env._warp_terrain_mesh
        )[0]

        # 5. Extract Heights and normalize relative to base
        heights = ray_hits.view(num_envs, num_points, 3)[..., 2]
        # RMA baseline: height relative to base COM
        rel_heights = heights - base_pos[:, 2:3]

        # FINAL SAFETY: Shield neural network from NaNs (edge of terrain, etc.)
        return torch.nan_to_num(rel_heights, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception:
        # Fallback if warp fails or returns garbage
        return torch.zeros((num_envs, num_points), device=env.device)

def stand_still_foot_motion_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.15,
    velocity_threshold: float = 0.2,
):
    """Penalize foot motion when the robot should be essentially standing still."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_is_small = torch.linalg.norm(command[:, :2], dim=1) < command_threshold
    body_is_slow = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1) < velocity_threshold
    standstill = torch.logical_and(cmd_is_small, body_is_slow).unsqueeze(-1)
    foot_speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)
    return torch.sum(foot_speed * standstill, dim=1)


def foot_clearance_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    target_height: float = 0.1,
    std: float = 0.05,
    tanh_mult: float = 2.0,
):
    """Reward swinging feet for clearing a target height off the ground."""
    asset = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def get_dynamics(env):
    """Consolidated privileged physical parameters (17-dim)."""
    # 1. Friction (1-dim)
    if "physics_material" in env.event_manager.active_terms:
        friction = env.event_manager.get_term("physics_material").dynamic_friction.reshape(-1, 1)
    else:
        friction = torch.ones((env.num_envs, 1), device=env.device)

    # 2. Mass Offset (1-dim)
    if "add_base_mass" in env.event_manager.active_terms:
        mass = env.event_manager.get_term("add_base_mass").mass_offset.reshape(-1, 1)
    else:
        mass = torch.zeros((env.num_envs, 1), device=env.device)

    # 3. CoM Offset (3-dim)
    if "base_com" in env.event_manager.active_terms:
        com = env.event_manager.get_term("base_com").last_offset
    else:
        com = torch.zeros((env.num_envs, 3), device=env.device)

    # 4. Motor Strength/Gains (12-dim)
    # FIX: Read the actual current gains from the actor model (GPU tensors)
    # This precisely tracks the randomization applied by the EventManager.
    actuator = env.scene["robot"].actuators["base_legs"]
    motor_stiffness = actuator.stiffness / 25.0 # Normalize by default 25.0
    # motor_damping = actuator.damping / 0.5    # Optional: can add damping if needed

    return torch.cat([friction, mass, com, motor_stiffness], dim=-1)

# ------------------------------------------------


# ------------------------------------------------
# Environment
# ------------------------------------------------

@configclass
class Go2RMATeacherRoughEnvCfg(UnitreeGo2RoughEnvCfg):

    def __post_init__(self):

        super().__post_init__()

        print("\n========== RMA TEACHER \ ==========\n")

        # ------------------------------------------------
        # Robot
        # ------------------------------------------------

        self.scene.robot.spawn.usd_path = "/home/bhuvan/assets/go2/go2.usd"
        self.scene.num_envs = 4096 # Cranked back up! Warp sampler is stable.

        # ------------------------------------------------
        # 🔴 RMA Bottleneck: RAW TERRAIN SAMPLING (NOT SENSOR)
        # ------------------------------------------------
        # We REMOVE the HeightScanner USD sensor (it causes hangs)
        # Instead, we will sample the terrain mesh directly using Warp in the MDP.
        # self.scene.height_scanner = HeightScannerCfg(...) # REMOVED
        self.observations.policy.height_scan = None # Actor is blind to terrain

        # ------------------------------------------------
        # Command space
        # ------------------------------------------------

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 0.0

        # Start from the easiest rough terrains and only move upward when the policy earns it.
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels.func = terrain_levels_vel_monotonic

        # Stage D Refinement: 90% Stairs (Force ascent/descent mastery)
        terrain_gen = self.scene.terrain.terrain_generator
        terrain_gen.sub_terrains["pyramid_stairs"].proportion = 0.45
        terrain_gen.sub_terrains["pyramid_stairs_inv"].proportion = 0.45
        terrain_gen.sub_terrains["boxes"].proportion = 0.1
        # Disabled generic rough to maximize obstacle exposure
        terrain_gen.sub_terrains["random_rough"].proportion = 0.0
        terrain_gen.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        terrain_gen.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0
        terrain_gen.sub_terrains["pyramid_stairs"].step_height_range = (0.03, 0.18)
        terrain_gen.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.03, 0.18)
        terrain_gen.sub_terrains["boxes"].grid_height_range = (0.02, 0.14)
        terrain_gen.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
        terrain_gen.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------------------------
        # Domain randomization
        # ------------------------------------------------

        self.events.physics_material.params["static_friction_range"] = (0.1, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 2.0)

        if self.events.add_base_mass is not None:
            self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 4.0)

        self.events.motor_strength = EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_distribution_params": (0.6, 1.4),
                "damping_distribution_params": (0.6, 1.4),
                "operation": "scale",
            },
        )

        self.events.base_external_force_torque = None

        # ------------------------------------------------
        # Privileged observations (teacher-only hidden state)
        # ------------------------------------------------

        self.observations.privileged = ObsGroup(
            enable_corruption=False,
            concatenate_terms=True,
        )
        # 1. Privileged Dynamics (Secrets) - 17 dims
        self.observations.privileged.dynamics = ObsTerm(func=get_dynamics)

        # 2. Privileged Geometry (Expert Vision) - 187 dims
        self.observations.privileged.height_scan = ObsTerm(
            func=warp_height_scan,
            params={"grid_size": (17, 11), "grid_res": 0.1},
        )

        # ------------------------------------------------
        # Rewards
        # ------------------------------------------------

        # STABILITY REBALANCING: Allow pitching and z-velocity for stairs
        self.rewards.track_lin_vel_xy_exp.weight = 2.0  # Encourage momentum
        self.rewards.flat_orientation_l2.weight = -0.5  # Down from -2.5 (ALLOW PITCH ON STAIRS)

        self.rewards.action_rate_l2.weight = -0.003
        self.rewards.dof_torques_l2.weight = -2e-4
        self.rewards.dof_acc_l2.weight = -5e-7

        self.rewards.lin_vel_z_l2.weight = -0.1  # Down from -1.0 (ALLOW VERTICAL MOVEMENT)

        self.rewards.feet_air_time.weight = 2.0  # Stage D: Encourage higher steps for obstacles

        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
        # BUG FIX: disabled because foot_z_target_error uses absolute world Z, meaning it punishes lifting feet on stairs
        self.rewards.foot_clearance = None

        # Stronger heading stabilization for a truly straight forward teacher.
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # Recover the flat-seed posture shaping that kept the gait clean.
        self.rewards.stand_still_joint_deviation = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.35,
            params={"command_name": "base_velocity", "command_threshold": 0.15},
        )
        self.rewards.stand_still_foot_motion = RewTerm(
            func=stand_still_foot_motion_penalty,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "velocity_threshold": 0.2,
            },
        )
        self.rewards.hip_joint_deviation = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=-0.12,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
        )

        # Keep the teacher closer to the original RMA objective: height is handled by termination and terrain adaptation, not by explicit shaping.
        self.rewards.base_height = None
        self.rewards.dof_pos_limits.weight = -0.1
