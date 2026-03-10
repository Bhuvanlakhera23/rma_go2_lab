from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from isaaclab.managers import RewardTermCfg as RewTerm
from rma_go2_lab.envs.teacher_rewards import base_height_l2
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class Go2RMATeacherEnvCfg(UnitreeGo2FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 12288

        # -----------------------
        # Command Restriction
        # -----------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.15


        # -----------------------
        # Domain Randomization
        # -----------------------

        # Friction variation
        self.events.physics_material.static_friction_range = (0.8, 1.2)
        self.events.physics_material.dynamic_friction_range = (0.8, 1.2)

        # Mass distribution (keep structure, widen range carefully)
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 4.0)

        # External push disturbance
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["force_range"] = (-50.0, 50.0)

        self.actions.joint_pos.scale = 0.25

        self.episode_length_s = 20.0

        self.rewards.base_height_l2 = RewTerm(
            func=base_height_l2,
            weight=-2.0,
            params={"target_height": 0.38},
        )

        self.rewards.track_ang_vel_z_exp.weight = 1.0

        self.rewards.dof_torques_l2.weight = -0.0002

        self.rewards.feet_air_time.weight = 0.0

        self.rewards.flat_orientation_l2.weight = -3.5

        # re-add undesired contacts (since Rough deleted it)
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names="^(?!.*_foot).*"
                ),
                "threshold": 1.0,
            },
        )

        # add slip penalty
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )

        self.rewards.stand_still_joint_deviation = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.5,
            params={"command_name": "base_velocity"}
        )