# RMA-Go2-Teacher-Rough-NA

Generated: 2026-04-01T16:58:39

## Observation Shapes
- policy: [1, 48]
  - base_lin_vel: base_lin_vel
  - base_ang_vel: base_ang_vel
  - projected_gravity: projected_gravity
  - velocity_commands: generated_commands
  - joint_pos: joint_pos_rel
  - joint_vel: joint_vel_rel
  - actions: last_action
- privileged: [1, 204]
  - dynamics: get_dynamics
  - height_scan: warp_height_scan

## Networks
- actor: 56 -> [512, 256, 128] -> 12
- critic: 252 -> [512, 256, 128] -> 1

## Commands
- resampling_time_range_s: [10.0, 10.0]
- heading_command: False
- heading_control_stiffness: 0.5
- rel_standing_envs: 0.02
- rel_heading_envs: 0.0
- lin_vel_x: [0.0, 1.0]
- lin_vel_y: [0.0, 0.0]
- ang_vel_z: [0.0, 0.0]
- heading: [0.0, 0.0]

## Randomization
- physics_material (startup): {'asset_cfg': {'name': 'robot', 'joint_names': None, 'joint_ids': 'slice(None, None, None)', 'fixed_tendon_names': None, 'fixed_tendon_ids': 'slice(None, None, None)', 'body_names': '.*', 'body_ids': 'slice(None, None, None)', 'object_collection_names': None, 'object_collection_ids': 'slice(None, None, None)', 'preserve_order': False}, 'static_friction_range': [0.1, 2.0], 'dynamic_friction_range': [0.1, 2.0], 'restitution_range': [0.0, 0.0], 'num_buckets': 64}
- add_base_mass (startup): {'asset_cfg': {'name': 'robot', 'joint_names': None, 'joint_ids': 'slice(None, None, None)', 'fixed_tendon_names': None, 'fixed_tendon_ids': 'slice(None, None, None)', 'body_names': 'base', 'body_ids': 'slice(None, None, None)', 'object_collection_names': None, 'object_collection_ids': 'slice(None, None, None)', 'preserve_order': False}, 'mass_distribution_params': [-2.0, 4.0], 'operation': 'add'}
- reset_base (reset): {'pose_range': {'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'yaw': [-3.14, 3.14]}, 'velocity_range': {'x': [0.0, 0.0], 'y': [0.0, 0.0], 'z': [0.0, 0.0], 'roll': [0.0, 0.0], 'pitch': [0.0, 0.0], 'yaw': [0.0, 0.0]}}
- reset_robot_joints (reset): {'position_range': [1.0, 1.0], 'velocity_range': [0.0, 0.0]}
- motor_strength (startup): {'asset_cfg': {'name': 'robot', 'joint_names': None, 'joint_ids': 'slice(None, None, None)', 'fixed_tendon_names': None, 'fixed_tendon_ids': 'slice(None, None, None)', 'body_names': None, 'body_ids': 'slice(None, None, None)', 'object_collection_names': None, 'object_collection_ids': 'slice(None, None, None)', 'preserve_order': False}, 'stiffness_distribution_params': [0.6, 1.4], 'damping_distribution_params': [0.6, 1.4], 'operation': 'scale'}

## Rewards
- track_lin_vel_xy_exp: weight=1.5 func=track_lin_vel_xy_exp
- track_ang_vel_z_exp: weight=0.2 func=track_ang_vel_z_exp
- lin_vel_z_l2: weight=-1.0 func=lin_vel_z_l2
- ang_vel_xy_l2: weight=-0.05 func=ang_vel_xy_l2
- dof_torques_l2: weight=-0.0002 func=joint_torques_l2
- dof_acc_l2: weight=-5e-07 func=joint_acc_l2
- action_rate_l2: weight=-0.003 func=action_rate_l2
- feet_air_time: weight=0.4 func=feet_air_time
- flat_orientation_l2: weight=-2.5 func=flat_orientation_l2
- dof_pos_limits: weight=-0.1 func=joint_pos_limits
- feet_slide: weight=-0.1 func=feet_slide
