# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""Export a structured manifest for the current RMA teacher configuration."""

import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export an RMA teacher manifest.")
parser.add_argument("--task", type=str, default="RMA-Go2-Teacher-Rough-NA", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent entry point name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to instantiate for shape introspection.")
parser.add_argument("--seed", type=int, default=42, help="Seed for config instantiation.")
parser.add_argument("--output-prefix", type=str, default=None, help="Output prefix path without extension.")
parser.add_argument("--run-dir", type=str, default=None, help="Optional IsaacLab run directory; used to derive a manifest filename automatically.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import yaml
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, EventTermCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def tensor_stats(tensor):
    return {
        'shape': list(tensor.shape),
        'min': float(tensor.min().item()),
        'max': float(tensor.max().item()),
        'mean': float(tensor.mean().item()),
        'std': float(tensor.std().item()) if tensor.numel() > 1 else 0.0,
    }


def runtime_sample_manifest(env, obs):
    sample = {'env_index': 0}
    command = env.unwrapped.command_manager.get_command('base_velocity')[0]
    sample['command'] = {
        'lin_vel_x': float(command[0].item()),
        'lin_vel_y': float(command[1].item()),
        'ang_vel_z': float(command[2].item()),
    }

    if 'physics_material' in env.unwrapped.event_manager.active_terms:
        term = env.unwrapped.event_manager.get_term('physics_material')
        sample['dynamic_friction'] = float(term.dynamic_friction[0].item())
        sample['static_friction'] = float(term.static_friction[0].item()) if hasattr(term, 'static_friction') else None
    if 'add_base_mass' in env.unwrapped.event_manager.active_terms:
        term = env.unwrapped.event_manager.get_term('add_base_mass')
        sample['base_mass_offset'] = float(term.mass_offset[0].item())
    if 'base_com' in env.unwrapped.event_manager.active_terms:
        term = env.unwrapped.event_manager.get_term('base_com')
        sample['base_com_offset'] = [float(v) for v in term.last_offset[0].tolist()]
    else:
        sample['base_com_offset'] = [0.0, 0.0, 0.0]

    actuator = env.unwrapped.scene['robot'].actuators['base_legs']
    sample['motor_stiffness'] = [float(v) for v in actuator.stiffness[0].tolist()]
    sample['motor_stiffness_normalized'] = [float(v / 25.0) for v in actuator.stiffness[0].tolist()]
    sample['motor_damping'] = [float(v) for v in actuator.damping[0].tolist()]

    sample['policy_obs_stats'] = tensor_stats(obs['policy'][0])
    if 'privileged' in obs:
        sample['privileged_obs_stats'] = tensor_stats(obs['privileged'][0])
        sample['height_scan_stats'] = tensor_stats(obs['privileged'][0, -187:])
        sample['dynamics_sample'] = [float(v) for v in obs['privileged'][0, :17].tolist()]

    return sample


def to_builtin(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if hasattr(value, "__dict__") and not isinstance(value, type):
        raw = {}
        for k, v in vars(value).items():
            if k.startswith("_"):
                continue
            raw[k] = to_builtin(v)
        return raw
    if hasattr(value, "__name__"):
        return value.__name__
    return repr(value)


def iter_cfg_terms(cfg_obj, expected_type=None):
    for name, value in vars(cfg_obj).items():
        if name.startswith("_") or value is None:
            continue
        if expected_type is None or isinstance(value, expected_type):
            yield name, value


def obs_group_manifest(group_cfg, obs_sample):
    terms = []
    for name, term in iter_cfg_terms(group_cfg, ObservationTermCfg):
        terms.append({
            "name": name,
            "func": getattr(term.func, "__name__", repr(term.func)),
            "params": to_builtin(term.params),
        })
    return {
        "shape": list(obs_sample.shape),
        "terms": terms,
    }


def rewards_manifest(reward_cfg):
    items = []
    for name, term in iter_cfg_terms(reward_cfg, RewardTermCfg):
        items.append({
            "name": name,
            "weight": term.weight,
            "func": getattr(term.func, "__name__", repr(term.func)),
            "params": to_builtin(term.params),
        })
    return items


def events_manifest(event_cfg):
    items = []
    for name, term in iter_cfg_terms(event_cfg, EventTermCfg):
        items.append({
            "name": name,
            "mode": term.mode,
            "func": getattr(term.func, "__name__", repr(term.func)),
            "params": to_builtin(term.params),
        })
    return items


def curriculum_manifest(curriculum_cfg):
    items = []
    for name, term in vars(curriculum_cfg).items():
        if name.startswith("_") or term is None:
            continue
        items.append({
            "name": name,
            "func": getattr(term.func, "__name__", repr(term.func)),
            "params": to_builtin(term.params),
        })
    return items


def build_markdown(manifest):
    lines = []
    lines.append(f"# {manifest['task']}")
    lines.append("")
    lines.append(f"Generated: {manifest['generated_at']}")
    lines.append("")
    lines.append("## Observation Shapes")
    for group_name, group in manifest["observations"].items():
        lines.append(f"- {group_name}: {group['shape']}")
        for term in group["terms"]:
            lines.append(f"  - {term['name']}: {term['func']}")
    lines.append("")
    lines.append("## Networks")
    lines.append(f"- actor: {manifest['policy_model']['actor_input_dim']} -> {manifest['policy_model']['actor_hidden_dims']} -> {manifest['policy_model']['action_dim']}")
    lines.append(f"- critic: {manifest['policy_model']['critic_input_dim']} -> {manifest['policy_model']['critic_hidden_dims']} -> 1")
    lines.append("")
    lines.append("## Commands")
    for key, value in manifest['commands'].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Randomization")
    for event in manifest['events']:
        lines.append(f"- {event['name']} ({event['mode']}): {event['params']}")
    lines.append("")
    lines.append("## Rewards")
    for reward in manifest['rewards']:
        lines.append(f"- {reward['name']}: weight={reward['weight']} func={reward['func']}")
    return "\n".join(lines) + "\n"


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, 'env_cfg_entry_point')
    agent_cfg_obj = load_cfg_from_registry(args_cli.task, args_cli.agent)
    agent_cfg = agent_cfg_obj.to_dict()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    policy_obs = obs['policy']
    privileged_obs = obs['privileged'] if 'privileged' in obs else None

    robot_cfg = env_cfg.scene.robot
    base_legs = robot_cfg.actuators.get('base_legs') if hasattr(robot_cfg, 'actuators') else None
    base_vel_cmd = env_cfg.commands.base_velocity
    joint_action = env_cfg.actions.joint_pos

    if args_cli.output_prefix:
        prefix = Path(args_cli.output_prefix)
    else:
        if args_cli.run_dir:
            run_name = Path(args_cli.run_dir).name
            prefix = Path('artifacts/manifests') / f"go2_rma_teacher_stage_b_{run_name}_manifest"
        else:
            prefix = Path('artifacts/manifests') / f"go2_rma_teacher_stage_b_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_manifest"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'task': args_cli.task,
        'seed': args_cli.seed,
        'num_envs_for_introspection': args_cli.num_envs,
        'observations': {
            'policy': obs_group_manifest(env_cfg.observations.policy, policy_obs),
            'privileged': obs_group_manifest(env_cfg.observations.privileged, privileged_obs),
        },
        'commands': {
            'resampling_time_range_s': to_builtin(base_vel_cmd.resampling_time_range),
            'heading_command': base_vel_cmd.heading_command,
            'heading_control_stiffness': getattr(base_vel_cmd, 'heading_control_stiffness', None),
            'rel_standing_envs': getattr(base_vel_cmd, 'rel_standing_envs', None),
            'rel_heading_envs': getattr(base_vel_cmd, 'rel_heading_envs', None),
            'lin_vel_x': to_builtin(base_vel_cmd.ranges.lin_vel_x),
            'lin_vel_y': to_builtin(base_vel_cmd.ranges.lin_vel_y),
            'ang_vel_z': to_builtin(base_vel_cmd.ranges.ang_vel_z),
            'heading': to_builtin(getattr(base_vel_cmd.ranges, 'heading', None)),
        },
        'actions': {
            'type': joint_action.__class__.__name__,
            'scale': joint_action.scale,
            'use_default_offset': joint_action.use_default_offset,
            'joint_names': to_builtin(joint_action.joint_names),
        },
        'policy_model': {
            'class_name': agent_cfg['policy']['class_name'],
            'actor_input_dim': agent_cfg['policy']['num_actor_obs'],
            'critic_input_dim': agent_cfg['policy']['num_critic_obs'],
            'actor_hidden_dims': to_builtin(agent_cfg['policy']['actor_hidden_dims']),
            'critic_hidden_dims': to_builtin(agent_cfg['policy']['critic_hidden_dims']),
            'latent_dim': 8,
            'action_dim': 12,
            'pretrained_path': agent_cfg['policy'].get('pretrained_path'),
        },
        'robot_defaults': {
            'spawn_usd': to_builtin(robot_cfg.spawn.usd_path),
            'base_init_pos': to_builtin(robot_cfg.init_state.pos),
            'joint_pos_defaults': to_builtin(robot_cfg.init_state.joint_pos),
            'joint_vel_defaults': to_builtin(robot_cfg.init_state.joint_vel),
            'soft_joint_pos_limit_factor': to_builtin(getattr(robot_cfg, 'soft_joint_pos_limit_factor', None)),
            'actuator': {
                'effort_limit': to_builtin(getattr(base_legs, 'effort_limit', None)) if base_legs else None,
                'saturation_effort': to_builtin(getattr(base_legs, 'saturation_effort', None)) if base_legs else None,
                'velocity_limit': to_builtin(getattr(base_legs, 'velocity_limit', None)) if base_legs else None,
                'stiffness_nominal': to_builtin(getattr(base_legs, 'stiffness', None)) if base_legs else None,
                'damping_nominal': to_builtin(getattr(base_legs, 'damping', None)) if base_legs else None,
                'friction_nominal': to_builtin(getattr(base_legs, 'friction', None)) if base_legs else None,
            },
        },
        'events': events_manifest(env_cfg.events),
        'rewards': rewards_manifest(env_cfg.rewards),
        'curriculum': curriculum_manifest(env_cfg.curriculum),
        'runtime_sample': runtime_sample_manifest(env, obs),
    }

    yaml_path = prefix.with_suffix('.yaml')
    md_path = prefix.with_suffix('.md')
    yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False))
    md_path.write_text(build_markdown(manifest))

    print(f'[INFO] Wrote YAML manifest to: {yaml_path}')
    print(f'[INFO] Wrote Markdown summary to: {md_path}')

    env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
