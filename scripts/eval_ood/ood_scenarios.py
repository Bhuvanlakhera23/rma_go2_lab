"""Exploratory OOD scenario manifests for frozen baseline probes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OODScenario:
    name: str
    terrain_type: str | None = None
    terrain_level: int | None = None
    latent_mode: str | None = None
    static_friction: float | None = None
    dynamic_friction: float | None = None
    mass_offset: float | None = None
    motor_stiffness_scale: float | None = None
    motor_damping_scale: float | None = None
    push_interval_s: tuple[float, float] | None = None
    push_velocity_range: dict[str, tuple[float, float]] | None = None
    switch_step: int | None = None
    switch_static_friction: float | None = None
    switch_dynamic_friction: float | None = None
    switch_mass_offset: float | None = None
    switch_motor_stiffness_scale: float | None = None
    switch_motor_damping_scale: float | None = None


def scenario_set(name: str) -> list[OODScenario]:
    if name == "ood_geometry_v1":
        return [
            OODScenario("ood_stairs_up_l5", terrain_type="pyramid_stairs", terrain_level=5),
            OODScenario("ood_stairs_down_l5", terrain_type="pyramid_stairs_inv", terrain_level=5),
            OODScenario("ood_boxes_l5", terrain_type="boxes", terrain_level=5),
            OODScenario("ood_random_rough_l9", terrain_type="random_rough", terrain_level=9),
        ]

    if name == "ood_dynamics_v1":
        return [
            OODScenario(
                "ood_ultra_low_friction_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                static_friction=0.05,
                dynamic_friction=0.05,
            ),
            OODScenario(
                "ood_ultra_high_friction_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                static_friction=3.0,
                dynamic_friction=3.0,
            ),
            OODScenario(
                "ood_very_heavy_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                mass_offset=6.0,
            ),
            OODScenario(
                "ood_very_weak_motor_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                motor_stiffness_scale=0.5,
                motor_damping_scale=0.5,
            ),
        ]

    if name == "ood_combo_v1":
        return [
            OODScenario(
                "ood_low_friction_heavy_mass_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                static_friction=0.1,
                dynamic_friction=0.1,
                mass_offset=4.0,
            ),
            OODScenario(
                "ood_low_friction_weak_motor_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                static_friction=0.1,
                dynamic_friction=0.1,
                motor_stiffness_scale=0.6,
                motor_damping_scale=0.6,
            ),
            OODScenario(
                "ood_heavy_mass_weak_motor_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                mass_offset=4.0,
                motor_stiffness_scale=0.6,
                motor_damping_scale=0.6,
            ),
            OODScenario(
                "ood_triple_combo_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                static_friction=0.1,
                dynamic_friction=0.1,
                mass_offset=4.0,
                motor_stiffness_scale=0.6,
                motor_damping_scale=0.6,
            ),
        ]

    if name == "ood_push_v1":
        return [
            OODScenario(
                "ood_push_lateral_medium_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                push_interval_s=(2.5, 2.5),
                push_velocity_range={"y": (-0.8, 0.8)},
            ),
            OODScenario(
                "ood_push_forward_medium_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                push_interval_s=(2.5, 2.5),
                push_velocity_range={"x": (-0.8, 0.8)},
            ),
            OODScenario(
                "ood_push_yaw_medium_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                push_interval_s=(2.5, 2.5),
                push_velocity_range={"yaw": (-0.8, 0.8)},
            ),
            OODScenario(
                "ood_push_lateral_repeated_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                push_interval_s=(1.25, 1.25),
                push_velocity_range={"y": (-0.5, 0.5)},
            ),
        ]

    if name == "ood_switch_v1":
        return [
            OODScenario(
                "ood_switch_ultra_low_friction_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                switch_step=300,
                switch_static_friction=0.05,
                switch_dynamic_friction=0.05,
            ),
            OODScenario(
                "ood_switch_very_heavy_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                switch_step=300,
                switch_mass_offset=6.0,
            ),
            OODScenario(
                "ood_switch_very_weak_motor_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                switch_step=300,
                switch_motor_stiffness_scale=0.5,
                switch_motor_damping_scale=0.5,
            ),
            OODScenario(
                "ood_switch_low_friction_heavy_random_rough_l5",
                terrain_type="random_rough",
                terrain_level=5,
                switch_step=300,
                switch_static_friction=0.1,
                switch_dynamic_friction=0.1,
                switch_mass_offset=4.0,
            ),
        ]

    if name == "ood_limit_v1":
        return [
            *scenario_set("ood_geometry_v1"),
            *scenario_set("ood_dynamics_v1"),
            *scenario_set("ood_combo_v1"),
            *scenario_set("ood_push_v1"),
            *scenario_set("ood_switch_v1"),
        ]

    raise ValueError(
        f"Unknown OOD scenario set '{name}'. "
        "Expected one of: ood_geometry_v1, ood_dynamics_v1, ood_combo_v1, ood_push_v1, ood_switch_v1, ood_limit_v1"
    )
