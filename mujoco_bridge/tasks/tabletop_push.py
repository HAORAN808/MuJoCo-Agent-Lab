from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ..object_library import get_object
from .base import TaskSpec, supported_space


PUSH_EXPERIMENT_SPACE = {
    "object_type": ["cube_5cm", "rect_block", "cylinder_can", "small_sphere", "flat_puck"],
    "object_mass": ["light", "medium", "heavy"],
    "friction": ["low", "medium", "high"],
    "push_angle": ["straight", "left", "right"],
    "target_distance": ["near", "medium", "far"],
    "control_freq": ["normal"],
}


TABLETOP_PUSH_SPEC = TaskSpec(
    task_id="tabletop_push",
    title="Tabletop object pushing",
    description=(
        "A planar pusher contacts a cube on a tabletop and pushes it toward "
        "a target. The scene is separate from the FR3 grasping scene and uses "
        "different physical variables."
    ),
    keywords=[
        "push",
        "pushing",
        "slide",
        "tabletop",
        "推",
        "推动",
        "推箱子",
        "滑动",
        "桌面",
    ],
    experiment_space=PUSH_EXPERIMENT_SPACE,
    metrics=[
        "success_rate",
        "failure_type",
        "trajectory_error",
        "final_distance",
        "contact_steps",
        "max_push_force",
    ],
    failure_types=["undershoot", "overshoot", "lateral_drift", "lost_contact"],
    supported_objects=[
        "cube_5cm",
        "cube_7cm",
        "rect_block",
        "cylinder_can",
        "small_sphere",
        "flat_puck",
    ],
    execution_kind="task_specific_simulation",
    manipulation_actor="planar MuJoCo pusher, not a robot arm",
    fidelity_notes=[
        "Uses real MuJoCo contact physics for a table, object, target, and pusher.",
        "The current runner does not control a full robot arm or solve arm kinematics.",
    ],
    runner_module="mujoco_bridge.tasks.tabletop_push",
)


@dataclass(frozen=True)
class PushConfig:
    run_id: str
    object_type: str
    object_mass: str
    friction: str
    push_angle: str
    target_distance: str
    control_freq: str = "normal"


@dataclass
class PushRunResult:
    run_id: str
    object_type: str
    object_mass: str
    friction: str
    push_angle: str
    target_distance: str
    control_freq: str
    success: bool
    failure_type: str
    trajectory_error: float
    collision_count: int
    final_distance: float
    contact_steps: int
    max_push_force: float
    overshoot: float
    lateral_error: float


MASS_VALUES = {"light": 0.05, "medium": 0.11, "heavy": 0.22}
FRICTION_VALUES = {"low": 0.18, "medium": 0.55, "high": 1.1}
DISTANCE_VALUES = {"near": 0.18, "medium": 0.25, "far": 0.32}
ANGLE_Y = {"straight": 0.0, "left": 0.045, "right": -0.045}


def seeded_noise(seed: int) -> float:
    value = math.sin(seed * 12.9898) * 43758.5453
    return value - math.floor(value)


def build_push_matrix(
    limit: int = 81,
    experiment_space: Mapping[str, Sequence[str]] | None = None,
) -> List[PushConfig]:
    space = supported_space(PUSH_EXPERIMENT_SPACE, experiment_space)
    rows: List[PushConfig] = []
    idx = 1
    if (
        limit == 27
        and len(space["object_type"]) >= 1
        and len(space["object_mass"]) == 3
        and len(space["friction"]) == 3
        and len(space["push_angle"]) == 3
        and len(space["target_distance"]) == 3
    ):
        for object_mass in space["object_mass"]:
            for friction in space["friction"]:
                for push_angle in space["push_angle"]:
                    target_distance = space["target_distance"][(idx - 1) % 3]
                    object_type = space["object_type"][(idx - 1) % len(space["object_type"])]
                    rows.append(
                        PushConfig(
                            run_id=f"push_{idx:03d}",
                            object_type=object_type,
                            object_mass=object_mass,
                            friction=friction,
                            push_angle=push_angle,
                            target_distance=target_distance,
                            control_freq=space["control_freq"][0],
                        )
                    )
                    idx += 1
        return rows

    for object_type in space["object_type"]:
        for object_mass in space["object_mass"]:
            for friction in space["friction"]:
                for push_angle in space["push_angle"]:
                    for target_distance in space["target_distance"]:
                        rows.append(
                            PushConfig(
                                run_id=f"push_{idx:03d}",
                                object_type=object_type,
                                object_mass=object_mass,
                                friction=friction,
                                push_angle=push_angle,
                                target_distance=target_distance,
                                control_freq=space["control_freq"][0],
                            )
                        )
                        idx += 1

    return rows[:limit]


def _classify_push_failure(
    config: PushConfig,
    final_x: float,
    final_y: float,
    target_x: float,
    contact_steps: int,
) -> str:
    if contact_steps < 20:
        return "lost_contact"
    if abs(final_y) > 0.055:
        return "lateral_drift"
    if final_x > target_x + 0.065:
        return "overshoot"
    return "undershoot"


class FallbackPushRunner:
    source = "fallback"

    def run_one(self, config: PushConfig, index: int) -> PushRunResult:
        target_x = DISTANCE_VALUES[config.target_distance]
        mass_penalty = {"light": 0.0, "medium": 0.015, "heavy": 0.045}[config.object_mass]
        friction_penalty = {"low": 0.035, "medium": 0.0, "high": 0.03}[config.friction]
        shape_penalty = {
            "cube_5cm": 0.0,
            "rect_block": 0.012,
            "cylinder_can": 0.02,
            "small_sphere": 0.035,
            "flat_puck": 0.006,
        }.get(config.object_type, 0.01)
        angle_error = abs(ANGLE_Y[config.push_angle])
        jitter = (seeded_noise(index + 31) - 0.5) * 0.025
        final_x = target_x - mass_penalty - friction_penalty - shape_penalty + jitter
        final_y = ANGLE_Y[config.push_angle] * (0.8 + seeded_noise(index + 32) * 0.5)
        final_distance = math.hypot(final_x - target_x, final_y)
        success = final_distance < 0.055 and angle_error < 0.05
        failure_type = (
            "none"
            if success
            else _classify_push_failure(config, final_x, final_y, target_x, 80)
        )
        return PushRunResult(
            **asdict(config),
            success=success,
            failure_type=failure_type,
            trajectory_error=round(final_distance + angle_error * 0.4, 3),
            collision_count=0,
            final_distance=round(final_distance, 3),
            contact_steps=80,
            max_push_force=round(4.0 + mass_penalty * 45 + friction_penalty * 30, 3),
            overshoot=round(max(0.0, final_x - target_x), 3),
            lateral_error=round(abs(final_y), 3),
        )


class MujocoPushRunner:
    source = "mujoco_push"

    def __init__(self) -> None:
        try:
            import mujoco  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "Python package 'mujoco' is not available. Start the server with "
                "--fallback or install mujoco before using the push runner."
            ) from exc
        self.mujoco = mujoco

    def _xml(self, config: PushConfig) -> str:
        obj = get_object(config.object_type)
        mass = MASS_VALUES[config.object_mass]
        friction = FRICTION_VALUES[config.friction]
        target_x = DISTANCE_VALUES[config.target_distance]
        geom_type = obj.geometry
        geom_size = " ".join(str(v) for v in obj.size_m)
        if geom_type == "box":
            z_pos = obj.size_m[2]
        elif geom_type == "sphere":
            z_pos = obj.size_m[0]
        elif geom_type == "cylinder":
            z_pos = obj.size_m[1]
        else:
            z_pos = 0.025
        return f"""
<mujoco model="tabletop_push">
  <option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 -0.6 0.8" dir="0 1 -1"/>
    <geom name="floor" type="plane" size="0.8 0.45 0.02" rgba="0.86 0.88 0.90 1" friction="{friction} 0.005 0.0001"/>
    <site name="target" pos="{target_x} 0 0.004" type="box" size="0.035 0.035 0.002" rgba="0.1 0.7 0.4 0.35"/>
    <body name="cube" pos="0 0 {z_pos}">
      <freejoint/>
      <geom name="cube_geom" type="{geom_type}" size="{geom_size}" mass="{mass}" rgba="0.1 0.35 0.9 1" friction="{friction} 0.005 0.0001"/>
    </body>
    <body name="pusher" pos="-0.105 0 0.025">
      <joint name="px" type="slide" axis="1 0 0" range="-0.16 0.45" damping="4"/>
      <joint name="py" type="slide" axis="0 1 0" range="-0.18 0.18" damping="4"/>
      <geom name="pusher_geom" type="box" size="0.018 0.055 0.025" mass="5" rgba="0.16 0.20 0.24 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="px" kp="900" ctrlrange="-0.16 0.45"/>
    <position name="ay" joint="py" kp="600" ctrlrange="-0.18 0.18"/>
  </actuator>
</mujoco>
"""

    def run_one(self, config: PushConfig, index: int) -> PushRunResult:
        mujoco = self.mujoco
        target_x = DISTANCE_VALUES[config.target_distance]
        model = mujoco.MjModel.from_xml_string(self._xml(config))
        data = mujoco.MjData(model)
        cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        pusher_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pusher_geom")
        cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        push_y = ANGLE_Y[config.push_angle]
        contact_steps = 0
        max_push_force = 0.0
        steps = 430
        for step in range(steps):
            phase = step / (steps - 1)
            forward = -0.105 + (target_x + 0.105) * min(1.0, phase * 1.18)
            data.ctrl[0] = forward
            data.ctrl[1] = push_y * min(1.0, phase * 1.4)
            mujoco.mj_step(model, data)

            for contact_idx in range(data.ncon):
                contact = data.contact[contact_idx]
                pair = {contact.geom1, contact.geom2}
                if pusher_geom in pair and cube_geom in pair:
                    contact_steps += 1
                    force = np.zeros(6, dtype=float)
                    mujoco.mj_contactForce(model, data, contact_idx, force)
                    max_push_force = max(max_push_force, math.sqrt(sum(v * v for v in force[:3])))

        final_x = float(data.xpos[cube_body][0])
        final_y = float(data.xpos[cube_body][1])
        final_distance = math.hypot(final_x - target_x, final_y)
        success = final_distance < 0.055 and contact_steps >= 20
        failure_type = (
            "none"
            if success
            else _classify_push_failure(config, final_x, final_y, target_x, contact_steps)
        )
        return PushRunResult(
            **asdict(config),
            success=success,
            failure_type=failure_type,
            trajectory_error=round(final_distance + abs(final_y) * 0.5, 3),
            collision_count=0,
            final_distance=round(final_distance, 3),
            contact_steps=contact_steps,
            max_push_force=round(max_push_force, 3),
            overshoot=round(max(0.0, final_x - target_x), 3),
            lateral_error=round(abs(final_y), 3),
        )


class TabletopPushTask:
    spec = TABLETOP_PUSH_SPEC

    def build_matrix(
        self,
        limit: int,
        experiment_space: Mapping[str, Sequence[str]] | None = None,
    ) -> List[PushConfig]:
        return build_push_matrix(limit=limit, experiment_space=experiment_space)

    def run_experiments(
        self,
        limit: int,
        use_fallback: bool = False,
        experiment_space: Mapping[str, Sequence[str]] | None = None,
    ) -> Dict[str, Any]:
        runner: Any = FallbackPushRunner() if use_fallback else MujocoPushRunner()
        configs = self.build_matrix(limit=limit, experiment_space=experiment_space)
        runs = [asdict(runner.run_one(config, i + 1)) for i, config in enumerate(configs)]
        return {
            "source": runner.source,
            "task_id": self.spec.task_id,
            "runs": runs,
        }

    def demo_trace(self, use_fallback: bool = False) -> Dict[str, Any]:
        if not use_fallback:
            from ..task_render import PROJECT_ROOT, render_controlled_scene

            config = PushConfig(
                run_id="push_demo",
                object_type="cube_5cm",
                object_mass="medium",
                friction="medium",
                push_angle="straight",
                target_distance="medium",
            )
            runner = MujocoPushRunner()
            mujoco = runner.mujoco
            model = mujoco.MjModel.from_xml_string(runner._xml(config))
            data = mujoco.MjData(model)
            target_x = DISTANCE_VALUES[config.target_distance]

            def control(step: int) -> str:
                phase = step / 429
                data.ctrl[0] = -0.105 + (target_x + 0.105) * min(1.0, phase * 1.18)
                data.ctrl[1] = 0.0
                if phase < 0.28:
                    return "approach"
                if phase < 0.82:
                    return "push"
                return "settle"

            images, labels = render_controlled_scene(
                model,
                data,
                control,
                steps=430,
                output_dir=PROJECT_ROOT / "web_demo" / "assets" / "tabletop_push_replay",
                web_prefix="assets/tabletop_push_replay",
            )
            return {
                "source": "mujoco_push",
                "task_id": self.spec.task_id,
                "model": "MuJoCo tabletop pushing scene",
                "title": "Tabletop push MuJoCo replay",
                "image_frames": images,
                "labels": labels,
                "width": 560,
                "height": 315,
            }

        target_x = DISTANCE_VALUES["medium"]
        frames = [
            {
                "label": "reset",
                "time": 0.0,
                "gripper": {"x": -0.105, "y": 0.0, "z": 0.025, "closed": 0},
                "cube": {"x": 0.0, "y": 0.0, "z": 0.025},
                "target": {"x": target_x, "y": 0.0, "z": 0.025},
            },
            {
                "label": "contact",
                "time": 0.5,
                "gripper": {"x": -0.035, "y": 0.0, "z": 0.025, "closed": 0},
                "cube": {"x": 0.0, "y": 0.0, "z": 0.025},
                "target": {"x": target_x, "y": 0.0, "z": 0.025},
            },
            {
                "label": "push",
                "time": 1.2,
                "gripper": {"x": 0.14, "y": 0.0, "z": 0.025, "closed": 0},
                "cube": {"x": 0.16, "y": 0.0, "z": 0.025},
                "target": {"x": target_x, "y": 0.0, "z": 0.025},
            },
            {
                "label": "settle",
                "time": 1.8,
                "gripper": {"x": 0.25, "y": 0.0, "z": 0.025, "closed": 0},
                "cube": {"x": target_x, "y": 0.0, "z": 0.025},
                "target": {"x": target_x, "y": 0.0, "z": 0.025},
            },
        ]
        return {
            "source": "fallback" if use_fallback else "mujoco_push",
            "task_id": self.spec.task_id,
            "model": "MuJoCo tabletop pushing scene",
            "frames": frames,
        }
