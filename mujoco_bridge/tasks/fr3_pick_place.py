from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .base import TaskSpec


FR3_PICK_PLACE_SPEC = TaskSpec(
    task_id="fr3_pick_place",
    title="Franka FR3 pick-and-place",
    description=(
        "A Franka FR3 arm uses a two-finger gripper to pick up a cube and "
        "place it near a target under pose, contact, and perception changes."
    ),
    keywords=[
        "pick",
        "place",
        "grasp",
        "gripper",
        "franka",
        "fr3",
        "抓",
        "抓取",
        "夹爪",
        "放置",
        "搬运",
    ],
    experiment_space={
        "object_offset": ["small", "medium", "large"],
        "friction": ["low", "medium", "high"],
        "grasp_height_offset": ["-2cm", "0", "+2cm"],
        "vision_noise": ["none", "light", "heavy"],
        "control_freq": ["normal"],
    },
    metrics=[
        "success_rate",
        "failure_type",
        "trajectory_error",
        "final_distance",
        "max_grip_force",
        "touch_contact_duration",
    ],
    failure_types=["grasp_miss", "slip", "collision", "timeout"],
    supported_objects=["cube_5cm", "cube_7cm", "rect_block", "cylinder_can"],
    execution_kind="robot_arm_simulation",
    manipulation_actor="Franka FR3 arm with two-finger gripper",
    fidelity_notes=[
        "Uses a MuJoCo FR3 arm scene and scripted pick-and-place control.",
        "This is the only current task that represents a full mechanical-arm manipulation loop.",
    ],
    runner_module="mujoco_bridge.runner",
)


class FR3PickPlaceTask:
    spec = FR3_PICK_PLACE_SPEC

    def build_matrix(
        self,
        limit: int,
        experiment_space: Mapping[str, Sequence[str]] | None = None,
    ) -> List[Any]:
        from ..runner import build_experiment_matrix

        return build_experiment_matrix(limit=limit, experiment_space=experiment_space)

    def run_experiments(
        self,
        limit: int,
        use_fallback: bool = False,
        experiment_space: Mapping[str, Sequence[str]] | None = None,
    ) -> Dict[str, Any]:
        from ..runner import run_experiments

        result = run_experiments(
            limit=limit,
            use_fallback=use_fallback,
            experiment_space=experiment_space,
        )
        result["task_id"] = self.spec.task_id
        return result

    def demo_trace(self, use_fallback: bool = False) -> Dict[str, Any]:
        from ..arm_runner import ArmSkillConfig, FR3ArmSkillRunner
        from ..task_render import PROJECT_ROOT

        trace = FR3ArmSkillRunner().render_skill_replay(
            ArmSkillConfig("pick_lift", object_id="rect_block", object_xy=(0.59, 0.0)),
            output_dir=PROJECT_ROOT / "web_demo" / "assets" / "fr3_pick_place_replay",
            web_prefix="assets/fr3_pick_place_replay",
        )
        trace["task_id"] = self.spec.task_id
        trace["title"] = "FR3 pick-and-place MuJoCo replay"
        return trace
