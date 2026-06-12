from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Sequence

from ..arm_runner import ArmSkillConfig, FR3ArmSkillRunner
from .base import TaskSpec, supported_space, summarize_runs


ARM_PRIMITIVE_SPACE = {
    "skill_id": ["pick_lift", "reach_touch", "button_press", "contact_sweep", "tool_contact_sweep", "peg_insert"],
    "object_id": ["rect_block", "cube_5cm", "cylinder_can", "cube_7cm", "screw_head", "insertion_socket", "button_target"],
    "tool_id": ["hammer", "spatula", "screwdriver", "peg"],
    "object_position": ["center", "left", "right"],
    "friction": ["medium", "low", "high"],
    "grasp_height_delta": ["nominal", "low", "high"],
    "sweep_scale": ["nominal", "short", "long"],
}

POSITION_XY = {
    "center": (0.59, 0.0),
    "left": (0.59, 0.025),
    "right": (0.59, -0.025),
}

FRICTION = {
    "low": 0.45,
    "medium": 0.9,
    "high": 1.4,
}

HEIGHT_DELTA = {
    "low": -0.08,
    "nominal": 0.0,
    "high": 0.06,
}

SWEEP_SCALE = {
    "short": 0.35,
    "nominal": 0.45,
    "long": 0.58,
}


ARM_PRIMITIVE_SPEC = TaskSpec(
    task_id="fr3_arm_primitives",
    title="General FR3 arm primitive experiments",
    description=(
        "A general mechanical-arm fallback task that runs implemented FR3 "
        "skills such as reach-touch, pick-lift, and contact-sweep under "
        "controlled object geometry, pose, and contact conditions."
    ),
    keywords=[
        "robot arm",
        "manipulation",
        "primitive",
        "reach",
        "touch",
        "lift",
        "contact",
        "机械臂",
        "通用",
        "基础技能",
        "接触",
        "拿起",
        "移动",
        "实验",
    ],
    experiment_space=ARM_PRIMITIVE_SPACE,
    metrics=[
        "success_rate",
        "failure_type",
        "object_id",
        "contact_steps",
        "max_touch_force",
        "object_displacement",
        "lifted_height",
        "tool_contact_steps",
    ],
    failure_types=["no_contact", "grasp_miss", "lift_failed", "weak_displacement"],
    supported_objects=["cube_5cm", "cube_7cm", "rect_block", "cylinder_can", "screw_head", "insertion_socket", "button_target"],
    execution_kind="robot_arm_skill_simulation",
    manipulation_actor="Franka FR3 arm with Franka Hand, using implemented arm skill primitives",
    fidelity_notes=[
        "Runs real MuJoCo FR3 arm skills instead of proxy carrier actuators.",
        "The experiment matrix only scans variables used by the selected skill, avoiding duplicate rows from irrelevant factors.",
        "This is a foundation for general arm experiments; it is still scripted and does not synthesize new scenes or controllers.",
    ],
    runner_module="mujoco_bridge.tasks.arm_primitives",
)


class FR3ArmPrimitiveTask:
    spec = ARM_PRIMITIVE_SPEC

    def build_matrix(
        self,
        limit: int,
        experiment_space: Mapping[str, Sequence[str]] | None = None,
    ) -> List[Dict[str, str]]:
        space = supported_space(ARM_PRIMITIVE_SPACE, experiment_space)
        rows: List[Dict[str, str]] = []
        for skill_id in space["skill_id"]:
            for object_id in space["object_id"]:
                for object_position in space["object_position"]:
                    for friction in space["friction"]:
                        if skill_id in {"contact_sweep", "tool_contact_sweep", "peg_insert"}:
                            for sweep_scale in space["sweep_scale"]:
                                tool_values = space["tool_id"] if skill_id == "tool_contact_sweep" else [""]
                                for tool_id in tool_values:
                                    rows.append(
                                        {
                                            "run_id": f"arm_{len(rows) + 1:03d}",
                                            "skill_id": skill_id,
                                            "object_id": object_id,
                                            "tool_id": tool_id,
                                            "object_position": object_position,
                                            "friction": friction,
                                            "sweep_scale": sweep_scale,
                                        }
                                    )
                                    if len(rows) >= limit:
                                        return rows
                        else:
                            for grasp_height_delta in space["grasp_height_delta"]:
                                rows.append(
                                    {
                                        "run_id": f"arm_{len(rows) + 1:03d}",
                                        "skill_id": skill_id,
                                        "object_id": object_id,
                                        "object_position": object_position,
                                        "friction": friction,
                                        "grasp_height_delta": grasp_height_delta,
                                    }
                                )
                                if len(rows) >= limit:
                                    return rows
        return rows

    def _config(self, row: Mapping[str, str]) -> ArmSkillConfig:
        return ArmSkillConfig(
            skill_id=row["skill_id"],
            object_id=row.get("object_id", "cube_5cm"),
            tool_id=row.get("tool_id", ""),
            object_xy=POSITION_XY[row["object_position"]],
            friction=FRICTION[row["friction"]],
            grasp_height_delta=HEIGHT_DELTA[row.get("grasp_height_delta", "nominal")],
            sweep_scale=SWEEP_SCALE[row.get("sweep_scale", "nominal")],
        )

    def _run_one(self, runner: FR3ArmSkillRunner, row: Mapping[str, str]) -> Dict[str, Any]:
        result = asdict(runner.run_skill(self._config(row)))
        result.pop("trace", None)
        return {
            **row,
            "success": result["success"],
            "failure_type": result["failure_type"],
            "object_id": result["object_id"],
            "contact_steps": result["contact_steps"],
            "max_touch_force": result["max_touch_force"],
            "object_displacement": result["object_displacement"],
            "lifted_height": result["lifted_height"],
            "tool_contact_steps": result["tool_contact_steps"],
            "final_object_pos": result["final_object_pos"],
            "end_effector_pos": result["end_effector_pos"],
        }

    def run_experiments(
        self,
        limit: int,
        use_fallback: bool = False,
        experiment_space: Mapping[str, Sequence[str]] | None = None,
    ) -> Dict[str, Any]:
        runner = FR3ArmSkillRunner()
        rows = [self._run_one(runner, row) for row in self.build_matrix(limit, experiment_space)]
        return {
            "source": "mujoco_fr3_arm_primitives",
            "task_id": self.spec.task_id,
            "runs": rows,
            "summary": summarize_runs(rows),
        }

    def demo_trace(self, use_fallback: bool = False) -> Dict[str, Any]:
        from ..task_render import PROJECT_ROOT

        runner = FR3ArmSkillRunner()
        trace = runner.render_skill_replay(
            ArmSkillConfig("pick_lift"),
            output_dir=PROJECT_ROOT / "web_demo" / "assets" / "fr3_arm_primitives_replay",
            web_prefix="assets/fr3_arm_primitives_replay",
        )
        trace["task_id"] = self.spec.task_id
        return trace
