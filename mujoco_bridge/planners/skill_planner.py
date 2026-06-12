from __future__ import annotations

from typing import Any, Mapping

from ..capabilities import select_robot_for_task
from ..protocols import SkillPlan
from ..skills.library import task_actions_to_skill_steps


def build_skill_plan(task_spec: Mapping[str, Any]) -> SkillPlan:
    actions = list(task_spec.get("actions") or [])
    skills = task_actions_to_skill_steps(actions)
    required_skill_names = [step.skill_name for step in skills]
    robot_selection = select_robot_for_task(task_spec, required_skill_names)
    robot_id = robot_selection.get("selected_robot_id") or (
        task_spec.get("robot_candidates") or ["franka_fr3"]
    )[0]

    return SkillPlan(
        plan_id=f"skill_{task_spec.get('task_id', 'dynamic')}",
        robot_id=str(robot_id),
        end_effector=str(task_spec.get("held_tool_id") or "default_gripper"),
        skills=skills,
        safety_limits=dict(task_spec.get("safety_limits") or {}),
    )
