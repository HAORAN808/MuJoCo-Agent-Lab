from __future__ import annotations

from typing import Any, Dict, Mapping

from ..protocols import ExperimentPlan, ExperimentVariable


DEFAULT_METRICS = {
    "press": ["success_rate", "contact_steps", "max_touch_force", "failure_type"],
    "insert": ["success_rate", "alignment_error", "tool_contact_steps", "failure_type"],
    "screwdriving": ["success_rate", "driver_alignment", "tool_contact_steps", "failure_type"],
    "push": ["success_rate", "object_displacement", "contact_steps", "failure_type"],
    "tool_use": ["success_rate", "tool_contact_steps", "object_displacement", "failure_type"],
    "pick_place": ["success_rate", "final_distance", "lifted_height", "failure_type"],
    "assembly": ["success_rate", "alignment_error", "completion_time", "failure_type"],
    "general": ["success_rate", "failure_type"],
}


def build_experiment_plan(task_spec: Mapping[str, Any], run_count: int = 3) -> ExperimentPlan:
    task_type = str(task_spec.get("task_type") or "general")
    variables = [
        ExperimentVariable(
            name=str(name),
            values=list(values),
            reason=f"Vary {name} to test sensitivity for {task_type}.",
        )
        for name, values in dict(task_spec.get("experiment_variables") or {}).items()
        if values
    ]
    if not variables:
        variables = [
            ExperimentVariable(
                name="initial_pose_noise",
                values=["none", "small"],
                reason="Minimal robustness probe when no explicit variable is available.",
            )
        ]

    controls = [
        "same_robot_model",
        "same_workspace",
        "same_primary_object",
        "same_success_criteria",
    ]
    if task_spec.get("held_tool_id"):
        controls.append("same_tool_attachment")

    return ExperimentPlan(
        plan_id=f"exp_{task_spec.get('task_id', task_type)}",
        task_id=str(task_spec.get("task_id") or task_type),
        hypothesis=f"Changing key controllable variables will affect {task_type} success rate and failure modes.",
        variables=variables,
        controls=controls,
        metrics=DEFAULT_METRICS.get(task_type, DEFAULT_METRICS["general"]),
        run_count=max(1, int(run_count)),
        expected_result="The run should produce success/failure labels, task metrics, and an actionable next experiment.",
        risk_notes=[
            "This plan is a baseline experiment plan; it should not be interpreted as a trained policy.",
            "Low sample counts are smoke-test evidence only.",
        ],
    )
