from __future__ import annotations

import copy
from typing import Any, Dict, Mapping


RETRY_RULES = {
    "ik_unreachable": {
        "change": "select_nearer_target_or_robot",
        "reason": "The target pose is outside the current robot or IK constraints.",
    },
    "trajectory_error": {
        "change": "increase_motion_steps",
        "reason": "The robot reached the target poorly; slower interpolation may help.",
    },
    "no_contact": {
        "change": "lower_approach_height",
        "reason": "The end effector or tool did not touch the target.",
    },
    "missed_contact": {
        "change": "adjust_lateral_offset",
        "reason": "The contact path missed the object.",
    },
    "insufficient_force": {
        "change": "increase_downforce_or_dwell",
        "reason": "Contact was detected but did not meet the force/duration criterion.",
    },
    "alignment_error": {
        "change": "add_align_step_and_reduce_offset",
        "reason": "Contact-rich insertion or screwdriving is sensitive to pose error.",
    },
    "jammed": {
        "change": "increase_compliance_and_reduce_speed",
        "reason": "The insertion stalled before reaching the target depth.",
    },
    "driver_slip": {
        "change": "increase_downforce_and_use_vertical_approach",
        "reason": "The screwdriver did not maintain stable contact.",
    },
    "table_penetration": {
        "change": "raise_approach_and_enforce_table_clearance",
        "reason": "The end-effector violated the table clearance safety constraint.",
    },
    "overshoot": {
        "change": "reduce_push_force_or_distance",
        "reason": "The object moved farther than the requested target displacement.",
    },
    "undershoot": {
        "change": "increase_contact_or_push_distance",
        "reason": "The object moved less than the requested target displacement.",
    },
    "unknown": {
        "change": "repeat_with_more_logging",
        "reason": "The failure label is not specific enough for a targeted retry.",
    },
}


def dominant_failure(evaluation_report: Mapping[str, Any]) -> str:
    failures = list(evaluation_report.get("failure_reasons") or [])
    if not failures:
        return "none"
    failures.sort(key=lambda row: int(row.get("count", 0)), reverse=True)
    return str(failures[0].get("failure_type") or "unknown")


def build_retry_plan(task_spec: Mapping[str, Any], evaluation_report: Mapping[str, Any]) -> Dict[str, Any]:
    failure = dominant_failure(evaluation_report)
    if failure == "none":
        return {
            "should_retry": False,
            "reason": "No failure was observed in the current sample.",
            "revised_task_spec": dict(task_spec),
            "changes": [],
        }

    rule = RETRY_RULES.get(failure, RETRY_RULES["unknown"])
    revised = copy.deepcopy(dict(task_spec))
    actions = [dict(action) for action in revised.get("actions") or []]
    changes = [rule["change"]]

    if rule["change"] == "increase_motion_steps":
        for action in actions:
            action["steps"] = int(action.get("steps", 240)) + 120
    elif rule["change"] == "lower_approach_height":
        for action in actions:
            if action.get("type") in {"reach", "press"}:
                action["approach_height_delta_m"] = -0.01
    elif rule["change"] == "adjust_lateral_offset":
        revised.setdefault("experiment_variables", {})["lateral_offset"] = ["centered", "offset_3mm"]
    elif rule["change"] == "increase_downforce_or_dwell":
        revised.setdefault("experiment_variables", {})["downforce"] = ["nominal", "heavy"]
        for action in actions:
            if action.get("type") in {"press", "insert"}:
                action["duration_steps"] = int(action.get("duration_steps", 80)) + 40
    elif rule["change"] == "add_align_step_and_reduce_offset":
        if not any(action.get("type") == "align" for action in actions):
            actions.insert(0, {"type": "align", "axis": "task_axis", "tolerance": 0.01})
        revised.setdefault("experiment_variables", {})["lateral_offset"] = ["centered", "offset_5mm"]
    elif rule["change"] == "increase_compliance_and_reduce_speed":
        revised.setdefault("experiment_variables", {})["compliance"] = ["nominal", "soft"]
        revised.setdefault("safety_limits", {})["max_speed_mps"] = min(
            0.1,
            float(revised.get("safety_limits", {}).get("max_speed_mps", 0.15)),
        )
    elif rule["change"] == "increase_downforce_and_use_vertical_approach":
        revised.setdefault("experiment_variables", {})["downforce"] = ["nominal", "heavy"]
        revised.setdefault("experiment_variables", {})["approach_angle"] = ["vertical"]
    elif rule["change"] == "raise_approach_and_enforce_table_clearance":
        revised.setdefault("success_criteria", {})["min_ee_clearance_m"] = 0.02
        revised.setdefault("experiment_variables", {})["push_force"] = ["medium", "strong"]
        for action in actions:
            if isinstance(action.get("target_pos"), list) and len(action["target_pos"]) >= 3:
                action["target_pos"][2] = max(float(action["target_pos"][2]) + 0.004, 0.436)
    elif rule["change"] == "reduce_push_force_or_distance":
        revised.setdefault("experiment_variables", {})["push_force"] = ["light", "medium"]
    elif rule["change"] == "increase_contact_or_push_distance":
        revised.setdefault("experiment_variables", {})["push_force"] = ["medium", "strong"]

    revised["actions"] = actions
    return {
        "should_retry": True,
        "dominant_failure": failure,
        "reason": rule["reason"],
        "changes": changes,
        "revised_task_spec": revised,
    }
