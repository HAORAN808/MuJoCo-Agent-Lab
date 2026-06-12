from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPABILITY_DIR = PROJECT_ROOT / "capabilities"


@dataclass(frozen=True)
class RobotCapability:
    robot_id: str
    name: str
    dof: int
    workspace: Dict[str, Any]
    payload_kg: float
    end_effectors: List[str]
    supported_skills: List[str]
    control_modes: List[str]
    safety_limits: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCapability:
    tool_id: str
    function: List[str]
    attachable_to: List[str]
    geometry: Dict[str, Any]
    required_force_range_n: List[float]


@dataclass(frozen=True)
class ObjectCapability:
    object_id: str
    category: str
    size_m: List[float]
    mass_kg: float
    interaction_zones: List[str]
    supported_tasks: List[str]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_robot_capabilities() -> List[RobotCapability]:
    data = _load_json(CAPABILITY_DIR / "robots.json")
    return [RobotCapability(**row) for row in data.get("robots", [])]


def list_tool_capabilities() -> List[ToolCapability]:
    data = _load_json(CAPABILITY_DIR / "tools.json")
    return [ToolCapability(**row) for row in data.get("tools", [])]


def list_object_capabilities() -> List[ObjectCapability]:
    data = _load_json(CAPABILITY_DIR / "objects.json")
    return [ObjectCapability(**row) for row in data.get("objects", [])]


def _point_in_workspace(point: Iterable[float], workspace: Mapping[str, Any]) -> bool:
    values = list(point)
    if len(values) < 3:
        return True
    ranges = [
        workspace.get("x_range_m"),
        workspace.get("y_range_m"),
        workspace.get("z_range_m"),
    ]
    for value, bounds in zip(values[:3], ranges):
        if not bounds:
            continue
        if float(value) < float(bounds[0]) or float(value) > float(bounds[1]):
            return False
    return True


def select_robot_for_task(task_spec: Mapping[str, Any], required_skills: Iterable[str]) -> Dict[str, Any]:
    requested = set(task_spec.get("robot_candidates") or [])
    required = set(required_skills)
    objects = task_spec.get("objects") or []
    target_points = [
        obj.get("initial_position")
        for obj in objects
        if isinstance(obj, Mapping) and obj.get("initial_position")
    ]
    candidates = []
    for robot in list_robot_capabilities():
        if requested and robot.robot_id not in requested:
            continue
        missing = sorted(required - set(robot.supported_skills))
        workspace_ok = all(_point_in_workspace(point, robot.workspace) for point in target_points)
        score = robot.dof + len(required & set(robot.supported_skills)) * 5
        if missing:
            score -= len(missing) * 10
        if not workspace_ok:
            score -= 20
        candidates.append(
            {
                "robot_id": robot.robot_id,
                "score": score,
                "missing_skills": missing,
                "workspace_ok": workspace_ok,
                "reason": (
                    "matches required skills and workspace"
                    if not missing and workspace_ok
                    else "candidate retained for explanation but has gaps"
                ),
            }
        )
    candidates.sort(key=lambda row: row["score"], reverse=True)
    selected = candidates[0] if candidates else None
    return {
        "selected_robot_id": selected["robot_id"] if selected else "",
        "candidates": candidates,
        "executable": bool(selected and not selected["missing_skills"] and selected["workspace_ok"]),
    }


def select_tool_for_task(task_type: str, robot_id: str, preferred_tool_id: str = "") -> Dict[str, Any]:
    if not preferred_tool_id:
        return {
            "selected_tool_id": "",
            "candidates": [],
            "reason": "No held tool is required by this task.",
        }
    matches = []
    for tool in list_tool_capabilities():
        if robot_id not in tool.attachable_to:
            continue
        if preferred_tool_id and tool.tool_id != preferred_tool_id:
            continue
        if task_type in tool.function or (task_type == "screwdriving" and "screw" in tool.function):
            matches.append(
                {
                    "tool_id": tool.tool_id,
                    "function": tool.function,
                    "reason": f"supports {task_type} and can attach to {robot_id}",
                }
            )
    return {
        "selected_tool_id": matches[0]["tool_id"] if matches else "",
        "candidates": matches,
    }


def explain_object_support(task_spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    task_type = str(task_spec.get("task_type") or "")
    library = {obj.object_id: obj for obj in list_object_capabilities()}
    rows = []
    for obj in task_spec.get("objects") or []:
        object_id = str(obj.get("object_id", ""))
        capability = library.get(object_id)
        rows.append(
            {
                "object_id": object_id,
                "known": capability is not None,
                "supports_task": bool(capability and task_type in capability.supported_tasks),
                "category": capability.category if capability else "",
            }
        )
    return rows
