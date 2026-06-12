from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .base import ExperimentTask
from .arm_primitives import FR3ArmPrimitiveTask
from .asset_manipulation import (
    AssemblyInsertionTask,
    ClothFoldingTask,
    ScrewdrivingTask,
    ToolUseTask,
)
from .fr3_pick_place import FR3PickPlaceTask
from .tabletop_push import TabletopPushTask


TASKS: Dict[str, ExperimentTask] = {
    "fr3_arm_primitives": FR3ArmPrimitiveTask(),
    "fr3_pick_place": FR3PickPlaceTask(),
    "tabletop_push": TabletopPushTask(),
    "screwdriving": ScrewdrivingTask(),
    "tool_use": ToolUseTask(),
    "assembly_insertion": AssemblyInsertionTask(),
    "cloth_folding": ClothFoldingTask(),
}


def get_task(task_id: str) -> ExperimentTask:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        supported = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task_id '{task_id}'. Supported tasks: {supported}") from exc


def list_task_specs() -> List[Dict[str, Any]]:
    return [asdict(task.spec) for task in TASKS.values()]


def resolve_task(goal: str, preferred_task_id: str | None = None) -> ExperimentTask:
    if preferred_task_id and preferred_task_id in TASKS:
        return TASKS[preferred_task_id]

    normalized = goal.lower()
    scores: Dict[str, int] = {}
    for task_id, task in TASKS.items():
        scores[task_id] = sum(1 for keyword in task.spec.keywords if keyword.lower() in normalized)

    selected_id = max(scores, key=lambda key: scores[key])
    if scores[selected_id] <= 0:
        selected_id = "fr3_arm_primitives"
    return TASKS[selected_id]
