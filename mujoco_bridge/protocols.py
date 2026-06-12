from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SUPPORTED_TASK_TYPES = {
    "pick_place",
    "press",
    "insert",
    "screwdriving",
    "push",
    "tool_use",
    "assembly",
    "general",
}

SUPPORTED_ACTIONS = {
    "reach",
    "grasp",
    "lift",
    "place",
    "push",
    "insert",
    "release",
    "wait",
    "press",
    "rotate",
    "sweep",
    "align",
}


@dataclass(frozen=True)
class TaskObject:
    object_id: str
    role: str
    initial_position: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class TaskAction:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StandardTaskSpec:
    task_id: str
    task_type: str
    goal: str
    robot_candidates: List[str]
    objects: List[TaskObject]
    actions: List[TaskAction]
    success_criteria: Dict[str, Any]
    held_tool_id: str | None = None
    workspace: str = "table"
    constraints: List[str] = field(default_factory=list)
    experiment_variables: Dict[str, List[Any]] = field(default_factory=dict)
    safety_limits: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["actions"] = [
            {"type": action.type, **action.params}
            for action in self.actions
        ]
        return data


@dataclass(frozen=True)
class ExperimentVariable:
    name: str
    values: List[Any]
    reason: str = ""


@dataclass(frozen=True)
class ExperimentPlan:
    plan_id: str
    task_id: str
    hypothesis: str
    variables: List[ExperimentVariable]
    controls: List[str]
    metrics: List[str]
    run_count: int
    expected_result: str = ""
    risk_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillStep:
    skill_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    expected_observations: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SkillPlan:
    plan_id: str
    robot_id: str
    skills: List[SkillStep]
    end_effector: str = "default"
    safety_limits: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    task_id: str
    success: bool
    success_rate: float
    sample_count: int
    success_count: int
    failure_count: int
    failure_reasons: List[Dict[str, Any]]
    confidence_notes: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _require(data: Mapping[str, Any], keys: Sequence[str], label: str) -> List[str]:
    return [f"{label}.{key} is required" for key in keys if key not in data]


def validate_task_spec(data: Mapping[str, Any]) -> List[str]:
    errors = _require(
        data,
        ["task_id", "task_type", "goal", "robot_candidates", "objects", "actions", "success_criteria"],
        "TaskSpec",
    )
    task_type = str(data.get("task_type", ""))
    if task_type and task_type not in SUPPORTED_TASK_TYPES:
        errors.append(f"TaskSpec.task_type '{task_type}' is unsupported")
    for idx, action in enumerate(data.get("actions") or []):
        if not isinstance(action, Mapping):
            errors.append(f"TaskSpec.actions[{idx}] must be an object")
            continue
        action_type = str(action.get("type", ""))
        if action_type not in SUPPORTED_ACTIONS:
            errors.append(f"TaskSpec.actions[{idx}].type '{action_type}' is unsupported")
    if not data.get("robot_candidates"):
        errors.append("TaskSpec.robot_candidates must not be empty")
    if not data.get("success_criteria", {}).get("primary"):
        errors.append("TaskSpec.success_criteria.primary is required")
    return errors


def validate_experiment_plan(data: Mapping[str, Any]) -> List[str]:
    errors = _require(
        data,
        ["plan_id", "task_id", "hypothesis", "variables", "controls", "metrics", "run_count"],
        "ExperimentPlan",
    )
    if int(data.get("run_count", 0) or 0) < 1:
        errors.append("ExperimentPlan.run_count must be >= 1")
    for idx, variable in enumerate(data.get("variables") or []):
        if not variable.get("name"):
            errors.append(f"ExperimentPlan.variables[{idx}].name is required")
        if not variable.get("values"):
            errors.append(f"ExperimentPlan.variables[{idx}].values must not be empty")
    return errors


def validate_skill_plan(data: Mapping[str, Any]) -> List[str]:
    errors = _require(data, ["plan_id", "robot_id", "skills"], "SkillPlan")
    for idx, skill in enumerate(data.get("skills") or []):
        if not skill.get("skill_name"):
            errors.append(f"SkillPlan.skills[{idx}].skill_name is required")
        if "params" not in skill:
            errors.append(f"SkillPlan.skills[{idx}].params is required")
    return errors


def evaluation_from_runs(
    task_id: str,
    runs: Iterable[Mapping[str, Any]],
    *,
    min_samples_for_confidence: int = 5,
) -> EvaluationReport:
    rows = list(runs)
    sample_count = len(rows)
    success_count = sum(1 for row in rows if bool(row.get("success")))
    failure_count = sample_count - success_count
    failures: Dict[str, int] = {}
    for row in rows:
        if row.get("success"):
            continue
        failure_type = str(row.get("failure_type") or "unknown")
        failures[failure_type] = failures.get(failure_type, 0) + 1

    failure_reasons = [
        {
            "failure_type": failure_type,
            "count": count,
            "rate": count / max(1, sample_count),
        }
        for failure_type, count in sorted(failures.items())
    ]
    success_rate = success_count / max(1, sample_count)
    confidence_notes = []
    if sample_count == 0:
        confidence_notes.append("No executable samples were recorded.")
    elif sample_count < min_samples_for_confidence:
        confidence_notes.append(
            f"Sample count is {sample_count}; treat the result as a smoke-test signal, not a generalization claim."
        )
    else:
        confidence_notes.append("Sample count is sufficient for a baseline stability check.")
    if failure_count:
        confidence_notes.append("Failure distribution is based on runner-reported failure_type labels.")

    next_actions = []
    if failure_count:
        next_actions.append("Inspect the dominant failure type and run a retry plan with narrowed parameters.")
    else:
        next_actions.append("Increase randomization and sample count before claiming cross-scene generalization.")

    return EvaluationReport(
        task_id=task_id,
        success=sample_count > 0 and failure_count == 0,
        success_rate=success_rate,
        sample_count=sample_count,
        success_count=success_count,
        failure_count=failure_count,
        failure_reasons=failure_reasons,
        confidence_notes=confidence_notes,
        next_actions=next_actions,
    )


def standard_task_from_dynamic_spec(data: Mapping[str, Any], goal: str) -> StandardTaskSpec:
    actions = []
    for action in data.get("actions") or []:
        if not isinstance(action, Mapping):
            continue
        params = {key: value for key, value in action.items() if key != "type"}
        actions.append(TaskAction(type=str(action.get("type", "wait")), params=params))
    objects = []
    for obj in data.get("objects") or []:
        if not isinstance(obj, Mapping):
            continue
        objects.append(
            TaskObject(
                object_id=str(obj.get("object_id", "")),
                role=str(obj.get("role", "target")),
                initial_position=list(obj.get("initial_position") or []),
            )
        )
    task_type = str(data.get("task_type") or "general")
    return StandardTaskSpec(
        task_id=str(data.get("task_id") or f"dynamic_{task_type}"),
        task_type=task_type,
        goal=goal,
        robot_candidates=[str(v) for v in data.get("robot_candidates") or ["franka_fr3"]],
        objects=objects,
        held_tool_id=data.get("held_tool_id"),
        actions=actions,
        workspace=str(data.get("workspace") or "table"),
        constraints=[str(v) for v in data.get("constraints") or []],
        success_criteria=dict(data.get("success_criteria") or {"primary": "runner_success"}),
        experiment_variables={
            str(key): list(value)
            for key, value in dict(data.get("experiment_variables") or {}).items()
        },
        safety_limits=dict(data.get("safety_limits") or {}),
    )
