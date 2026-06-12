from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    description: str
    keywords: List[str]
    experiment_space: Dict[str, List[str]]
    metrics: List[str]
    failure_types: List[str]
    supported_objects: List[str] = field(default_factory=list)
    execution_kind: str = "task_specific_simulation"
    manipulation_actor: str = "task-specific scripted actuator"
    fidelity_notes: List[str] = field(default_factory=list)
    runner_status: str = "implemented"
    runner_module: str = ""
    verified_by: List[str] = field(default_factory=lambda: ["python -m mujoco_bridge.smoke_all_tasks --task <task_id>"])


class ExperimentTask(Protocol):
    spec: TaskSpec

    def build_matrix(
        self,
        limit: int,
        experiment_space: Mapping[str, Sequence[str]] | None = None,
    ) -> List[Any]:
        ...

    def run_experiments(
        self,
        limit: int,
        use_fallback: bool = False,
        experiment_space: Mapping[str, Sequence[str]] | None = None,
    ) -> Dict[str, Any]:
        ...

    def demo_trace(self, use_fallback: bool = False) -> Dict[str, Any]:
        ...


def summarize_runs(runs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(runs)
    success_count = sum(1 for row in rows if row.get("success"))
    failures = [row for row in rows if not row.get("success")]
    distribution: Dict[str, int] = {}
    for row in failures:
        failure_type = str(row.get("failure_type", "unknown"))
        distribution[failure_type] = distribution.get(failure_type, 0) + 1
    return {
        "num_runs": len(rows),
        "success_rate": success_count / max(1, len(rows)),
        "failure_distribution": {
            key: value / max(1, len(rows)) for key, value in distribution.items()
        },
    }


def supported_space(
    default_space: Mapping[str, Sequence[str]],
    requested_space: Mapping[str, Sequence[str]] | None,
) -> Dict[str, List[str]]:
    space: Dict[str, List[str]] = {}
    for key, defaults in default_space.items():
        supported = {str(v) for v in defaults}
        requested = requested_space.get(key) if requested_space else None
        if requested:
            values = [str(v) for v in requested if str(v) in supported]
            space[key] = values or list(defaults)
        else:
            space[key] = list(defaults)
    return space
