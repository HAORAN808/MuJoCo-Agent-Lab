from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class SkillFailure:
    failure_type: str
    description: str
    retry_hint: str


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    input_params: List[str]
    preconditions: List[str]
    expected_observations: List[str]
    success_criteria: List[str]
    failure_modes: List[SkillFailure]
    tunable_params: Dict[str, List[Any]] = field(default_factory=dict)
    implemented_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillResult:
    skill_name: str
    success: bool
    failure_type: str = "none"
    metrics: Dict[str, Any] = field(default_factory=dict)
    trace_ref: str = ""
