from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Protocol


@dataclass(frozen=True)
class SafetyCheckResult:
    ok: bool
    adapter: str
    checks: Dict[str, bool]
    warnings: list[str] = field(default_factory=list)
    blocked_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InstrumentAdapter(Protocol):
    adapter_name: str

    def validate(self, skill_plan: Mapping[str, Any]) -> SafetyCheckResult:
        ...

    def execute(self, skill_plan: Mapping[str, Any]) -> Dict[str, Any]:
        ...

    def observe(self) -> Dict[str, Any]:
        ...

    def stop(self) -> Dict[str, Any]:
        ...

    def get_safety_state(self) -> Dict[str, Any]:
        ...
