from __future__ import annotations

from typing import Any, Dict, Mapping

from .base import SafetyCheckResult


class RealRobotAdapter:
    adapter_name = "real_robot_mock"

    def validate(self, skill_plan: Mapping[str, Any]) -> SafetyCheckResult:
        safety_limits = dict(skill_plan.get("safety_limits") or {})
        skills = list(skill_plan.get("skills") or [])
        checks = {
            "has_robot": bool(skill_plan.get("robot_id")),
            "has_skills": bool(skills),
            "manual_confirmation": bool(safety_limits.get("requires_human_confirmation", True)),
            "speed_limited": float(safety_limits.get("max_speed_mps", 0.0) or 0.0) <= 0.25,
            "force_limited": float(safety_limits.get("max_contact_force_n", 20.0) or 20.0) <= 25.0,
        }
        warnings = [
            "This adapter is a safety gate and mock interface; no physical robot driver is connected."
        ]
        ok = all(checks.values())
        return SafetyCheckResult(
            ok=ok,
            adapter=self.adapter_name,
            checks=checks,
            warnings=warnings,
            blocked_reason="" if ok else "real_robot_safety_check_failed",
        )

    def execute(self, skill_plan: Mapping[str, Any]) -> Dict[str, Any]:
        validation = self.validate(skill_plan)
        return {
            "ok": False,
            "adapter": self.adapter_name,
            "validation": validation.to_dict(),
            "blocked_reason": "physical_driver_not_configured",
        }

    def observe(self) -> Dict[str, Any]:
        return {"adapter": self.adapter_name, "state": "mock_only"}

    def stop(self) -> Dict[str, Any]:
        return {"adapter": self.adapter_name, "stopped": True, "note": "mock stop only"}

    def get_safety_state(self) -> Dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "mode": "mock_real_robot",
            "human_confirmation_required": True,
            "physical_driver_connected": False,
        }
