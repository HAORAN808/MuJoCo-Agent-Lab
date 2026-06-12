from __future__ import annotations

from typing import Any, Dict, Mapping

from .base import SafetyCheckResult


class MujocoAdapter:
    adapter_name = "mujoco"

    def validate(self, skill_plan: Mapping[str, Any]) -> SafetyCheckResult:
        skills = list(skill_plan.get("skills") or [])
        checks = {
            "has_robot": bool(skill_plan.get("robot_id")),
            "has_skills": bool(skills),
            "all_skills_named": all(bool(step.get("skill_name")) for step in skills),
            "simulated_execution": True,
        }
        ok = all(checks.values())
        return SafetyCheckResult(
            ok=ok,
            adapter=self.adapter_name,
            checks=checks,
            warnings=[] if ok else ["SkillPlan is incomplete for simulated execution."],
            blocked_reason="" if ok else "invalid_skill_plan",
        )

    def execute(self, skill_plan: Mapping[str, Any]) -> Dict[str, Any]:
        validation = self.validate(skill_plan)
        if not validation.ok:
            return {"ok": False, "validation": validation.to_dict(), "runs": []}
        return {
            "ok": True,
            "adapter": self.adapter_name,
            "validation": validation.to_dict(),
            "note": "Execution is delegated to existing MuJoCo runners in this baseline adapter.",
            "skill_count": len(skill_plan.get("skills") or []),
        }

    def observe(self) -> Dict[str, Any]:
        return {"adapter": self.adapter_name, "state": "ready"}

    def stop(self) -> Dict[str, Any]:
        return {"adapter": self.adapter_name, "stopped": True}

    def get_safety_state(self) -> Dict[str, Any]:
        return {"adapter": self.adapter_name, "mode": "simulation", "human_confirmation_required": False}
