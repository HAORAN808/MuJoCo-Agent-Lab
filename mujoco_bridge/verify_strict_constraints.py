from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .agent import run_nlp_agent_round


PUSH_5CM_GOAL = "\u7528 fr3 \u673a\u68b0\u81c2\u63a8\u52a8\u65b9\u5757\u79fb\u52a85cm\uff0c\u4e0d\u80fd\u9677\u8fdb\u684c\u5b50"


def verify_strict_constraints(limit: int = 3) -> Dict[str, Any]:
    previous_force = os.environ.get("AGENT_FORCE_LOCAL")
    os.environ["AGENT_FORCE_LOCAL"] = "1"
    try:
        result = run_nlp_agent_round(
            PUSH_5CM_GOAL,
            limit=limit,
            language="zh",
            robot_id="franka_fr3",
        )
    finally:
        if previous_force is None:
            os.environ.pop("AGENT_FORCE_LOCAL", None)
        else:
            os.environ["AGENT_FORCE_LOCAL"] = previous_force

    task_spec = result.get("task_spec") or {}
    criteria = task_spec.get("success_criteria") or {}
    target = float(criteria.get("target_displacement_m") or 0.0)
    tolerance = float(criteria.get("tolerance") or 0.0)
    rows: List[Dict[str, Any]] = []
    for run in result.get("runs") or []:
        displacement = float(run.get("object_displacement") or 0.0)
        error = abs(displacement - target)
        success = bool(run.get("success"))
        table_clearance_ok = bool(run.get("table_clearance_ok"))
        rows.append(
            {
                "run_id": run.get("run_id"),
                "success": success,
                "failure_type": run.get("failure_type"),
                "target_displacement_m": target,
                "object_displacement": displacement,
                "displacement_error_m": error,
                "within_tolerance": error <= tolerance,
                "table_clearance_ok": table_clearance_ok,
                "ok": (not success) or (error <= tolerance and table_clearance_ok),
            }
        )

    return {
        "ok": (
            task_spec.get("task_type") == "push"
            and abs(target - 0.05) < 1e-9
            and all(row["ok"] for row in rows)
        ),
        "goal": PUSH_5CM_GOAL,
        "task_type": task_spec.get("task_type"),
        "success_criteria": criteria,
        "summary": result.get("summary"),
        "runs": rows,
    }


def main() -> None:
    result = verify_strict_constraints()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
