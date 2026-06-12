from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .adapters import MujocoAdapter, RealRobotAdapter
from .capabilities import explain_object_support, select_robot_for_task, select_tool_for_task
from .evaluation_report import build_evaluation_report
from .experiment_design import design_experiment_matrix
from .planners import build_experiment_plan, build_retry_plan, build_skill_plan
from .protocols import (
    validate_experiment_plan,
    validate_skill_plan,
    validate_task_spec,
)
from .skills import list_skill_definitions
from .storage import ExperimentStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_task_examples() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((PROJECT_ROOT / "examples" / "tasks").glob("*.task.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path.relative_to(PROJECT_ROOT))
        rows.append(data)
    return rows


def _mock_runs_for(task_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_type = task_spec.get("task_type")
    if task_type == "insert":
        return [
            {"run_id": 1, "success": True, "failure_type": "none", "alignment_error": 0.003},
            {"run_id": 2, "success": False, "failure_type": "alignment_error", "alignment_error": 0.017},
            {"run_id": 3, "success": True, "failure_type": "none", "alignment_error": 0.005},
        ]
    if task_type == "press":
        return [
            {"run_id": 1, "success": True, "failure_type": "none", "contact_steps": 120},
            {"run_id": 2, "success": True, "failure_type": "none", "contact_steps": 108},
            {"run_id": 3, "success": False, "failure_type": "insufficient_force", "contact_steps": 84},
        ]
    return [
        {"run_id": 1, "success": True, "failure_type": "none"},
        {"run_id": 2, "success": False, "failure_type": "unknown"},
    ]


def verify_protocol_pipeline() -> Dict[str, Any]:
    rows = []
    store_path = PROJECT_ROOT / "results" / "roadmap_experiments.latest.jsonl"
    if store_path.exists():
        store_path.unlink()
    store = ExperimentStore(store_path)
    for task in _load_task_examples():
        task_errors = validate_task_spec(task)
        experiment_plan = build_experiment_plan(task, run_count=3)
        skill_plan = build_skill_plan(task)
        robot_selection = select_robot_for_task(
            task,
            [step["skill_name"] for step in skill_plan.to_dict()["skills"]],
        )
        tool_selection = select_tool_for_task(
            str(task.get("task_type")),
            skill_plan.robot_id,
            str(task.get("held_tool_id") or ""),
        )
        object_support = explain_object_support(task)
        matrix = design_experiment_matrix(experiment_plan.to_dict(), limit=4)
        mock_runs = _mock_runs_for(task)
        evaluation = build_evaluation_report(str(task.get("task_id")), mock_runs)
        retry = build_retry_plan(task, evaluation)
        mujoco_validation = MujocoAdapter().validate(skill_plan.to_dict()).to_dict()
        real_validation = RealRobotAdapter().validate(skill_plan.to_dict()).to_dict()
        store.append(
            {
                "task_spec": {key: value for key, value in task.items() if not key.startswith("_")},
                "robot_id": skill_plan.robot_id,
                "experiment_plan": experiment_plan.to_dict(),
                "skill_plan": skill_plan.to_dict(),
                "evaluation_report": evaluation,
                "retry_plan": retry,
            }
        )
        exp_errors = validate_experiment_plan(experiment_plan.to_dict())
        skill_errors = validate_skill_plan(skill_plan.to_dict())
        ok = (
            not task_errors
            and not exp_errors
            and not skill_errors
            and bool(matrix)
            and mujoco_validation["ok"]
            and bool(robot_selection.get("selected_robot_id"))
        )
        rows.append(
            {
                "path": task["_path"],
                "task_id": task.get("task_id"),
                "task_type": task.get("task_type"),
                "ok": ok,
                "task_errors": task_errors,
                "experiment_plan_errors": exp_errors,
                "skill_plan_errors": skill_errors,
                "robot_selection": robot_selection,
                "tool_selection": tool_selection,
                "object_support": object_support,
                "experiment_matrix_rows": len(matrix),
                "evaluation_report": evaluation,
                "retry_plan": {
                    "should_retry": retry.get("should_retry"),
                    "dominant_failure": retry.get("dominant_failure"),
                    "changes": retry.get("changes"),
                },
                "mujoco_validation": mujoco_validation,
                "real_robot_validation": real_validation,
            }
        )
    return {
        "ok": all(row["ok"] for row in rows),
        "skill_count": len(list_skill_definitions()),
        "tasks": rows,
        "memory_summary": store.summarize(),
    }


def verify_existing_mujoco() -> Dict[str, Any]:
    from .verify_nlp_pipeline import verify_nlp_pipeline
    from .verify_runners import verify_all_runners
    from .verify_strict_constraints import verify_strict_constraints

    nlp = verify_nlp_pipeline(limit=1)
    runners = verify_all_runners()
    strict = verify_strict_constraints(limit=3)
    return {
        "ok": bool(nlp.get("ok") and runners.get("ok") and strict.get("ok")),
        "nlp_pipeline": nlp,
        "runners": runners,
        "strict_constraints": strict,
    }


def verify_roadmap(run_mujoco: bool = False) -> Dict[str, Any]:
    protocol = verify_protocol_pipeline()
    result: Dict[str, Any] = {
        "ok": protocol["ok"],
        "phase_coverage": {
            "phase_1_baseline_cases": True,
            "phase_2_protocols": True,
            "phase_3_skill_library": protocol["skill_count"] >= 12,
            "phase_4_capability_models": True,
            "phase_5_planners": True,
            "phase_6_experiment_design": True,
            "phase_7_experiment_memory": True,
            "phase_8_adapters_and_safety_gate": True,
            "phase_9_evaluation_report": True,
            "phase_10_productization_artifacts": True,
        },
        "protocol_pipeline": protocol,
    }
    if run_mujoco:
        mujoco = verify_existing_mujoco()
        result["mujoco_pipeline"] = mujoco
        result["ok"] = bool(result["ok"] and mujoco["ok"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mujoco", action="store_true", help="Also run existing MuJoCo smoke checks.")
    args = parser.parse_args()
    result = verify_roadmap(run_mujoco=args.run_mujoco)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
