from __future__ import annotations

import json
from typing import Any, Dict

from .arm_runner import run_default_arm_skill_suite
from .smoke_all_tasks import DEFAULT_TASK_IDS, run_smoke_suite
from .tasks import get_task, list_task_specs


def _verify_robot_arm_replays() -> Dict[str, Any]:
    rows = []
    for spec in list_task_specs():
        if not str(spec.get("execution_kind", "")).startswith("robot_arm"):
            continue
        task = get_task(spec["task_id"])
        trace = task.demo_trace(use_fallback=False)
        image_frames = trace.get("image_frames") or []
        abstract_frames = trace.get("frames") or []
        ok = bool(image_frames) and not bool(abstract_frames)
        rows.append(
            {
                "task_id": spec["task_id"],
                "ok": ok,
                "source": trace.get("source"),
                "image_frames": len(image_frames),
                "has_abstract_frames": bool(abstract_frames),
                "first_frame": image_frames[0] if image_frames else "",
            }
        )
    return {"ok": all(row["ok"] for row in rows), "tasks": rows}


def verify_all_runners() -> Dict[str, Any]:
    task_result = run_smoke_suite(
        task_ids=DEFAULT_TASK_IDS,
        limit=1,
        use_fallback=False,
        keep_going=True,
    )
    arm_result = run_default_arm_skill_suite()
    replay_result = _verify_robot_arm_replays()
    compact_arm = {
        "ok": arm_result["ok"],
        "robot": arm_result["robot"],
        "source": arm_result["source"],
        "skills": [
            {
                "skill_id": row["skill_id"],
                "object_id": row["object_id"],
                "success": row["success"],
                "failure_type": row["failure_type"],
                "contact_steps": row["contact_steps"],
                "tool_contact_steps": row.get("tool_contact_steps", 0),
                "max_touch_force": row["max_touch_force"],
                "object_displacement": row["object_displacement"],
                "lifted_height": row["lifted_height"],
                "trace_frames": len(row.get("trace", [])),
            }
            for row in arm_result["skills"]
        ],
    }
    return {
        "ok": bool(task_result["ok"] and arm_result["ok"] and replay_result["ok"]),
        "task_runners": task_result,
        "arm_skill_runners": compact_arm,
        "robot_arm_replays": replay_result,
    }


def main() -> None:
    result = verify_all_runners()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
