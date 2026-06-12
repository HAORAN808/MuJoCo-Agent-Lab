from __future__ import annotations

import json

from .arm_runner import run_default_arm_skill_suite


def main() -> None:
    result = run_default_arm_skill_suite()
    compact = {
        "ok": result["ok"],
        "robot": result["robot"],
        "source": result["source"],
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
            for row in result["skills"]
        ],
        "note": result["note"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
