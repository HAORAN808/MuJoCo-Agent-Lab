from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import mujoco

from .agent import run_nlp_agent_round
from .robot_registry import get_robot
from .scene_composer import compose_scene


CASES = [
    {
        "goal": "use the fr3 robot arm to press a button target",
        "expected_type": "press",
        "expected_tool": None,
    },
    {
        "goal": "use the fr3 robot arm to insert a peg into a hole",
        "expected_type": "insert",
        "expected_tool": "peg",
    },
    {
        "goal": "use the fr3 robot arm and screwdriver to work on a screw",
        "expected_type": "screwdriving",
        "expected_tool": "screwdriver",
    },
    {
        "goal": "use the fr3 robot arm with a spatula tool to push an object",
        "expected_type": "tool_use",
        "expected_tool": "spatula",
    },
]


def _has_rendered_frames(result: Dict[str, Any]) -> bool:
    trace = result.get("demo_trace") or {}
    for replay in trace.get("replays") or []:
        if replay.get("image_frames"):
            return True
    return False


def _verify_aliases() -> List[Dict[str, Any]]:
    aliases = {
        "panda": "franka_emika_panda",
        "ur5e": "universal_robots_ur5e",
        "xarm7": "ufactory_xarm7",
    }
    rows: List[Dict[str, Any]] = []
    for alias, expected in aliases.items():
        try:
            spec = get_robot(alias)
            actual = spec.robot_id
            ok = actual == expected
        except Exception as exc:
            actual = type(exc).__name__
            ok = False
        rows.append({"alias": alias, "expected": expected, "actual": actual, "ok": ok})
    return rows


def _verify_tool_attachment() -> Dict[str, Any]:
    scene = compose_scene(
        "franka_fr3",
        objects=[{"object_id": "screw_head", "position": [0.5, 0.0, 0.395]}],
        held_tool_id="screwdriver",
    )
    model = scene["model"]
    held_tool_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "held_tool")
    tool_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tool_shaft")
    return {
        "ok": held_tool_body >= 0 and tool_geom >= 0,
        "held_tool_body": held_tool_body,
        "tool_shaft_geom": tool_geom,
    }


def verify_nlp_pipeline(limit: int = 1) -> Dict[str, Any]:
    previous_force = os.environ.get("AGENT_FORCE_LOCAL")
    previous_registered = os.environ.get("NLP_USE_REGISTERED_RUNNERS")
    os.environ["AGENT_FORCE_LOCAL"] = "1"
    os.environ.pop("NLP_USE_REGISTERED_RUNNERS", None)

    rows: List[Dict[str, Any]] = []
    try:
        for case in CASES:
            result = run_nlp_agent_round(
                case["goal"],
                limit=limit,
                language="zh",
                robot_id="franka_fr3",
            )
            task_spec = result.get("task_spec") or {}
            expected_tool = case["expected_tool"]
            actual_tool = task_spec.get("held_tool_id")
            rows.append(
                {
                    "goal": case["goal"],
                    "expected_type": case["expected_type"],
                    "actual_type": task_spec.get("task_type"),
                    "expected_tool": expected_tool,
                    "actual_tool": actual_tool,
                    "num_actions": len(task_spec.get("actions") or []),
                    "num_runs": result.get("num_runs", len(result.get("runs") or [])),
                    "agent_trace": result.get("agent_trace") or [],
                    "has_rendered_frames": _has_rendered_frames(result),
                    "ok": (
                        task_spec.get("task_type") == case["expected_type"]
                        and (expected_tool is None or actual_tool == expected_tool)
                        and bool(task_spec.get("actions"))
                        and "compose_scene" in (result.get("agent_trace") or [])
                        and bool(result.get("runs"))
                        and _has_rendered_frames(result)
                    ),
                }
            )
    finally:
        if previous_force is None:
            os.environ.pop("AGENT_FORCE_LOCAL", None)
        else:
            os.environ["AGENT_FORCE_LOCAL"] = previous_force
        if previous_registered is None:
            os.environ.pop("NLP_USE_REGISTERED_RUNNERS", None)
        else:
            os.environ["NLP_USE_REGISTERED_RUNNERS"] = previous_registered

    alias_rows = _verify_aliases()
    tool_attachment = _verify_tool_attachment()
    ok = all(row["ok"] for row in rows) and all(row["ok"] for row in alias_rows) and tool_attachment["ok"]
    return {
        "ok": ok,
        "nlp_cases": rows,
        "robot_aliases": alias_rows,
        "tool_attachment": tool_attachment,
    }


def main() -> None:
    result = verify_nlp_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
