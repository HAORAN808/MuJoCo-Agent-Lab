from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .agent import run_agent_round
from .tasks import get_task


GOALS = [
    {
        "goal": "研究机械臂拧螺丝时下压力和对准误差对成功率的影响",
        "expected_task": "screwdriving",
    },
    {
        "goal": "让机械臂把 peg 插入孔里并分析装配失败",
        "expected_task": "assembly_insertion",
    },
    {
        "goal": "用机械臂拿工具推动物体",
        "expected_task": "tool_use",
    },
    {
        "goal": "研究机械臂对不同物体进行抓取、接触和移动的基础能力",
        "expected_task": "fr3_arm_primitives",
    },
    {
        "goal": "让机械臂按下按钮并研究接触力是否足够",
        "expected_task": "fr3_arm_primitives",
    },
]


def verify_agent_loop(limit: int = 9) -> Dict[str, Any]:
    previous = os.environ.get("AGENT_FORCE_LOCAL")
    os.environ["AGENT_FORCE_LOCAL"] = "1"
    rows: List[Dict[str, Any]] = []
    try:
        for item in GOALS:
            result = run_agent_round(item["goal"], limit=limit, language="zh")
            task = get_task(result["task_id"])
            trace = task.demo_trace(use_fallback=False)
            image_frames = trace.get("image_frames") or []
            ok = (
                result["task_id"] == item["expected_task"]
                and result["runs"]
                and result["summary"]["num_runs"] > 0
                and result["analysis"]["findings"]
                and task.spec.execution_kind.startswith("robot_arm")
                and bool(image_frames)
                and not bool(trace.get("frames"))
            )
            rows.append(
                {
                    "goal": item["goal"],
                    "expected_task": item["expected_task"],
                    "task_id": result["task_id"],
                    "ok": ok,
                    "provider": result["agent_provider"],
                    "source": result["source"],
                    "num_runs": result["summary"]["num_runs"],
                    "success_rate": result["summary"]["success_rate"],
                    "execution_kind": task.spec.execution_kind,
                    "image_frames": len(image_frames),
                    "api_calls": result["model_api_calls"]["count"],
                }
            )
    finally:
        if previous is None:
            os.environ.pop("AGENT_FORCE_LOCAL", None)
        else:
            os.environ["AGENT_FORCE_LOCAL"] = previous
    return {"ok": all(row["ok"] for row in rows), "agent_loops": rows}


def main() -> None:
    result = verify_agent_loop()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
