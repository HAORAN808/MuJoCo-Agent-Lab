from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List

from .tasks import get_task, list_task_specs
from .tasks.base import summarize_runs


DEFAULT_TASK_IDS = [task["task_id"] for task in list_task_specs()]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def run_task_smoke(task_id: str, limit: int, use_fallback: bool) -> Dict[str, Any]:
    task = get_task(task_id)
    started = time.perf_counter()
    result = task.run_experiments(limit=limit, use_fallback=use_fallback)
    elapsed_s = round(time.perf_counter() - started, 3)
    runs = result.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise RuntimeError(f"{task_id} returned no runs")
    first = runs[0]
    required_fields = {"run_id", "success", "failure_type"}
    missing = sorted(field for field in required_fields if field not in first)
    if missing:
        raise RuntimeError(f"{task_id} first run missing required fields: {missing}")
    return {
        "task_id": task_id,
        "ok": True,
        "source": result.get("source"),
        "elapsed_s": elapsed_s,
        "num_runs": len(runs),
        "summary": summarize_runs(runs),
        "first_run": _jsonable(first),
    }


def run_smoke_suite(
    task_ids: Iterable[str],
    limit: int,
    use_fallback: bool,
    keep_going: bool,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for task_id in task_ids:
        try:
            rows.append(run_task_smoke(task_id, limit, use_fallback))
        except Exception as exc:
            row = {
                "task_id": task_id,
                "ok": False,
                "error": str(exc),
            }
            rows.append(row)
            if not keep_going:
                break
    return {
        "ok": all(row.get("ok") for row in rows),
        "mode": "fallback" if use_fallback else "mujoco",
        "limit": limit,
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-shot smoke checks for registered MuJoCo tasks.")
    parser.add_argument(
        "--task",
        dest="task_ids",
        action="append",
        help="Task id to check. May be passed multiple times. Defaults to all registered tasks.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Runs per task. Defaults to 1.")
    parser.add_argument("--fallback", action="store_true", help="Use fallback mode where a task supports it.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a task fails.")
    args = parser.parse_args()

    task_ids = args.task_ids or DEFAULT_TASK_IDS
    result = run_smoke_suite(
        task_ids=task_ids,
        limit=max(1, args.limit),
        use_fallback=args.fallback,
        keep_going=args.keep_going,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
