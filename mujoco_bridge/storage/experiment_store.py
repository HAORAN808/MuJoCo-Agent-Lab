from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class ExperimentQuery:
    task_type: str = ""
    robot_id: str = ""
    object_id: str = ""
    failure_type: str = ""


class ExperimentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(record)
        payload.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        payload.setdefault("record_type", "experiment")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def list_records(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows

    def query(
        self,
        *,
        task_type: str = "",
        robot_id: str = "",
        object_id: str = "",
        failure_type: str = "",
    ) -> List[Dict[str, Any]]:
        rows = self.list_records()
        filtered = []
        for row in rows:
            task_spec = row.get("task_spec") or {}
            evaluation = row.get("evaluation_report") or {}
            objects = task_spec.get("objects") or []
            failures = evaluation.get("failure_reasons") or []
            if task_type and task_spec.get("task_type") != task_type:
                continue
            if robot_id and row.get("robot_id") != robot_id:
                continue
            if object_id and not any(obj.get("object_id") == object_id for obj in objects):
                continue
            if failure_type and not any(item.get("failure_type") == failure_type for item in failures):
                continue
            filtered.append(row)
        return filtered

    def summarize(self, records: Iterable[Mapping[str, Any]] | None = None) -> Dict[str, Any]:
        rows = list(records) if records is not None else self.list_records()
        total = len(rows)
        task_counts: Dict[str, int] = {}
        failure_counts: Dict[str, int] = {}
        best_by_task: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            task_spec = row.get("task_spec") or {}
            evaluation = row.get("evaluation_report") or {}
            task_type = str(task_spec.get("task_type") or "unknown")
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
            for failure in evaluation.get("failure_reasons") or []:
                name = str(failure.get("failure_type") or "unknown")
                failure_counts[name] = failure_counts.get(name, 0) + int(failure.get("count", 0))
            current_best = best_by_task.get(task_type)
            if current_best is None or evaluation.get("success_rate", 0.0) > current_best.get("success_rate", 0.0):
                best_by_task[task_type] = {
                    "task_id": task_spec.get("task_id"),
                    "success_rate": evaluation.get("success_rate", 0.0),
                    "recorded_at": row.get("recorded_at"),
                }
        return {
            "total_records": total,
            "task_counts": task_counts,
            "failure_counts": failure_counts,
            "best_by_task": best_by_task,
        }
