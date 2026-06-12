from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from .protocols import EvaluationReport, evaluation_from_runs


def build_evaluation_report(task_id: str, runs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    return evaluation_from_runs(task_id, runs).to_dict()


def report_to_markdown(report: EvaluationReport | Mapping[str, Any]) -> str:
    data = report.to_dict() if isinstance(report, EvaluationReport) else dict(report)
    lines = [
        f"# Evaluation Report: {data.get('task_id', 'unknown')}",
        "",
        f"- Success: {data.get('success')}",
        f"- Sample count: {data.get('sample_count')}",
        f"- Success count: {data.get('success_count')}",
        f"- Failure count: {data.get('failure_count')}",
        f"- Success rate: {float(data.get('success_rate', 0.0)):.3f}",
        "",
        "## Failure Reasons",
    ]
    failures = data.get("failure_reasons") or []
    if not failures:
        lines.append("- none")
    else:
        for failure in failures:
            lines.append(
                f"- {failure.get('failure_type')}: {failure.get('count')} "
                f"({float(failure.get('rate', 0.0)):.3f})"
            )
    lines.extend(["", "## Confidence Notes"])
    for note in data.get("confidence_notes") or []:
        lines.append(f"- {note}")
    lines.extend(["", "## Next Actions"])
    for action in data.get("next_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"
