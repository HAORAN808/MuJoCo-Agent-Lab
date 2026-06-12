"""Multi-round experiment history tracking.

Stores each round's complete context (task_spec, design, runs, summary,
analysis) and provides variable space narrowing and convergence detection
for iterative experiment optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ExperimentRound:
    """A single experiment round's complete data."""
    round_id: int
    goal: str
    task_spec: Dict[str, Any]
    design: Dict[str, Any]
    runs: List[Dict[str, Any]]
    summary: Dict[str, Any]
    analysis: Dict[str, Any]
    experiment_space: Dict[str, List[str]]
    robot_id: str = ""


class ExperimentHistory:
    """Tracks multiple experiment rounds and provides iteration guidance."""

    def __init__(self):
        self.rounds: List[ExperimentRound] = []

    def add_round(
        self,
        goal: str,
        task_spec: Dict[str, Any],
        design: Dict[str, Any],
        runs: List[Dict[str, Any]],
        summary: Dict[str, Any],
        analysis: Dict[str, Any],
        experiment_space: Dict[str, List[str]],
        robot_id: str = "",
    ) -> ExperimentRound:
        """Add a completed experiment round."""
        round_data = ExperimentRound(
            round_id=len(self.rounds) + 1,
            goal=goal,
            task_spec=task_spec,
            design=design,
            runs=runs,
            summary=summary,
            analysis=analysis,
            experiment_space=experiment_space,
            robot_id=robot_id,
        )
        self.rounds.append(round_data)
        return round_data

    def get_narrowed_space(
        self,
        current_space: Dict[str, List[str]],
        min_rounds: int = 2,
    ) -> Dict[str, List[str]]:
        """Narrow variable space based on historical results.

        Removes variable values that consistently fail and keeps
        values that show mixed or positive results.
        """
        if len(self.rounds) < min_rounds:
            return current_space

        # Collect success/failure per variable value
        value_outcomes: Dict[str, Dict[str, List[bool]]] = {}
        for rnd in self.rounds:
            for run in rnd.runs:
                for key in current_space:
                    val = run.get(key)
                    if val is not None:
                        if key not in value_outcomes:
                            value_outcomes[key] = {}
                        if val not in value_outcomes[key]:
                            value_outcomes[key][val] = []
                        value_outcomes[key][val].append(bool(run.get("success", False)))

        narrowed: Dict[str, List[str]] = {}
        for key, values in current_space.items():
            if key not in value_outcomes:
                narrowed[key] = values
                continue

            kept = []
            for val in values:
                outcomes = value_outcomes[key].get(val, [])
                if not outcomes:
                    kept.append(val)  # no data, keep
                elif all(outcomes):
                    kept.append(val)  # always succeeds, keep
                elif any(outcomes):
                    kept.append(val)  # mixed, keep for contrast
                # all fail → remove

            narrowed[key] = kept if kept else values[:1]  # keep at least one

        return narrowed

    def is_converged(
        self,
        window: int = 3,
        success_rate_tol: float = 0.1,
    ) -> bool:
        """Check if the last N rounds have similar success rates."""
        if len(self.rounds) < window:
            return False

        recent = self.rounds[-window:]
        rates = [r.summary.get("success_rate", 0.0) for r in recent]

        # Check if all rates are within tolerance of the mean
        mean_rate = sum(rates) / len(rates)
        return all(abs(r - mean_rate) < success_rate_tol for r in rates)

    def get_history_summary(self) -> Dict[str, Any]:
        """Get a summary of all rounds for LLM context."""
        return {
            "total_rounds": len(self.rounds),
            "rounds": [
                {
                    "round_id": r.round_id,
                    "success_rate": r.summary.get("success_rate", 0.0),
                    "num_runs": len(r.runs),
                    "experiment_space": r.experiment_space,
                    "robot_id": r.robot_id,
                    "key_findings": [
                        f.get("title", "") for f in r.analysis.get("findings", [])[:3]
                    ],
                }
                for r in self.rounds
            ],
            "converged": self.is_converged(),
        }

    def get_analysis_context(self) -> str:
        """Generate text context for LLM analysis prompts."""
        lines = [f"Experiment history: {len(self.rounds)} rounds completed."]
        for r in self.rounds:
            rate = r.summary.get("success_rate", 0.0)
            lines.append(
                f"  Round {r.round_id}: success_rate={rate:.3f}, "
                f"runs={len(r.runs)}, space={r.experiment_space}"
            )
            for finding in r.analysis.get("findings", [])[:2]:
                lines.append(f"    Finding: {finding.get('title', '')}")
            for rec in r.analysis.get("recommendations", [])[:1]:
                lines.append(f"    Recommendation: {rec.get('body', '')}")
        if self.is_converged():
            lines.append("STATUS: Results appear converged. Consider stopping or changing approach.")
        return "\n".join(lines)
