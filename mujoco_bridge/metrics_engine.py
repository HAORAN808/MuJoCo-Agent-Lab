"""Generalized metrics engine for task-agnostic evaluation.

Provides contact detection, object pose tracking, force monitoring,
and success/failure classification that works with any robot and any
task description from the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
import numpy.typing as npt


@dataclass
class ContactEvent:
    """A detected contact between two bodies."""
    step: int
    body_a: str
    body_b: str
    force_magnitude: float
    position: npt.NDArray[np.float64]


@dataclass
class PoseTrack:
    """Tracks an object's pose over time."""
    body_name: str
    body_id: int
    positions: List[npt.NDArray[np.float64]] = field(default_factory=list)
    quaternions: List[npt.NDArray[np.float64]] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)


@dataclass
class SuccessCriterion:
    """A task-independent success criterion."""
    criterion_type: str  # "object_at_target", "contact_achieved", "object_displaced", "lifted"
    target_position: Optional[Tuple[float, float, float]] = None
    tolerance: float = 0.05
    min_contact_steps: int = 0
    min_displacement: float = 0.0
    min_lift_height: float = 0.0
    min_force: float = 0.0


class MetricsEngine:
    """Task-agnostic metrics collection and evaluation.

    Works with any MuJoCo model and any robot. Collects contact data,
    tracks object poses, and evaluates success criteria defined by
    the LLM or the experiment design.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.contacts: List[ContactEvent] = []
        self.pose_tracks: Dict[str, PoseTrack] = {}
        self._step = 0

    def track_body(self, body_name: str) -> None:
        """Start tracking a body's pose."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            self.pose_tracks[body_name] = PoseTrack(
                body_name=body_name, body_id=bid
            )

    def log_step(self) -> None:
        """Log current step's data (call after mj_step)."""
        self._step += 1

        # Track poses
        for name, track in self.pose_tracks.items():
            track.positions.append(self.data.xpos[track.body_id].copy())
            track.quaternions.append(self.data.xquat[track.body_id].copy())
            track.steps.append(self._step)

        # Detect contacts
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom_{con.geom1}"
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom_{con.geom2}"

            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1) or f"body_{body1}"
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2) or f"body_{body2}"

            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            force_mag = float(np.linalg.norm(force[:3]))

            if force_mag > 0.01:  # threshold to ignore micro-contacts
                self.contacts.append(ContactEvent(
                    step=self._step,
                    body_a=body1_name,
                    body_b=body2_name,
                    force_magnitude=force_mag,
                    position=self.data.contact_pos[i].copy(),
                ))

    def evaluate(
        self,
        criteria: List[SuccessCriterion],
    ) -> Dict[str, Any]:
        """Evaluate success criteria against collected data."""
        results = []
        overall_success = True

        for criterion in criteria:
            result = self._evaluate_single(criterion)
            results.append(result)
            if not result["met"]:
                overall_success = False

        return {
            "overall_success": overall_success,
            "criteria_results": results,
            "contact_summary": self._contact_summary(),
            "pose_summary": self._pose_summary(),
        }

    def _evaluate_single(self, criterion: SuccessCriterion) -> Dict[str, Any]:
        """Evaluate a single success criterion."""
        ctype = criterion.criterion_type

        if ctype == "object_at_target":
            return self._check_at_target(criterion)
        elif ctype == "contact_achieved":
            return self._check_contact(criterion)
        elif ctype == "object_displaced":
            return self._check_displacement(criterion)
        elif ctype == "lifted":
            return self._check_lifted(criterion)
        else:
            return {"criterion": ctype, "met": False, "reason": f"Unknown criterion type: {ctype}"}

    def _check_at_target(self, criterion: SuccessCriterion) -> Dict[str, Any]:
        """Check if any tracked object is at the target position."""
        if not criterion.target_position:
            return {"criterion": "object_at_target", "met": False, "reason": "No target position specified"}

        target = np.array(criterion.target_position)
        best_dist = float("inf")
        best_body = ""

        for name, track in self.pose_tracks.items():
            if track.positions:
                final_pos = track.positions[-1]
                dist = float(np.linalg.norm(final_pos - target))
                if dist < best_dist:
                    best_dist = dist
                    best_body = name

        met = best_dist <= criterion.tolerance
        return {
            "criterion": "object_at_target",
            "met": met,
            "distance": best_dist,
            "tolerance": criterion.tolerance,
            "body": best_body,
        }

    def _check_contact(self, criterion: SuccessCriterion) -> Dict[str, Any]:
        """Check if sufficient contact was achieved."""
        contact_steps = len(set(c.step for c in self.contacts))
        met = contact_steps >= criterion.min_contact_steps
        return {
            "criterion": "contact_achieved",
            "met": met,
            "contact_steps": contact_steps,
            "required": criterion.min_contact_steps,
        }

    def _check_displacement(self, criterion: SuccessCriterion) -> Dict[str, Any]:
        """Check if any object was displaced sufficiently."""
        max_disp = 0.0
        best_body = ""

        for name, track in self.pose_tracks.items():
            if len(track.positions) >= 2:
                disp = float(np.linalg.norm(track.positions[-1] - track.positions[0]))
                if disp > max_disp:
                    max_disp = disp
                    best_body = name

        met = max_disp >= criterion.min_displacement
        return {
            "criterion": "object_displaced",
            "met": met,
            "displacement": max_disp,
            "required": criterion.min_displacement,
            "body": best_body,
        }

    def _check_lifted(self, criterion: SuccessCriterion) -> Dict[str, Any]:
        """Check if any object was lifted sufficiently."""
        max_lift = 0.0
        best_body = ""

        for name, track in self.pose_tracks.items():
            if len(track.positions) >= 2:
                lift = float(track.positions[-1][2] - track.positions[0][2])
                if lift > max_lift:
                    max_lift = lift
                    best_body = name

        met = max_lift >= criterion.min_lift_height
        return {
            "criterion": "lifted",
            "met": met,
            "lift_height": max_lift,
            "required": criterion.min_lift_height,
            "body": best_body,
        }

    def _contact_summary(self) -> Dict[str, Any]:
        """Summarize all contacts."""
        if not self.contacts:
            return {"total_events": 0, "contact_steps": 0, "max_force": 0.0}

        forces = [c.force_magnitude for c in self.contacts]
        steps = set(c.step for c in self.contacts)

        # Most common body pair
        pairs: Dict[Tuple[str, str], int] = {}
        for c in self.contacts:
            pair = tuple(sorted([c.body_a, c.body_b]))
            pairs[pair] = pairs.get(pair, 0) + 1

        top_pair = max(pairs, key=pairs.get) if pairs else ("", "")

        return {
            "total_events": len(self.contacts),
            "contact_steps": len(steps),
            "max_force": max(forces),
            "mean_force": sum(forces) / len(forces),
            "top_body_pair": list(top_pair),
            "top_pair_count": pairs.get(top_pair, 0),
        }

    def _pose_summary(self) -> Dict[str, Any]:
        """Summarize tracked poses."""
        summaries = {}
        for name, track in self.pose_tracks.items():
            if track.positions:
                initial = track.positions[0]
                final = track.positions[-1]
                displacement = float(np.linalg.norm(final - initial))
                lift = float(final[2] - initial[2])
                summaries[name] = {
                    "initial_pos": initial.tolist(),
                    "final_pos": final.tolist(),
                    "displacement": displacement,
                    "lift_height": lift,
                }
        return summaries


def create_success_criteria_from_spec(
    spec: Dict[str, Any],
) -> List[SuccessCriterion]:
    """Create SuccessCriterion objects from LLM-parsed task spec."""
    criteria = []
    sc = spec.get("success_criteria", {})
    primary = sc.get("primary", "object_at_target")

    if primary == "object_at_target":
        criteria.append(SuccessCriterion(
            criterion_type="object_at_target",
            target_position=tuple(sc["target_position"]) if "target_position" in sc else None,
            tolerance=sc.get("tolerance", 0.05),
        ))
    elif primary == "contact_achieved":
        criteria.append(SuccessCriterion(
            criterion_type="contact_achieved",
            min_contact_steps=sc.get("min_contact_steps", 10),
        ))
    elif primary == "object_displaced":
        criteria.append(SuccessCriterion(
            criterion_type="object_displaced",
            min_displacement=sc.get("min_displacement", 0.01),
        ))
    elif primary == "lifted":
        criteria.append(SuccessCriterion(
            criterion_type="lifted",
            min_lift_height=sc.get("min_lift_height", 0.04),
        ))

    # Always add contact as secondary criterion
    if primary != "contact_achieved":
        criteria.append(SuccessCriterion(
            criterion_type="contact_achieved",
            min_contact_steps=sc.get("min_contact_steps", 5),
        ))

    return criteria
