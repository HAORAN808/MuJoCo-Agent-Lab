"""Robot-agnostic task-space motion primitives.

Provides composable motion commands (reach, grasp, lift, place, push, insert)
that work with any robot described by a ``RobotSpec``, using Jacobian IK for
trajectory generation and MuJoCo physics for execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
import numpy.typing as npt

from .ik_solver import IKResult, JacobianIKSolver
from .robot_registry import RobotSpec

TABLE_TOP_Z = 0.39
EE_TABLE_CLEARANCE_M = 0.006


# ---------------------------------------------------------------------------
# Gripper controller
# ---------------------------------------------------------------------------

class GripperController:
    """Abstract gripper control, adapts per robot gripper type.

    Supports:
    - Tendon-based grippers (Franka Hand, xArm7): ctrl value on tendon actuator
    - Joint-based grippers: ctrl values on individual joint actuators
    - No gripper: no-op
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, spec: RobotSpec):
        self.model = model
        self.data = data
        self.spec = spec
        self._actuator_ids: List[int] = []
        self._ctrl_indices: List[int] = []

        if not spec.has_gripper:
            return

        for aname in spec.gripper_actuator_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                self._actuator_ids.append(aid)
                self._ctrl_indices.append(aid)

        # If no gripper actuators found by name, try to find by tendon
        if not self._actuator_ids and spec.gripper_type == "parallel":
            # Search for actuator on "split" tendon
            for i in range(model.nu):
                if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
                    tendon_id = int(model.actuator_trnid[i, 0])
                    tname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id)
                    if tname == "split":
                        self._actuator_ids.append(i)
                        self._ctrl_indices.append(i)

    def open(self) -> None:
        """Open the gripper."""
        if not self.spec.has_gripper:
            return
        ctrl_val = self.spec.gripper_ctrl_open
        for ci in self._ctrl_indices:
            self.data.ctrl[ci] = ctrl_val

    def close(self, force_scale: float = 1.0) -> None:
        """Close the gripper.

        Parameters
        ----------
        force_scale : float
            0.0 = fully open, 1.0 = fully closed.
        """
        if not self.spec.has_gripper:
            return
        open_val = self.spec.gripper_ctrl_open
        closed_val = self.spec.gripper_ctrl_closed
        ctrl_val = open_val + force_scale * (closed_val - open_val)
        for ci in self._ctrl_indices:
            self.data.ctrl[ci] = ctrl_val

    @property
    def is_available(self) -> bool:
        return self.spec.has_gripper and len(self._ctrl_indices) > 0


# ---------------------------------------------------------------------------
# Contact logger
# ---------------------------------------------------------------------------

@dataclass
class ContactInfo:
    """Per-step contact information."""
    step: int
    contact_count: int
    max_force: float
    ee_pos: npt.NDArray[np.float64]


class ContactLogger:
    """Logs contact information during motion execution."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_a_prefix: str = "",
        body_b_name: str = "",
    ):
        self.model = model
        self.data = data
        self.body_a_prefix = body_a_prefix
        self.body_b_name = body_b_name
        self.log: List[ContactInfo] = []
        self.total_contact_steps = 0
        self.max_touch_force = 0.0

    def log_step(self, step: int, ee_pos: npt.NDArray[np.float64]) -> None:
        count, max_f = self._count_contacts()
        if count > 0:
            self.total_contact_steps += 1
        self.max_touch_force = max(self.max_touch_force, max_f)
        self.log.append(ContactInfo(step=step, contact_count=count, max_force=max_f, ee_pos=ee_pos.copy()))

    def _count_contacts(self) -> Tuple[int, float]:
        data = self.data
        model = self.model
        count = 0
        max_f = 0.0

        for i in range(data.ncon):
            con = data.contact[i]
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""

            match = False
            if self.body_a_prefix and self.body_b_name:
                if (geom1_name.startswith(self.body_a_prefix) and self.body_b_name in geom2_name) or \
                   (geom2_name.startswith(self.body_a_prefix) and self.body_b_name in geom1_name):
                    match = True

            if match:
                count += 1
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                max_f = max(max_f, float(np.linalg.norm(force[:3])))

        return count, max_f


# ---------------------------------------------------------------------------
# Motion trace
# ---------------------------------------------------------------------------

@dataclass
class TraceFrame:
    """Single frame in a motion trace."""
    step: int
    qpos: npt.NDArray[np.float64]
    ee_pos: npt.NDArray[np.float64]
    object_pos: Optional[npt.NDArray[np.float64]] = None
    contact_count: int = 0
    touch_force: float = 0.0


@dataclass
class MotionTrace:
    """Full trace of a motion execution."""
    frames: List[TraceFrame] = field(default_factory=list)
    ik_results: List[IKResult] = field(default_factory=list)
    success: bool = False
    failure_type: str = ""
    final_ee_pos: Optional[npt.NDArray[np.float64]] = None
    final_object_pos: Optional[npt.NDArray[np.float64]] = None
    object_displacement: float = 0.0
    lifted_height: float = 0.0
    contact_steps: int = 0
    max_touch_force: float = 0.0
    min_ee_z: float = 999.0
    table_clearance_ok: bool = True
    table_penetration_steps: int = 0
    table_contact_steps: int = 0
    table_contact_pairs: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Motion primitives
# ---------------------------------------------------------------------------

class MotionPrimitiveType(str, Enum):
    REACH = "reach"
    GRASP = "grasp"
    LIFT = "lift"
    PLACE = "place"
    PUSH = "push"
    INSERT = "insert"
    RELEASE = "release"
    WAIT = "wait"


@dataclass
class MotionPrimitive:
    """A single motion command."""
    primitive_type: MotionPrimitiveType
    target_pos: Optional[Tuple[float, float, float]] = None
    target_quat: Optional[Tuple[float, float, float, float]] = None
    height: float = 0.0
    direction: Optional[Tuple[float, float, float]] = None
    distance: float = 0.0
    force_scale: float = 1.0
    steps: int = 300
    approach_axis: Optional[Tuple[float, float, float]] = None

    @staticmethod
    def reach(target_pos: Sequence[float], target_quat: Optional[Sequence[float]] = None, steps: int = 300) -> MotionPrimitive:
        return MotionPrimitive(
            primitive_type=MotionPrimitiveType.REACH,
            target_pos=tuple(target_pos),
            target_quat=tuple(target_quat) if target_quat is not None else None,
            steps=steps,
        )

    @staticmethod
    def grasp(force_scale: float = 1.0, steps: int = 150) -> MotionPrimitive:
        return MotionPrimitive(
            primitive_type=MotionPrimitiveType.GRASP,
            force_scale=force_scale,
            steps=steps,
        )

    @staticmethod
    def lift(height: float = 0.08, steps: int = 300) -> MotionPrimitive:
        return MotionPrimitive(
            primitive_type=MotionPrimitiveType.LIFT,
            height=height,
            steps=steps,
        )

    @staticmethod
    def place(target_pos: Sequence[float], steps: int = 300) -> MotionPrimitive:
        return MotionPrimitive(
            primitive_type=MotionPrimitiveType.PLACE,
            target_pos=tuple(target_pos),
            steps=steps,
        )

    @staticmethod
    def push(direction: Sequence[float], distance: float = 0.05, steps: int = 300) -> MotionPrimitive:
        return MotionPrimitive(
            primitive_type=MotionPrimitiveType.PUSH,
            direction=tuple(direction),
            distance=distance,
            steps=steps,
        )

    @staticmethod
    def insert(target_pos: Sequence[float], approach_axis: Optional[Sequence[float]] = None, steps: int = 400) -> MotionPrimitive:
        return MotionPrimitive(
            primitive_type=MotionPrimitiveType.INSERT,
            target_pos=tuple(target_pos),
            approach_axis=tuple(approach_axis) if approach_axis else (0.0, 0.0, -1.0),
            steps=steps,
        )

    @staticmethod
    def release(steps: int = 100) -> MotionPrimitive:
        return MotionPrimitive(
            primitive_type=MotionPrimitiveType.RELEASE,
            steps=steps,
        )

    @staticmethod
    def wait(steps: int = 50) -> MotionPrimitive:
        return MotionPrimitive(
            primitive_type=MotionPrimitiveType.WAIT,
            steps=steps,
        )


# ---------------------------------------------------------------------------
# Motion plan
# ---------------------------------------------------------------------------

@dataclass
class MotionPlan:
    """A sequence of motion primitives forming a complete task."""
    primitives: List[MotionPrimitive] = field(default_factory=list)

    def add(self, primitive: MotionPrimitive) -> MotionPlan:
        self.primitives.append(primitive)
        return self

    @staticmethod
    def pick_and_place(
        pre_grasp_pos: Sequence[float],
        grasp_pos: Sequence[float],
        place_pos: Sequence[float],
        lift_height: float = 0.08,
    ) -> MotionPlan:
        """Standard pick-and-place motion plan."""
        plan = MotionPlan()
        plan.add(MotionPrimitive.reach(pre_grasp_pos))
        plan.add(MotionPrimitive.reach(grasp_pos))
        plan.add(MotionPrimitive.grasp())
        plan.add(MotionPrimitive.lift(lift_height))
        plan.add(MotionPrimitive.reach(place_pos))
        plan.add(MotionPrimitive.release())
        plan.add(MotionPrimitive.wait(50))
        return plan

    @staticmethod
    def from_action_sequence(actions: List[Dict[str, Any]]) -> MotionPlan:
        """Build a plan from LLM-parsed action descriptions."""
        plan = MotionPlan()
        for action in actions:
            atype = action.get("type", "")
            if atype == "reach":
                target = action.get("target_pos", action.get("target", [0.5, 0.0, 0.4]))
                if isinstance(target, str):
                    target = [0.5, 0.0, 0.4]  # fallback
                plan.add(MotionPrimitive.reach(target))
            elif atype == "grasp":
                force = action.get("force_scale", action.get("force", 1.0))
                if isinstance(force, str):
                    force = {"low": 0.5, "medium": 0.8, "high": 1.0}.get(force, 1.0)
                plan.add(MotionPrimitive.grasp(force))
            elif atype == "lift":
                height = action.get("height", 0.08)
                plan.add(MotionPrimitive.lift(float(height)))
            elif atype == "place":
                target = action.get("target_pos", action.get("target", [0.5, 0.0, 0.4]))
                if isinstance(target, str):
                    target = [0.5, 0.0, 0.4]
                plan.add(MotionPrimitive.place(target))
            elif atype == "push":
                direction = action.get("direction", [1.0, 0.0, 0.0])
                distance = action.get("distance", 0.05)
                plan.add(MotionPrimitive.push(direction, float(distance)))
            elif atype == "insert":
                target = action.get("target_pos", action.get("target", [0.5, 0.0, 0.35]))
                if isinstance(target, str):
                    target = [0.5, 0.0, 0.35]
                plan.add(MotionPrimitive.insert(target))
            elif atype == "release":
                plan.add(MotionPrimitive.release())
            elif atype == "wait":
                steps = action.get("steps", 50)
                plan.add(MotionPrimitive.wait(int(steps)))
        return plan


# ---------------------------------------------------------------------------
# Universal motion executor
# ---------------------------------------------------------------------------

class UniversalMotionExecutor:
    """Executes motion plans on any MuJoCo robot using IK + interpolation.

    Parameters
    ----------
    model : mujoco.MjModel
        The loaded scene model.
    data : mujoco.MjData
        The MuJoCo data instance.
    spec : RobotSpec
        The robot specification.
    object_body_id : int, optional
        MuJoCo body ID of the manipulated object (for tracking).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        spec: RobotSpec,
        object_body_id: int = -1,
    ):
        self.model = model
        self.data = data
        self.spec = spec
        self.object_body_id = object_body_id

        self.ik = JacobianIKSolver(model, data, spec)
        self.gripper = GripperController(model, data, spec)

        # Arm actuator ctrl indices
        self._arm_ctrl_indices: List[int] = []
        for aname in spec.actuator_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                self._arm_ctrl_indices.append(aid)
        # Fallback: if no actuators found by name, use positional indices
        # (some MJCF actuators have no name attribute, only joint)
        if not self._arm_ctrl_indices and spec.dof > 0:
            self._arm_ctrl_indices = list(range(min(spec.dof, model.nu)))

        # Record initial object position
        self._initial_object_pos: Optional[npt.NDArray[np.float64]] = None
        if object_body_id >= 0:
            self._initial_object_pos = data.xpos[object_body_id].copy()

    def _safe_target_pos(self, target: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        safe = np.asarray(target, dtype=np.float64).copy()
        if safe.shape[0] >= 3:
            safe[2] = max(float(safe[2]), TABLE_TOP_Z + EE_TABLE_CLEARANCE_M)
        return safe

    def _robot_table_contact_pairs(self) -> List[str]:
        pairs: List[str] = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
            pair = f"{geom1}|{geom2}".lower()
            if "table" not in pair:
                continue
            other = geom2.lower() if "table" in geom1.lower() else geom1.lower()
            if other.startswith("obj_") or "floor" in other:
                continue
            if "finger" in other or "pad" in other:
                continue
            # Treat table contact by the robot body/hand/links as a global safety
            # violation. Fingertip/tool contact can be valid for tabletop skills, so
            # physical penetration is enforced separately through EE/table clearance.
            robotish = (
                other.startswith("fr3_")
                or other.startswith("panda_")
                or "hand" in other
                or "link" in other
                or "collision" in other
            )
            if robotish:
                pairs.append(pair)
        return pairs

    def _update_safety(self, trace: MotionTrace, ee_pos: npt.NDArray[np.float64]) -> None:
        z = float(ee_pos[2])
        trace.min_ee_z = min(trace.min_ee_z, z)
        if z < TABLE_TOP_Z + EE_TABLE_CLEARANCE_M:
            trace.table_penetration_steps += 1
            trace.table_clearance_ok = False
        contact_pairs = self._robot_table_contact_pairs()
        if contact_pairs:
            trace.table_contact_steps += 1
            trace.table_clearance_ok = False
            for pair in contact_pairs:
                if pair not in trace.table_contact_pairs:
                    trace.table_contact_pairs.append(pair)

    def _arm_qpos(self) -> npt.NDArray[np.float64]:
        """Read arm qpos via resolved robot joint addresses."""
        return self.ik._get_arm_qpos()

    def execute_plan(
        self,
        plan: MotionPlan,
        sample_every: int = 10,
        contact_body_prefix: str = "",
        contact_target_body: str = "",
        render_callback=None,
    ) -> MotionTrace:
        """Execute a full motion plan and return the trace.

        If *render_callback* is provided, it will be called with
        ``(step: int, frame: np.ndarray)`` at regular intervals during
        execution so that callers can capture MuJoCo render frames.
        """
        self._render_callback = render_callback
        self._render_step_counter = 0
        trace = MotionTrace()

        contact_logger = ContactLogger(
            self.model, self.data,
            body_a_prefix=contact_body_prefix,
            body_b_name=contact_target_body,
        )

        for prim in plan.primitives:
            self._execute_primitive(prim, trace, contact_logger, sample_every)

        # Compute final metrics
        trace.final_ee_pos = self.ik.get_ee_pos()
        if self.object_body_id >= 0:
            mujoco.mj_forward(self.model, self.data)
            trace.final_object_pos = self.data.xpos[self.object_body_id].copy()
            if self._initial_object_pos is not None:
                trace.object_displacement = float(
                    np.linalg.norm(trace.final_object_pos - self._initial_object_pos)
                )

        trace.contact_steps = contact_logger.total_contact_steps
        trace.max_touch_force = contact_logger.max_touch_force

        return trace

    def _execute_primitive(
        self,
        prim: MotionPrimitive,
        trace: MotionTrace,
        contact_logger: ContactLogger,
        sample_every: int,
    ) -> None:
        """Execute a single motion primitive."""
        ptype = prim.primitive_type

        if ptype == MotionPrimitiveType.REACH:
            self._do_reach(prim, trace, contact_logger, sample_every)
        elif ptype == MotionPrimitiveType.GRASP:
            self._do_gripper(prim, trace, contact_logger, sample_every, close=True)
        elif ptype == MotionPrimitiveType.RELEASE:
            self._do_gripper(prim, trace, contact_logger, sample_every, close=False)
        elif ptype == MotionPrimitiveType.LIFT:
            self._do_lift(prim, trace, contact_logger, sample_every)
        elif ptype == MotionPrimitiveType.PLACE:
            self._do_reach(prim, trace, contact_logger, sample_every)
        elif ptype == MotionPrimitiveType.PUSH:
            self._do_push(prim, trace, contact_logger, sample_every)
        elif ptype == MotionPrimitiveType.INSERT:
            self._do_insert(prim, trace, contact_logger, sample_every)
        elif ptype == MotionPrimitiveType.WAIT:
            self._do_wait(prim, trace, contact_logger, sample_every)

    def _do_reach(
        self,
        prim: MotionPrimitive,
        trace: MotionTrace,
        contact_logger: ContactLogger,
        sample_every: int,
    ) -> None:
        """Reach to target position using IK."""
        if prim.target_pos is None:
            return

        target = self._safe_target_pos(np.array(prim.target_pos, dtype=np.float64))
        target_quat = np.array(prim.target_quat, dtype=np.float64) if prim.target_quat else None

        # Solve IK
        ik_result = self.ik.solve(
            target_pos=target,
            target_quat=target_quat,
            max_iters=200,
            pos_tol=1e-3,
        )
        trace.ik_results.append(ik_result)

        # Drive to IK solution using smooth interpolation
        self._drive_to_qpos(ik_result.qpos, prim.steps, trace, contact_logger, sample_every)

    def _do_lift(
        self,
        prim: MotionPrimitive,
        trace: MotionTrace,
        contact_logger: ContactLogger,
        sample_every: int,
    ) -> None:
        """Lift from current position by height."""
        current_pos = self.ik.get_ee_pos()
        target_pos = current_pos.copy()
        target_pos[2] += prim.height
        target_pos = self._safe_target_pos(target_pos)

        ik_result = self.ik.solve(target_pos=target_pos, max_iters=200, pos_tol=1e-3)
        trace.ik_results.append(ik_result)

        self._drive_to_qpos(ik_result.qpos, prim.steps, trace, contact_logger, sample_every)

        # Track lifted height
        if self.object_body_id >= 0:
            mujoco.mj_forward(self.model, self.data)
            obj_pos = self.data.xpos[self.object_body_id]
            if self._initial_object_pos is not None:
                trace.lifted_height = float(obj_pos[2] - self._initial_object_pos[2])

    def _do_push(
        self,
        prim: MotionPrimitive,
        trace: MotionTrace,
        contact_logger: ContactLogger,
        sample_every: int,
    ) -> None:
        """Push along direction by distance, scaled by force_scale."""
        current_pos = self.ik.get_ee_pos()
        direction = np.array(prim.direction or [1.0, 0.0, 0.0], dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        scaled_distance = prim.distance * prim.force_scale
        target_pos = self._safe_target_pos(current_pos + direction * scaled_distance)

        ik_result = self.ik.solve(target_pos=target_pos, max_iters=200, pos_tol=1e-3)
        trace.ik_results.append(ik_result)

        self._drive_to_qpos(ik_result.qpos, prim.steps, trace, contact_logger, sample_every)

    def _do_insert(
        self,
        prim: MotionPrimitive,
        trace: MotionTrace,
        contact_logger: ContactLogger,
        sample_every: int,
    ) -> None:
        """Insert: approach along axis, then move to target."""
        if prim.target_pos is None:
            return

        target = self._safe_target_pos(np.array(prim.target_pos, dtype=np.float64))
        approach = np.array(prim.approach_axis or [0.0, 0.0, -1.0], dtype=np.float64)
        approach = approach / (np.linalg.norm(approach) + 1e-8)

        # Phase 1: approach from the opposite side of the insertion direction.
        approach_pos = self._safe_target_pos(target - approach * 0.05)
        ik1 = self.ik.solve(target_pos=approach_pos, max_iters=200, pos_tol=1e-3)
        trace.ik_results.append(ik1)
        self._drive_to_qpos(ik1.qpos, prim.steps // 2, trace, contact_logger, sample_every)

        # Phase 2: insert
        ik2 = self.ik.solve(target_pos=target, max_iters=200, pos_tol=1e-3)
        trace.ik_results.append(ik2)
        self._drive_to_qpos(ik2.qpos, prim.steps // 2, trace, contact_logger, sample_every)

    def _do_gripper(
        self,
        prim: MotionPrimitive,
        trace: MotionTrace,
        contact_logger: ContactLogger,
        sample_every: int,
        close: bool,
    ) -> None:
        """Open or close gripper for specified steps."""
        if close:
            self.gripper.close(prim.force_scale)
        else:
            self.gripper.open()

        render_cb = getattr(self, "_render_callback", None)
        for step in range(prim.steps):
            mujoco.mj_step(self.model, self.data)
            if render_cb and step % 28 == 0:
                render_cb(step, self.data)
            if step % sample_every == 0:
                ee_pos = self.ik.get_ee_pos()
                self._update_safety(trace, ee_pos)
                contact_logger.log_step(step, ee_pos)
                trace.frames.append(TraceFrame(
                    step=step,
                    qpos=self._arm_qpos(),
                    ee_pos=ee_pos,
                ))

    def _do_wait(
        self,
        prim: MotionPrimitive,
        trace: MotionTrace,
        contact_logger: ContactLogger,
        sample_every: int,
    ) -> None:
        """Wait (step physics) for specified steps."""
        render_cb = getattr(self, "_render_callback", None)
        for step in range(prim.steps):
            mujoco.mj_step(self.model, self.data)
            if render_cb and step % 28 == 0:
                render_cb(step, self.data)
            if step % sample_every == 0:
                ee_pos = self.ik.get_ee_pos()
                self._update_safety(trace, ee_pos)
                contact_logger.log_step(step, ee_pos)
                trace.frames.append(TraceFrame(
                    step=step,
                    qpos=self._arm_qpos(),
                    ee_pos=ee_pos,
                ))

    def _drive_to_qpos(
        self,
        target_qpos: npt.NDArray[np.float64],
        steps: int,
        trace: MotionTrace,
        contact_logger: ContactLogger,
        sample_every: int,
    ) -> None:
        """Smoothly interpolate from current arm qpos to target qpos.

        Uses actuator control (data.ctrl) to drive the motion, letting
        MuJoCo's physics engine handle the actual joint movement. This
        produces realistic, stable trajectories.
        """
        model = self.model
        data = self.data
        spec = self.spec
        render_cb = getattr(self, "_render_callback", None)
        render_interval = 28  # capture a frame every 28 physics steps

        start_qpos = self._arm_qpos()
        target_qpos = np.asarray(target_qpos, dtype=np.float64)[:spec.dof]
        if len(target_qpos) < spec.dof:
            padded = start_qpos.copy()
            padded[:len(target_qpos)] = target_qpos
            target_qpos = padded

        for step in range(steps):
            # Linear blend (same as arm_runner._drive)
            blend = min(1.0, (step + 1) / (steps * 0.55))
            interp_qpos = start_qpos + blend * (target_qpos - start_qpos)

            # Set actuator targets (ctrl), not qpos directly
            for i in range(min(spec.dof, len(self._arm_ctrl_indices))):
                data.ctrl[self._arm_ctrl_indices[i]] = interp_qpos[i]

            # Step physics — actuators drive the joints
            mujoco.mj_step(model, data)

            # Render callback
            if render_cb and step % render_interval == 0:
                render_cb(step, data)

            # Log
            if step % sample_every == 0:
                ee_pos = self.ik.get_ee_pos()
                self._update_safety(trace, ee_pos)
                contact_logger.log_step(step, ee_pos)

                obj_pos = None
                if self.object_body_id >= 0:
                    obj_pos = data.xpos[self.object_body_id].copy()

                trace.frames.append(TraceFrame(
                    step=step,
                    qpos=self._arm_qpos(),
                    ee_pos=ee_pos,
                    object_pos=obj_pos,
                    contact_count=contact_logger.log[-1].contact_count if contact_logger.log else 0,
                    touch_force=contact_logger.log[-1].max_force if contact_logger.log else 0.0,
                ))
