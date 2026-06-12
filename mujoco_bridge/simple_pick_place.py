"""Physics-based pick-and-place with FR3 arm + Franka Hand.

Uses the combined scene from :mod:`scene_builder` and drives the arm via
pre-defined joint-space waypoints.  The cube is placed at the position where
the hand actually settles with the actuators, so grasping works despite
gravity-induced steady-state error.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from .runner import ExperimentConfig, RunResult, seeded_noise

ROOT = Path(__file__).resolve().parent

# Experiment-space parameter mappings
# Offsets from hand's natural settle position (in meters).
# Finger pads are 17mm wide (±8.5mm), cube is 40mm wide.
# Cube face at offset-20mm from center; pad edge at ±8.5mm from hand.
# Contact requires: offset < 28.5mm (20+8.5).
OBJECT_OFFSETS: Dict[str, Tuple[float, float]] = {
    "small": (0.003, 0.002),
    "medium": (0.010, -0.007),
    "large": (0.020, 0.014),
}

FRICTION_VALUES = {
    "low": 0.3,
    "medium": 0.8,
    "high": 1.5,
}

HEIGHT_OFFSETS = {
    "-2cm": -0.020,
    "0": 0.0,
    "+2cm": 0.020,
}

NOISE_VALUES = {
    "none": 0.0,
    "light": 0.018,
    "heavy": 0.045,
}

# Table surface z in the scene
TABLE_Z = 0.37
CUBE_HALF = 0.02
CUBE_REST_Z = TABLE_Z + CUBE_HALF  # 0.39

# Pre-defined joint-space waypoints for the pick-and-place trajectory.
# These are ctrl targets for the 7 arm actuators.  The hand settles at a
# gravity-determined position that may differ from the IK solution.
WAYPOINT_APPROACH = [0.0, 1.5, 0.0, -1.0, 0.0, 1.8, -0.7853]
WAYPOINT_GRASP = [0.0, 1.8, 0.0, -0.5, 0.0, 2.0, -0.7853]
WAYPOINT_LIFT = [0.0, 1.4, 0.0, -0.5, 0.0, 2.0, -0.7853]
WAYPOINT_MOVE = [0.0, 0.9, 0.0, -0.6, 0.0, 1.2, -0.79]
WAYPOINT_PLACE = [0.0, 1.1, 0.0, -0.5, 0.0, 1.7, -0.79]


def _name_id(mujoco, model, obj_type, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return idx


class FR3PickPlaceSim:
    """Physics-driven FR3 pick-and-place simulation.

    Each ``run()`` builds a fresh scene, places the cube at the position where
    the hand actually reaches (accounting for actuator gravity error), and
    executes a scripted trajectory whose success depends on real contact physics.
    """

    def __init__(self) -> None:
        import mujoco  # type: ignore

        self.mujoco = mujoco
        from .scene_builder import build_scene

        self._build_scene = build_scene

    # ------------------------------------------------------------------
    # Hand position calibration
    # ------------------------------------------------------------------

    def _find_hand_settle_pos(
        self, model, data, ctrl: List[float], steps: int = 0
    ) -> np.ndarray:
        """Return the midpoint between fingertip touch sites.

        If *steps* > 0, drive the arm to *ctrl* first.
        """
        mujoco = self.mujoco
        for _ in range(steps):
            data.ctrl[:7] = ctrl
            data.ctrl[7] = 255  # open gripper
            mujoco.mj_step(model, data)
        lf_site = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "left_finger_touch")
        rf_site = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "right_finger_touch")
        return (data.site_xpos[lf_site] + data.site_xpos[rf_site]) / 2.0

    # ------------------------------------------------------------------
    # Simulation driver
    # ------------------------------------------------------------------

    def _drive_to(
        self,
        model,
        data,
        target_ctrl: List[float],
        grip_ctrl: float,
        steps: int,
        trace: List[dict] | None,
        label: str,
        sensor_log: Dict[str, List[float]] | None = None,
        render_callback: Callable[[object, object, str], None] | None = None,
        render_interval: int = 50,
    ) -> None:
        """Step the simulation, commanding *target_ctrl* for *steps* timesteps."""
        mujoco = self.mujoco
        for step_idx in range(steps):
            data.ctrl[:7] = target_ctrl
            data.ctrl[7] = grip_ctrl
            mujoco.mj_step(model, data)
            if trace is not None and step_idx % 20 == 0:
                self._sample_trace(model, data, trace, label)
            if sensor_log is not None:
                self._log_sensors(model, data, sensor_log)
            if render_callback is not None and step_idx % render_interval == 0:
                render_callback(model, data, label)

    # ------------------------------------------------------------------
    # Sensor helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_sensors(mujoco, data, log: Dict[str, List[float]]) -> None:
        log.setdefault("left_touch", []).append(float(data.sensordata[0]))
        log.setdefault("right_touch", []).append(float(data.sensordata[1]))
        log.setdefault("finger1_pos", []).append(float(data.sensordata[2]))
        log.setdefault("finger2_pos", []).append(float(data.sensordata[3]))
        log.setdefault("cube_pos_x", []).append(float(data.sensordata[6]))
        log.setdefault("cube_pos_y", []).append(float(data.sensordata[7]))
        log.setdefault("cube_pos_z", []).append(float(data.sensordata[8]))

    # ------------------------------------------------------------------
    # Trace collection
    # ------------------------------------------------------------------

    def _sample_trace(self, model, data, trace: List[dict], label: str) -> None:
        mujoco = self.mujoco
        cube_body_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")
        lf_site = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "left_finger_touch")
        rf_site = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "right_finger_touch")
        grip_pos = (data.site_xpos[lf_site] + data.site_xpos[rf_site]) / 2.0
        # closed: 0 = open command (ctrl=255), 1 = close command (ctrl<128)
        grip_ctrl = float(data.ctrl[7])
        closed = 0.0 if grip_ctrl > 128 else 1.0
        trace.append(
            {
                "label": label,
                "time": round(float(data.time), 3),
                "gripper": {
                    "x": round(float(grip_pos[0]), 4),
                    "y": round(float(grip_pos[1]), 4),
                    "z": round(float(grip_pos[2]), 4),
                    "closed": closed,
                },
                "cube": {
                    "x": round(float(data.xpos[cube_body_id][0]), 4),
                    "y": round(float(data.xpos[cube_body_id][1]), 4),
                    "z": round(float(data.xpos[cube_body_id][2]), 4),
                },
                "target": {"x": 0.60, "y": 0.0, "z": CUBE_REST_Z},
            }
        )

    # ------------------------------------------------------------------
    # Cube placement
    # ------------------------------------------------------------------

    def _place_cube(self, model, data, pos: Tuple[float, float], z: float = CUBE_REST_Z) -> None:
        mujoco = self.mujoco
        cube_joint_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        qadr = model.jnt_qposadr[cube_joint_id]
        data.qpos[qadr : qadr + 7] = [pos[0], pos[1], z, 1, 0, 0, 0]
        data.qvel[qadr : qadr + 6] = 0

    def _apply_friction(self, model, config: ExperimentConfig) -> None:
        mujoco = self.mujoco
        friction = FRICTION_VALUES[config.friction]
        cube_geom_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        model.geom_friction[cube_geom_id, 0] = friction
        # Finger friction matches cube friction so contact pair friction varies.
        finger_friction = friction
        for body_name in ("left_finger", "right_finger"):
            body_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == body_id and model.geom_contype[g] > 0:
                    model.geom_friction[g, 0] = finger_friction

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, config: ExperimentConfig, index: int) -> RunResult:
        result, _ = self.run_with_trace(config, index, collect_trace=False)
        return result

    def run_with_trace(
        self,
        config: ExperimentConfig,
        index: int,
        collect_trace: bool = True,
        render_callback: Callable[[object, object, str], None] | None = None,
        render_interval: int = 50,
    ) -> Tuple[RunResult, List[dict]]:
        mujoco = self.mujoco
        model, data = self._build_scene(mujoco)

        # Reset to keyframe
        mujoco.mj_resetDataKeyframe(model, data, 0)
        self._apply_friction(model, config)

        # --- Cube placement: offset from hand's natural settle position ---
        # Hand settles at ~(0.61, 0.0) with the grasp waypoint.
        offsets = OBJECT_OFFSETS[config.object_offset]
        noise = NOISE_VALUES[config.vision_noise]
        # Object offset (systematic) + vision noise (random per-run)
        jit_x = (seeded_noise(index + 501) - 0.5) * 0.003
        jit_y = (seeded_noise(index + 502) - 0.5) * 0.003
        noise_x = (seeded_noise(index + 601) - 0.5) * noise
        noise_y = (seeded_noise(index + 602) - 0.5) * noise
        cube_x = 0.61 + offsets[0] + jit_x + noise_x
        cube_y = 0.00 + offsets[1] + jit_y + noise_y

        sensor_log: Dict[str, List[float]] = {}
        height_offset = HEIGHT_OFFSETS[config.grasp_height_offset]
        grasp_ctrl = WAYPOINT_GRASP.copy()
        grasp_ctrl[1] += height_offset * 5.0  # scale height offset to joint radians

        OPEN = 255.0
        CLOSED = 0.0  # fully closed, no extra squeeze (friction-dependent)

        trace: List[dict] = []

        # ---- Phase 0: Pre-position arm to grasp pose (gripper open) ----
        # Don't collect trace during pre-position (arm swings through cube location)
        self._drive_to(
            model, data, grasp_ctrl, OPEN, 500,
            None, "approach", sensor_log,
        )

        # Place cube at actual position (arm is now in place)
        self._place_cube(model, data, (cube_x, cube_y), z=CUBE_REST_Z)
        mujoco.mj_forward(model, data)

        if collect_trace:
            self._sample_trace(model, data, trace, "cube-ready")
        if render_callback is not None:
            render_callback(model, data, "cube-ready")

        # ---- Phase 1: Close gripper ----
        self._drive_to(
            model, data, grasp_ctrl, CLOSED, 800,
            trace if collect_trace else None, "close-gripper", sensor_log,
            render_callback, render_interval,
        )

        # Record grip force after close (before lift)
        grip_after_close = max(
            max(sensor_log.get("left_touch", [0])),
            max(sensor_log.get("right_touch", [0])),
        )

        # ---- Phase 2: Lift (vertical, physics-based) ----
        lift_ctrl = WAYPOINT_LIFT.copy()
        self._drive_to(
            model, data, lift_ctrl, CLOSED, 800,
            trace if collect_trace else None, "lift-object", sensor_log,
            render_callback, render_interval,
        )

        # Check if cube was actually lifted
        cube_body_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")
        cube_pos_after_lift = data.xpos[cube_body_id].copy()
        cube_was_lifted = cube_pos_after_lift[2] > TABLE_Z + 0.05

        # Detect slip during lift: cube reaches peak then drops significantly
        cube_z_log = sensor_log.get("cube_pos_z", [])
        lift_slip = False
        if cube_was_lifted and len(cube_z_log) > 1000:
            lift_z = cube_z_log[800:]  # lift phase z values
            if len(lift_z) > 50:
                peak_z = max(lift_z)
                final_z = lift_z[-1]
                # Slip: cube was lifted but dropped back down
                lift_slip = (peak_z - final_z) > 0.06

        if cube_was_lifted and not lift_slip:
            # ---- Phase 3: Move using only contact physics, no cube teleport ----
            move_ctrl = WAYPOINT_MOVE.copy()
            self._drive_to(
                model, data, move_ctrl, CLOSED, 800,
                trace if collect_trace else None, "move-to-target", sensor_log,
                render_callback, render_interval,
            )
            place_ctrl = WAYPOINT_PLACE.copy()
            self._drive_to(
                model, data, place_ctrl, CLOSED, 500,
                trace if collect_trace else None, "place-object", sensor_log,
                render_callback, render_interval,
            )
        else:
            self._drive_to(
                model, data, lift_ctrl, CLOSED, 200,
                trace if collect_trace else None, "move-to-target", sensor_log,
                render_callback, render_interval,
            )

        # ---- Phase 5: Release ----
        self._drive_to(
            model, data, grasp_ctrl, OPEN, 300,
            trace if collect_trace else None, "release", sensor_log,
            render_callback, render_interval,
        )

        # ---- Phase 6: Retreat ----
        retreat_ctrl = WAYPOINT_APPROACH.copy()
        self._drive_to(
            model, data, retreat_ctrl, OPEN, 200,
            trace if collect_trace else None, "retreat", sensor_log,
            render_callback, render_interval,
        )

        # ---- Evaluate outcome ----
        cube_body_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")
        cube_pos = data.xpos[cube_body_id].copy()
        target_x, target_y = 0.60, 0.0
        final_distance = float(
            ((cube_pos[0] - target_x) ** 2 + (cube_pos[1] - target_y) ** 2) ** 0.5
        )

        # Classify failure BEFORE checking success
        max_grip_force = max(
            max(sensor_log.get("left_touch", [0])),
            max(sensor_log.get("right_touch", [0])),
        )
        touch_contact_duration = sum(
            1 for v in sensor_log.get("left_touch", []) if v > 0.001
        ) + sum(
            1 for v in sensor_log.get("right_touch", []) if v > 0.001
        )
        touch_contact_duration *= 0.002  # timestep = 0.002s

        # Detect general slip (cube z variation during closed-gripper phases)
        cube_z_log = sensor_log.get("cube_pos_z", [])
        slip_detected = lift_slip

        # Success requires: cube was actually lifted, placed near target, grasped, no slip
        pos_ok = final_distance < 0.075 and TABLE_Z - 0.01 < cube_pos[2] < TABLE_Z + 0.06
        grasp_ok = max_grip_force > 0.01
        success = bool(cube_was_lifted and pos_ok and grasp_ok and not slip_detected)

        if success:
            failure_type = "none"
        elif not grasp_ok:
            failure_type = "grasp_miss"
        elif slip_detected:
            failure_type = "slip"
        elif not cube_was_lifted and grasp_ok:
            # Fingers contacted cube but couldn't lift it → slip
            failure_type = "slip"
        elif cube_pos[2] < TABLE_Z - 0.01:
            failure_type = "collision"
        else:
            failure_type = "grasp_miss"

        result = RunResult(
            run_id=config.run_id,
            object_offset=config.object_offset,
            friction=config.friction,
            grasp_height_offset=config.grasp_height_offset,
            vision_noise=config.vision_noise,
            control_freq=config.control_freq,
            success=bool(success),
            failure_type=failure_type,
            trajectory_error=round(float(final_distance), 3),
            collision_count=1 if failure_type == "collision" else 0,
            final_distance=round(float(final_distance), 3),
            max_grip_force=round(float(max_grip_force), 3),
            cube_slip_detected=bool(slip_detected),
            touch_contact_duration=round(float(touch_contact_duration), 3),
        )
        return result, trace


def run_smoke() -> dict:
    """Quick smoke test: run one pick-and-place and return metrics."""
    env = FR3PickPlaceSim()
    config = ExperimentConfig(
        run_id="smoke_001",
        object_offset="small",
        friction="medium",
        grasp_height_offset="0",
        vision_noise="none",
    )
    result, trace = env.run_with_trace(config, 1)
    payload = asdict(result)
    payload["trace_frames"] = len(trace)
    return payload
