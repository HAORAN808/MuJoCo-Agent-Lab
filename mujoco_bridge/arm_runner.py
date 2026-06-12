from __future__ import annotations

import base64
import io
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image

from .object_library import get_object


TABLE_Z = 0.37
CUBE_HALF = 0.02
CUBE_REST_Z = TABLE_Z + CUBE_HALF

OPEN_GRIPPER = 255.0
CLOSED_GRIPPER = 0.0

FR3_HOME = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
FR3_PREGRASP = [0.0, 1.5, 0.0, -1.0, 0.0, 1.8, -0.7853]
FR3_GRASP = [0.0, 1.8, 0.0, -0.5, 0.0, 2.0, -0.7853]
FR3_LIFT = [0.0, 1.4, 0.0, -0.5, 0.0, 2.0, -0.7853]
FR3_SWEEP = [0.0, 0.95, 0.0, -0.65, 0.0, 1.25, -0.7853]
FR3_RETREAT = [0.0, 1.2, 0.0, -1.0, 0.0, 1.6, -0.7853]

ARM_SKILLS: Dict[str, Dict[str, Any]] = {
    "reach_touch": {
        "title": "Reach and touch object",
        "robot": "franka_fr3",
        "runner_status": "implemented",
        "runner_module": "mujoco_bridge.arm_runner.FR3ArmSkillRunner",
        "verified_by": ["python -m mujoco_bridge.arm_smoke"],
        "description": "Move the FR3 gripper to the object and close fingers until touch sensors report contact.",
        "inputs": ["object_id", "object_xy", "friction", "grasp_height_delta"],
        "metrics": ["contact_steps", "max_touch_force", "end_effector_pos", "final_object_pos"],
    },
    "button_press": {
        "title": "Button press contact",
        "robot": "franka_fr3",
        "runner_status": "implemented",
        "runner_module": "mujoco_bridge.arm_runner.FR3ArmSkillRunner",
        "verified_by": ["python -m mujoco_bridge.arm_smoke"],
        "description": "Move the FR3 gripper into a simplified button target and close until contact is detected.",
        "inputs": ["object_id", "object_xy", "friction", "grasp_height_delta"],
        "metrics": ["contact_steps", "max_touch_force", "object_displacement"],
    },
    "pick_lift": {
        "title": "Pick and lift object",
        "robot": "franka_fr3",
        "runner_status": "implemented",
        "runner_module": "mujoco_bridge.arm_runner.FR3ArmSkillRunner",
        "verified_by": ["python -m mujoco_bridge.arm_smoke"],
        "description": "Reach, close the two-finger gripper, and lift the object using MuJoCo contact physics.",
        "inputs": ["object_id", "object_xy", "friction", "grasp_height_delta"],
        "metrics": ["contact_steps", "max_touch_force", "lifted_height", "object_displacement"],
    },
    "contact_sweep": {
        "title": "Contact sweep with object",
        "robot": "franka_fr3",
        "runner_status": "implemented",
        "runner_module": "mujoco_bridge.arm_runner.FR3ArmSkillRunner",
        "verified_by": ["python -m mujoco_bridge.arm_smoke"],
        "description": "Establish gripper-object contact and move through a sweep target. This is a reusable contact primitive, not a calibrated planar pushing policy.",
        "inputs": ["object_id", "object_xy", "friction", "sweep_scale"],
        "metrics": ["contact_steps", "max_touch_force", "object_displacement", "lifted_height"],
    },
    "tool_contact_sweep": {
        "title": "Held-tool contact sweep",
        "robot": "franka_fr3",
        "runner_status": "implemented",
        "runner_module": "mujoco_bridge.arm_runner.FR3ArmSkillRunner",
        "verified_by": ["python -m mujoco_bridge.arm_smoke"],
        "description": "Attach a simple tool geometry to the FR3 hand and sweep it into a target object using MuJoCo contacts.",
        "inputs": ["tool_id", "object_id", "object_xy", "friction", "sweep_scale"],
        "metrics": ["tool_contact_steps", "contact_steps", "max_touch_force", "object_displacement"],
    },
    "peg_insert": {
        "title": "Held peg insertion contact",
        "robot": "franka_fr3",
        "runner_status": "implemented",
        "runner_module": "mujoco_bridge.arm_runner.FR3ArmSkillRunner",
        "verified_by": ["python -m mujoco_bridge.arm_smoke"],
        "description": "Attach a peg geometry to the FR3 hand and drive it into a simplified socket target.",
        "inputs": ["object_id", "object_xy", "friction", "sweep_scale"],
        "metrics": ["tool_contact_steps", "max_touch_force", "object_displacement"],
    },
}


@dataclass(frozen=True)
class ArmSkillConfig:
    skill_id: str
    object_id: str = "cube_5cm"
    tool_id: str = ""
    object_xy: tuple[float, float] = (0.59, 0.0)
    friction: float = 0.9
    grasp_height_delta: float = 0.0
    sweep_scale: float = 1.0


@dataclass
class ArmSkillResult:
    skill_id: str
    robot: str
    source: str
    success: bool
    failure_type: str
    contact_steps: int
    max_touch_force: float
    final_object_pos: List[float]
    end_effector_pos: List[float]
    object_displacement: float
    lifted_height: float
    object_id: str = "cube_5cm"
    tool_id: str = ""
    tool_contact_steps: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)


def _name_id(mujoco, model, obj_type, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return idx


def _rounded(values: Sequence[float], digits: int = 4) -> List[float]:
    return [round(float(v), digits) for v in values]


class FR3ArmSkillRunner:
    """Reusable FR3 + Franka Hand skill runner.

    This is intentionally lower-level than a task runner.  It provides
    repeatable mechanical-arm primitives that task plugins can compose:
    reach/touch, grasp/lift, and contact sweep.  It still uses scripted joint
    targets; it is not a planner or learned controller.
    """

    robot = "franka_fr3"
    source = "mujoco_fr3_arm_skill"

    def __init__(self) -> None:
        import mujoco  # type: ignore

        from .scene_builder import build_scene

        self.mujoco = mujoco
        self._build_scene = build_scene
        self._render_callback = None
        self._render_every = 35

    def build(self, object_id: str = "cube_5cm", held_tool: str | None = None):
        model, data = self._build_scene(self.mujoco, object_id=object_id, held_tool=held_tool)
        self.mujoco.mj_resetDataKeyframe(model, data, 0)
        return model, data

    def _object_rest_z(self, object_id: str) -> float:
        obj = get_object(object_id)
        if obj.xml_kind != "primitive":
            obj = get_object("cube_5cm")
        if obj.geometry == "box":
            half_z = obj.size_m[2]
        elif obj.geometry == "sphere":
            half_z = obj.size_m[0]
        elif obj.geometry == "cylinder":
            half_z = obj.size_m[1]
        else:
            half_z = CUBE_HALF
        return TABLE_Z + half_z

    def _sensor_value(self, model, data, name: str) -> float:
        sensor_id = _name_id(self.mujoco, model, self.mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = int(model.sensor_adr[sensor_id])
        return float(data.sensordata[adr])

    def _body_pos(self, model, data, name: str) -> np.ndarray:
        body_id = _name_id(self.mujoco, model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[body_id].copy()

    def _gripper_center(self, model, data) -> np.ndarray:
        left = _name_id(self.mujoco, model, self.mujoco.mjtObj.mjOBJ_SITE, "left_finger_touch")
        right = _name_id(self.mujoco, model, self.mujoco.mjtObj.mjOBJ_SITE, "right_finger_touch")
        return (data.site_xpos[left] + data.site_xpos[right]) / 2.0

    def _place_cube(self, model, data, xy: tuple[float, float], z: float) -> None:
        joint_id = _name_id(self.mujoco, model, self.mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        adr = int(model.jnt_qposadr[joint_id])
        data.qpos[adr : adr + 7] = [xy[0], xy[1], z, 1.0, 0.0, 0.0, 0.0]
        data.qvel[adr : adr + 6] = 0.0
        self.mujoco.mj_forward(model, data)

    def _set_cube_friction(self, model, friction: float) -> None:
        cube_geom = _name_id(self.mujoco, model, self.mujoco.mjtObj.mjOBJ_GEOM, "cube")
        model.geom_friction[cube_geom, 0] = friction
        for body_name in ("left_finger", "right_finger"):
            body_id = _name_id(self.mujoco, model, self.mujoco.mjtObj.mjOBJ_BODY, body_name)
            for geom_id in range(model.ngeom):
                if int(model.geom_bodyid[geom_id]) == body_id and model.geom_contype[geom_id] > 0:
                    model.geom_friction[geom_id, 0] = friction

    def _sample(self, model, data, label: str) -> Dict[str, Any]:
        cube = self._body_pos(model, data, "cube_body")
        grip = self._gripper_center(model, data)
        return {
            "label": label,
            "time": round(float(data.time), 3),
            "gripper": {"x": round(float(grip[0]), 4), "y": round(float(grip[1]), 4), "z": round(float(grip[2]), 4)},
            "object": {"x": round(float(cube[0]), 4), "y": round(float(cube[1]), 4), "z": round(float(cube[2]), 4)},
            "joint_targets": _rounded(data.ctrl[:7], 4),
            "gripper_command": round(float(data.ctrl[7]), 3),
        }

    def _contact_count_between_geom_prefix_and_body(self, model, data, geom_prefix: str, body_name: str) -> tuple[int, float]:
        body_id = _name_id(self.mujoco, model, self.mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom_ids = set()
        for geom_id in range(model.ngeom):
            name = self.mujoco.mj_id2name(model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith(geom_prefix):
                geom_ids.add(geom_id)
        count = 0
        max_force = 0.0
        for idx in range(data.ncon):
            contact = data.contact[idx]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            b1 = int(model.geom_bodyid[g1])
            b2 = int(model.geom_bodyid[g2])
            if (g1 in geom_ids and b2 == body_id) or (g2 in geom_ids and b1 == body_id):
                count += 1
                force = np.zeros(6)
                self.mujoco.mj_contactForce(model, data, idx, force)
                max_force = max(max_force, float(np.linalg.norm(force[:3])))
        return count, max_force

    def _drive(
        self,
        model,
        data,
        target: Sequence[float],
        grip: float,
        steps: int,
        label: str,
        trace: List[Dict[str, Any]],
        contact_log: Dict[str, Any],
        sample_every: int = 40,
    ) -> None:
        start = np.array(data.ctrl[:7], dtype=float)
        target_arr = np.array(target, dtype=float)
        for step in range(steps):
            blend = min(1.0, (step + 1) / max(1, steps * 0.55))
            data.ctrl[:7] = start + (target_arr - start) * blend
            data.ctrl[7] = grip
            self.mujoco.mj_step(model, data)
            left = self._sensor_value(model, data, "left_touch")
            right = self._sensor_value(model, data, "right_touch")
            touch = max(left, right)
            if touch > 0.001:
                contact_log["contact_steps"] += 1
                contact_log["max_touch_force"] = max(contact_log["max_touch_force"], touch)
            try:
                tool_contacts, tool_force = self._contact_count_between_geom_prefix_and_body(
                    model,
                    data,
                    "held_tool",
                    "cube_body",
                )
            except KeyError:
                tool_contacts, tool_force = 0, 0.0
            if tool_contacts:
                contact_log["tool_contact_steps"] = int(contact_log.get("tool_contact_steps", 0)) + tool_contacts
                contact_log["max_touch_force"] = max(contact_log["max_touch_force"], tool_force)
            if step % sample_every == 0:
                trace.append(self._sample(model, data, label))
            if self._render_callback is not None and step % self._render_every == 0:
                self._render_callback(model, data, label)

    def run_skill(self, config: ArmSkillConfig | Mapping[str, Any] | str) -> ArmSkillResult:
        if isinstance(config, str):
            cfg = ArmSkillConfig(skill_id=config)
        elif isinstance(config, Mapping):
            cfg = ArmSkillConfig(**config)
        else:
            cfg = config

        if cfg.skill_id == "reach_touch":
            return self._run_reach_touch(cfg)
        if cfg.skill_id == "button_press":
            return self._run_button_press(cfg)
        if cfg.skill_id == "pick_lift":
            return self._run_pick_lift(cfg)
        if cfg.skill_id in {"contact_sweep", "push_sweep"}:
            return self._run_contact_sweep(cfg)
        if cfg.skill_id == "tool_contact_sweep":
            return self._run_tool_contact_sweep(cfg)
        if cfg.skill_id == "peg_insert":
            return self._run_peg_insert(cfg)
        supported = ", ".join(ARM_SKILLS)
        raise ValueError(f"Unknown arm skill '{cfg.skill_id}'. Supported skills: {supported}")

    def _prepare(self, cfg: ArmSkillConfig):
        model, data = self.build(cfg.object_id, held_tool=(cfg.tool_id or None))
        rest_z = self._object_rest_z(cfg.object_id)
        self._set_cube_friction(model, cfg.friction)
        trace: List[Dict[str, Any]] = []
        contact_log = {"contact_steps": 0, "max_touch_force": 0.0}
        self._place_cube(model, data, (0.18, -0.24), z=rest_z)
        self._drive(model, data, FR3_PREGRASP, OPEN_GRIPPER, 350, "pregrasp", trace, contact_log)
        self._place_cube(model, data, cfg.object_xy, z=rest_z)
        trace.append(self._sample(model, data, "object-ready"))
        contact_log["contact_steps"] = 0
        contact_log["max_touch_force"] = 0.0
        return model, data, trace, contact_log

    def _finish(
        self,
        cfg: ArmSkillConfig,
        model,
        data,
        trace: List[Dict[str, Any]],
        contact_log: Dict[str, Any],
        start_pos: np.ndarray,
        success: bool,
        failure_type: str,
    ) -> ArmSkillResult:
        cube = self._body_pos(model, data, "cube_body")
        grip = self._gripper_center(model, data)
        displacement = float(np.linalg.norm(cube[:2] - start_pos[:2]))
        lifted_height = float(cube[2] - self._object_rest_z(cfg.object_id))
        return ArmSkillResult(
            skill_id=cfg.skill_id,
            robot=self.robot,
            source=self.source,
            object_id=cfg.object_id,
            tool_id=cfg.tool_id,
            success=bool(success),
            failure_type=failure_type,
            contact_steps=int(contact_log["contact_steps"]),
            max_touch_force=round(float(contact_log["max_touch_force"]), 3),
            final_object_pos=_rounded(cube, 4),
            end_effector_pos=_rounded(grip, 4),
            object_displacement=round(displacement, 4),
            lifted_height=round(lifted_height, 4),
            tool_contact_steps=int(contact_log.get("tool_contact_steps", 0)),
            trace=trace,
        )

    def _run_reach_touch(self, cfg: ArmSkillConfig) -> ArmSkillResult:
        model, data, trace, contact_log = self._prepare(cfg)
        start_pos = self._body_pos(model, data, "cube_body")
        target = list(FR3_GRASP)
        target[1] += cfg.grasp_height_delta
        self._drive(model, data, target, OPEN_GRIPPER, 250, "reach", trace, contact_log)
        self._drive(model, data, target, CLOSED_GRIPPER, 360, "touch-close", trace, contact_log)
        success = contact_log["contact_steps"] > 20
        failure = "none" if success else "no_contact"
        return self._finish(cfg, model, data, trace, contact_log, start_pos, success, failure)

    def _run_pick_lift(self, cfg: ArmSkillConfig) -> ArmSkillResult:
        model, data, trace, contact_log = self._prepare(cfg)
        start_pos = self._body_pos(model, data, "cube_body")
        target = list(FR3_GRASP)
        target[1] += cfg.grasp_height_delta
        self._drive(model, data, target, OPEN_GRIPPER, 250, "reach", trace, contact_log)
        self._drive(model, data, target, CLOSED_GRIPPER, 650, "grasp", trace, contact_log)
        self._drive(model, data, FR3_LIFT, CLOSED_GRIPPER, 650, "lift", trace, contact_log)
        cube = self._body_pos(model, data, "cube_body")
        lifted = float(cube[2] - self._object_rest_z(cfg.object_id))
        success = lifted > 0.045 and contact_log["max_touch_force"] > 0.01
        failure = "none" if success else ("grasp_miss" if contact_log["contact_steps"] < 20 else "lift_failed")
        return self._finish(cfg, model, data, trace, contact_log, start_pos, success, failure)

    def _run_contact_sweep(self, cfg: ArmSkillConfig) -> ArmSkillResult:
        model, data, trace, contact_log = self._prepare(cfg)
        start_pos = self._body_pos(model, data, "cube_body")
        self._drive(model, data, FR3_GRASP, OPEN_GRIPPER, 300, "reach", trace, contact_log)
        self._drive(model, data, FR3_GRASP, CLOSED_GRIPPER, 280, "make-contact", trace, contact_log)
        sweep = list(FR3_SWEEP)
        sweep[1] = FR3_GRASP[1] + (FR3_SWEEP[1] - FR3_GRASP[1]) * cfg.sweep_scale
        sweep[3] = FR3_GRASP[3] + (FR3_SWEEP[3] - FR3_GRASP[3]) * cfg.sweep_scale
        self._drive(model, data, sweep, CLOSED_GRIPPER, 850, "sweep", trace, contact_log)
        cube = self._body_pos(model, data, "cube_body")
        displacement = float(np.linalg.norm(cube[:2] - start_pos[:2]))
        success = displacement > 0.015 and contact_log["contact_steps"] > 10
        failure = "none" if success else ("no_contact" if contact_log["contact_steps"] <= 10 else "weak_displacement")
        return self._finish(cfg, model, data, trace, contact_log, start_pos, success, failure)

    def _run_tool_contact_sweep(self, cfg: ArmSkillConfig) -> ArmSkillResult:
        tool_cfg = cfg if cfg.tool_id else ArmSkillConfig(**(asdict(cfg) | {"tool_id": "hammer"}))
        model, data, trace, contact_log = self._prepare(tool_cfg)
        start_pos = self._body_pos(model, data, "cube_body")
        self._drive(model, data, FR3_GRASP, OPEN_GRIPPER, 300, "tool-approach", trace, contact_log)
        sweep = list(FR3_SWEEP)
        sweep[1] = FR3_GRASP[1] + (FR3_SWEEP[1] - FR3_GRASP[1]) * tool_cfg.sweep_scale
        sweep[3] = FR3_GRASP[3] + (FR3_SWEEP[3] - FR3_GRASP[3]) * tool_cfg.sweep_scale
        for _ in range(4):
            self._drive(model, data, sweep, OPEN_GRIPPER, 160, "tool-sweep", trace, contact_log)
        cube = self._body_pos(model, data, "cube_body")
        displacement = float(np.linalg.norm(cube[:2] - start_pos[:2]))
        success = displacement > 0.012 and (
            contact_log.get("tool_contact_steps", 0) > 0 or contact_log["contact_steps"] > 10
        )
        failure = "none" if success else ("no_tool_contact" if contact_log.get("tool_contact_steps", 0) <= 0 else "weak_displacement")
        return self._finish(tool_cfg, model, data, trace, contact_log, start_pos, success, failure)

    def _run_peg_insert(self, cfg: ArmSkillConfig) -> ArmSkillResult:
        peg_cfg = ArmSkillConfig(
            "peg_insert",
            object_id=cfg.object_id or "insertion_socket",
            tool_id="peg",
            object_xy=cfg.object_xy,
            friction=cfg.friction,
            sweep_scale=cfg.sweep_scale,
        )
        result = self._run_tool_contact_sweep(peg_cfg)
        result.skill_id = "peg_insert"
        if result.tool_contact_steps > 80 and result.object_displacement < 0.075:
            result.success = True
            result.failure_type = "none"
        elif result.tool_contact_steps <= 80:
            result.success = False
            result.failure_type = "no_contact"
        else:
            result.success = False
            result.failure_type = "fixture_shift"
        return result

    def _run_button_press(self, cfg: ArmSkillConfig) -> ArmSkillResult:
        press_cfg = ArmSkillConfig(
            "button_press",
            object_id=cfg.object_id or "button_target",
            tool_id="peg",
            object_xy=cfg.object_xy,
            friction=cfg.friction,
            sweep_scale=cfg.sweep_scale or 0.45,
        )
        result = self._run_tool_contact_sweep(press_cfg)
        result.skill_id = "button_press"
        if result.tool_contact_steps > 80 and result.max_touch_force > 1.0:
            result.success = True
            result.failure_type = "none"
        elif result.tool_contact_steps <= 80:
            result.success = False
            result.failure_type = "no_contact"
        else:
            result.success = False
            result.failure_type = "weak_press"
        return result

    def render_skill_replay(
        self,
        config: ArmSkillConfig | Mapping[str, Any] | str = "pick_lift",
        width: int = 560,
        height: int = 315,
        frame_count: int = 40,
        output_dir: Path | None = None,
        web_prefix: str | None = None,
    ) -> Dict[str, Any]:
        captured: List[tuple[np.ndarray, str]] = []
        renderer = None
        camera = self.mujoco.MjvCamera()
        camera.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.68, 0.0, 0.56)
        camera.distance = 0.86
        camera.azimuth = 68
        camera.elevation = -18

        def capture(model, data, label: str) -> None:
            nonlocal renderer
            if renderer is None:
                renderer = self.mujoco.Renderer(model, height=height, width=width)
            renderer.update_scene(data, camera=camera)
            captured.append((renderer.render().copy(), label))

        previous_callback = self._render_callback
        previous_every = self._render_every
        self._render_callback = capture
        self._render_every = 28
        try:
            result = self.run_skill(config)
        finally:
            self._render_callback = previous_callback
            self._render_every = previous_every
            if renderer is not None:
                renderer.close()

        selected = _select_evenly(captured, frame_count)
        images: List[str] = []
        labels: List[str] = []

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            for old in output_dir.glob("frame_*.jpg"):
                old.unlink()

        for idx, (frame, label) in enumerate(selected):
            labels.append(label)
            if output_dir is None:
                images.append(_encode_jpeg(frame))
            else:
                name = f"frame_{idx:03d}.jpg"
                Image.fromarray(frame).save(output_dir / name, format="JPEG", quality=80, optimize=True)
                images.append(f"{web_prefix.rstrip('/')}/{name}" if web_prefix else str(output_dir / name))

        return {
            "source": self.source,
            "model": "MuJoCo FR3 + Franka Hand",
            "title": f"FR3 arm skill: {result.skill_id}",
            "robot": self.robot,
            "result": asdict(result) | {"trace": []},
            "image_frames": images,
            "labels": labels,
            "width": width,
            "height": height,
        }


def list_arm_skill_specs() -> List[Dict[str, Any]]:
    return [{"skill_id": skill_id, **spec} for skill_id, spec in ARM_SKILLS.items()]


def _encode_jpeg(frame: np.ndarray) -> str:
    image = Image.fromarray(frame)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=76, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _select_evenly(items: List[tuple[np.ndarray, str]], count: int) -> List[tuple[np.ndarray, str]]:
    if len(items) <= count:
        return items
    return [items[round(i * (len(items) - 1) / max(1, count - 1))] for i in range(count)]


def run_default_arm_skill_suite() -> Dict[str, Any]:
    runner = FR3ArmSkillRunner()
    configs = [
        ArmSkillConfig("reach_touch", object_id="cube_5cm", object_xy=(0.59, 0.0)),
        ArmSkillConfig("button_press", object_id="button_target", object_xy=(0.59, 0.0)),
        ArmSkillConfig("pick_lift", object_id="rect_block", object_xy=(0.59, 0.0)),
        ArmSkillConfig("contact_sweep", object_id="cube_5cm", object_xy=(0.59, 0.0), sweep_scale=0.45),
        ArmSkillConfig("tool_contact_sweep", tool_id="hammer", object_id="rect_block", object_xy=(0.59, 0.0), sweep_scale=0.45),
        ArmSkillConfig("peg_insert", object_id="insertion_socket", object_xy=(0.59, 0.0), sweep_scale=0.45),
    ]
    results = [asdict(runner.run_skill(config)) for config in configs]
    return {
        "ok": all(row["success"] for row in results),
        "robot": runner.robot,
        "source": runner.source,
        "skills": results,
        "note": "These are scripted FR3 arm primitives, intended as reusable runner foundations rather than final high-fidelity task policies.",
    }
