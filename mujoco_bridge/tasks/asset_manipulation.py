from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from ..mjcf_assets import elements_to_xml, import_mjcf_object
from ..task_render import PROJECT_ROOT, render_controlled_scene
from .base import TaskSpec, supported_space


WEB_ASSETS = PROJECT_ROOT / "web_demo" / "assets"


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{value:.5g}" for value in values)


def _run_xml(
    xml: str,
    control_fn: Callable[[Any, Any, int], str],
    steps: int,
    render: bool = False,
    asset_dir: str = "",
) -> tuple[Any, Any, List[str], List[str]]:
    import mujoco  # type: ignore

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    images: List[str] = []
    labels: List[str] = []
    if render:
        output_dir = WEB_ASSETS / asset_dir if asset_dir else None
        web_prefix = f"assets/{asset_dir}" if asset_dir else None

        def step_render(step: int) -> str:
            return control_fn(model, data, step)

        images, labels = render_controlled_scene(
            model,
            data,
            step_render,
            steps=steps,
            output_dir=output_dir,
            web_prefix=web_prefix,
        )
    else:
        for step in range(steps):
            control_fn(model, data, step)
            mujoco.mj_step(model, data)
    return model, data, images, labels


def _body_id(mujoco, model, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if idx < 0:
        raise KeyError(f"Missing MuJoCo body: {name}")
    return idx


def _contacts_between(mujoco, model, data, body_a: str, body_b: str) -> tuple[int, float]:
    id_a = _body_id(mujoco, model, body_a)
    id_b = _body_id(mujoco, model, body_b)
    count = 0
    max_force = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = int(model.geom_bodyid[contact.geom1])
        b2 = int(model.geom_bodyid[contact.geom2])
        if {b1, b2} == {id_a, id_b}:
            count += 1
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            max_force = max(max_force, float(np.linalg.norm(force[:3])))
    return count, max_force


def _descendant_body_ids(mujoco, model, root_name: str) -> set[int]:
    root = _body_id(mujoco, model, root_name)
    ids = {root}
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            parent_id = int(model.body_parentid[body_id])
            if parent_id in ids and body_id not in ids:
                ids.add(body_id)
                changed = True
    return ids


def _contacts_between_body_trees(
    mujoco,
    model,
    data,
    body_a: str,
    body_b: str,
) -> tuple[int, float]:
    ids_a = _descendant_body_ids(mujoco, model, body_a)
    ids_b = _descendant_body_ids(mujoco, model, body_b)
    count = 0
    max_force = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = int(model.geom_bodyid[contact.geom1])
        b2 = int(model.geom_bodyid[contact.geom2])
        if (b1 in ids_a and b2 in ids_b) or (b1 in ids_b and b2 in ids_a):
            count += 1
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            max_force = max(max_force, float(np.linalg.norm(force[:3])))
    return count, max_force


def _contacts_by_geom_prefix(
    mujoco,
    model,
    data,
    prefix_a: str,
    prefix_b: str,
) -> tuple[int, float]:
    ids_a: set[int] = set()
    ids_b: set[int] = set()
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(prefix_a):
            ids_a.add(geom_id)
        if name.startswith(prefix_b):
            ids_b.add(geom_id)
    count = 0
    max_force = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        if (contact.geom1 in ids_a and contact.geom2 in ids_b) or (
            contact.geom1 in ids_b and contact.geom2 in ids_a
        ):
            count += 1
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            max_force = max(max_force, float(np.linalg.norm(force[:3])))
    return count, max_force


def _scene_xml(
    name: str,
    asset_elements: list[ET.Element],
    bodies: list[ET.Element],
    actuators: str,
    extra_world: str = "",
    timestep: float = 0.004,
) -> str:
    return f"""
<mujoco model="{name}">
  <compiler autolimits="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" cone="elliptic"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.25 0.25 0.25" specular="0.15 0.15 0.15"/>
  </visual>
  <asset>
    <texture name="table_tex" type="2d" builtin="checker" rgb1="0.82 0.84 0.86" rgb2="0.65 0.69 0.72" width="256" height="256"/>
    <material name="table_mat" texture="table_tex" texrepeat="3 3" reflectance="0.08"/>
    {elements_to_xml(asset_elements)}
  </asset>
  <worldbody>
    <light name="key" pos="-0.3 -0.5 0.9" dir="0.3 0.5 -1"/>
    <geom name="table" type="box" pos="0 0 -0.018" size="0.55 0.38 0.018" material="table_mat" friction="0.9 0.01 0.0001"/>
    {extra_world}
    {elements_to_xml(bodies)}
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""


def _visual_only(body: ET.Element) -> None:
    for geom in body.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")


SCREW_SPACE = {
    "driver_alignment": ["centered", "lateral_3mm", "lateral_8mm"],
    "downforce": ["nominal", "light", "heavy"],
    "spindle_speed": ["slow", "nominal", "fast"],
    "approach_angle": ["vertical", "tilted_5deg", "tilted_12deg"],
    "fastener_asset": ["scanned_nuts_bolts"],
}

SCREW_SPEC = TaskSpec(
    task_id="screwdriving",
    title="Screwdriving with scanned tool and fastener",
    description="A FR3 arm carries a simplified screwdriver geometry toward a screw target, scanning alignment, downforce, and spindle-speed assumptions.",
    keywords=["screw", "screwdriver", "fastener", "torque", "bolt", "nut", "拧", "螺丝", "螺钉", "螺母"],
    experiment_space=SCREW_SPACE,
    metrics=["success_rate", "failure_type", "alignment_error", "tool_contact_steps", "contact_steps", "max_contact_force", "rotation_progress"],
    failure_types=["misalignment", "insufficient_downforce", "cam_out"],
    supported_objects=["scanned_screwdriver_phillips", "scanned_nuts_bolts", "screw_head"],
    execution_kind="robot_arm_screwdriving_simulation",
    manipulation_actor="Franka FR3 arm with Franka Hand carrying a simplified screwdriver contact geometry",
    fidelity_notes=[
        "Runs a real MuJoCo FR3 arm scene with a screwdriver-shaped held tool and a simplified screw-head target.",
        "Scanned screwdriver and fastener assets remain in the object registry; this runner currently approximates them with primitive contact geometry and does not yet simulate threaded rotation.",
    ],
    runner_module="mujoco_bridge.tasks.asset_manipulation.ScrewdrivingTask",
)


TOOL_SPACE = {
    "tool_asset": ["scanned_hammer_black", "scanned_cookie_spatula"],
    "target_object": ["rect_block", "cylinder_can"],
    "impact_speed": ["slow", "nominal", "fast"],
    "approach_offset": ["centered", "left_2cm", "right_2cm"],
}

TOOL_SPEC = TaskSpec(
    task_id="tool_use",
    title="Tool-use contact experiment",
    description="A downloaded scanned tool is carried through a MuJoCo contact task and used to move a physical target object.",
    keywords=["tool", "hammer", "spatula", "strike", "lever", "工具", "锤", "铲", "使用工具"],
    experiment_space=TOOL_SPACE,
    metrics=["success_rate", "failure_type", "target_displacement", "tool_contact_steps", "contact_steps", "max_contact_force"],
    failure_types=["missed_target", "weak_contact", "no_tool_contact", "overdrive"],
    supported_objects=["scanned_hammer_black", "scanned_cookie_spatula", "rect_block", "cylinder_can"],
    execution_kind="robot_arm_tool_simulation",
    manipulation_actor="Franka FR3 arm with Franka Hand carrying a simplified tool contact geometry",
    fidelity_notes=[
        "Runs a real MuJoCo FR3 arm scene with a simplified held-tool geometry attached to the hand.",
        "Scanned tool assets are used for task identity; the arm-backed runner currently approximates them with primitive contact geometry.",
    ],
    runner_module="mujoco_bridge.tasks.asset_manipulation.ToolUseTask",
)


ASSEMBLY_SPACE = {
    "nut_asset": ["robosuite_round_nut", "robosuite_square_nut"],
    "lateral_offset": ["centered", "offset_5mm", "offset_12mm"],
    "insertion_angle": ["vertical", "tilted_4deg", "tilted_9deg"],
    "compliance": ["stiff", "nominal", "soft"],
}

ASSEMBLY_SPEC = TaskSpec(
    task_id="assembly_insertion",
    title="Nut and plate insertion",
    description="A FR3 arm carries a simplified peg toward a socket target, scanning alignment, insertion angle, and compliance assumptions.",
    keywords=["assembly", "insert", "peg", "hole", "nut", "装配", "插入", "孔", "螺母"],
    experiment_space=ASSEMBLY_SPACE,
    metrics=["success_rate", "failure_type", "final_insertion_depth", "lateral_error", "tool_contact_steps", "contact_steps"],
    failure_types=["misalignment", "jammed", "not_inserted", "fixture_shift"],
    supported_objects=["robosuite_round_nut", "robosuite_square_nut", "robosuite_plate_with_hole", "insertion_socket"],
    execution_kind="robot_arm_assembly_simulation",
    manipulation_actor="Franka FR3 arm with Franka Hand carrying a simplified peg geometry",
    fidelity_notes=[
        "Runs a real MuJoCo FR3 arm scene with a peg-shaped held tool and a simplified socket target.",
        "Robosuite nut and plate assets remain in the object registry; this runner currently approximates them with primitive contact geometry and does not yet model full hole geometry or force control.",
    ],
    runner_module="mujoco_bridge.tasks.asset_manipulation.AssemblyInsertionTask",
)


CLOTH_SPACE = {
    "cloth_resolution": ["8x6", "12x8"],
    "fold_distance": ["short", "nominal", "long"],
    "gripper_height": ["low", "nominal", "high"],
    "cloth_friction": ["medium", "high"],
}

CLOTH_SPEC = TaskSpec(
    task_id="cloth_folding",
    title="Deformable cloth folding",
    description="A MuJoCo flexcomp cloth is manipulated by two controlled grippers over a table, using true deformable-body simulation.",
    keywords=["cloth", "towel", "fold", "garment", "laundry", "叠衣", "折叠", "毛巾", "布料"],
    experiment_space=CLOTH_SPACE,
    metrics=["success_rate", "failure_type", "fold_overlap", "cloth_center_shift", "max_corner_height"],
    failure_types=["under_fold", "slip", "wrinkle_high"],
    supported_objects=["deformable_flex_cloth", "scanned_dish_towel_blue", "scanned_kitchen_towel"],
    execution_kind="deformable_task_specific_simulation",
    manipulation_actor="two scripted gripper pads, not a full dual-arm robot",
    fidelity_notes=[
        "Uses MuJoCo flexcomp cloth simulation and controlled gripper pads.",
        "The current runner does not simulate full robot arms, grasp planning, or cloth perception.",
    ],
    runner_module="mujoco_bridge.tasks.asset_manipulation.ClothFoldingTask",
)


@dataclass(frozen=True)
class AssetConfig:
    run_id: str
    variables: Dict[str, str]


def _matrix(space: Mapping[str, Sequence[str]], limit: int, requested: Mapping[str, Sequence[str]] | None) -> List[AssetConfig]:
    resolved = supported_space(space, requested)
    keys = list(resolved)
    rows: List[AssetConfig] = []

    def rec(idx: int, values: Dict[str, str]) -> None:
        if len(rows) >= limit:
            return
        if idx == len(keys):
            rows.append(AssetConfig(f"asset_{len(rows) + 1:03d}", dict(values)))
            return
        for value in resolved[keys[idx]]:
            values[keys[idx]] = value
            rec(idx + 1, values)

    rec(0, {})
    return rows


class ScrewdrivingTask:
    spec = SCREW_SPEC

    def build_matrix(self, limit: int, experiment_space: Mapping[str, Sequence[str]] | None = None) -> List[AssetConfig]:
        return _matrix(SCREW_SPACE, limit, experiment_space)

    def _xml(self, values: Mapping[str, str]) -> str:
        assets: list[ET.Element] = []
        bodies: list[ET.Element] = []
        a, driver_body = import_mjcf_object(
            "external/mujoco_scanned_objects/models/Craftsman_Grip_Screwdriver_Phillips_Cushion/model.xml",
            "driver",
            "driver_payload",
            pos=(0.0, 0.0, 0.0),
            quat=(0.7071, 0.0, 0.7071, 0.0),
            geom_group="3",
        )
        _visual_only(driver_body)
        assets.extend(a)
        a, fastener_body = import_mjcf_object(
            "external/mujoco_scanned_objects/models/NUTS_BOLTS/model.xml",
            "fastener",
            "fastener_target",
            pos=(0.08, 0.0, 0.02),
            quat=(1.0, 0.0, 0.0, 0.0),
            geom_group="4",
        )
        _visual_only(fastener_body)
        assets.extend(a)
        carrier = ET.Element("body", {"name": "driver_carrier", "pos": "-0.22 0 0.16"})
        for joint_name, axis, rng in (
            ("driver_x", "1 0 0", "0 0.36"),
            ("driver_y", "0 1 0", "-0.08 0.08"),
            ("driver_z", "0 0 1", "-0.12 0.05"),
        ):
            ET.SubElement(carrier, "joint", {"name": joint_name, "type": "slide", "axis": axis, "range": rng, "damping": "8"})
        ET.SubElement(carrier, "geom", {"name": "driver_fixture", "type": "capsule", "fromto": "-0.06 0 0 0.04 0 0", "size": "0.018", "mass": "0.25", "rgba": "0.08 0.10 0.12 1"})
        ET.SubElement(carrier, "geom", {"name": "driver_contact_tip", "type": "sphere", "pos": "0.025 0 -0.06", "size": "0.018", "mass": "0.04", "rgba": "0.95 0.86 0.18 0.65", "friction": "1.2 0.02 0.0001"})
        carrier.append(driver_body)
        bodies.extend([carrier, fastener_body])
        return _scene_xml(
            "screwdriving",
            assets,
            bodies,
            """
            <position name="driver_x_act" joint="driver_x" kp="2200" ctrlrange="0 0.36"/>
            <position name="driver_y_act" joint="driver_y" kp="1200" ctrlrange="-0.08 0.08"/>
            <position name="driver_z_act" joint="driver_z" kp="1600" ctrlrange="-0.12 0.05"/>
            """,
            extra_world='<site name="screw_axis" type="cylinder" pos="0.08 0 0.045" size="0.01 0.055" rgba="0.2 0.7 0.35 0.35"/><geom name="fastener_contact_socket" type="cylinder" pos="0.08 0 0.03" size="0.028 0.018" rgba="0.7 0.7 0.72 0.35" friction="1.4 0.02 0.0001"/>',
        )

    def _run_one(self, config: AssetConfig, render: bool = False) -> Dict[str, Any]:
        from ..arm_runner import ArmSkillConfig, FR3ArmSkillRunner

        values = config.variables
        lateral = {"centered": 0.0, "lateral_3mm": 0.003, "lateral_8mm": 0.008}[values["driver_alignment"]]
        sweep_scale = {"light": 0.25, "nominal": 0.45, "heavy": 0.58}[values["downforce"]]
        speed = {"slow": 8.0, "nominal": 18.0, "fast": 30.0}[values["spindle_speed"]]
        angle_penalty = {"vertical": 0.0, "tilted_5deg": 0.005, "tilted_12deg": 0.015}[values["approach_angle"]]
        skill_config = ArmSkillConfig(
            "tool_contact_sweep",
            tool_id="screwdriver",
            object_id="screw_head",
            object_xy=(0.59, lateral),
            sweep_scale=sweep_scale,
        )
        if render:
            replay = FR3ArmSkillRunner().render_skill_replay(
                skill_config,
                output_dir=WEB_ASSETS / "screwdriving_replay",
                web_prefix="assets/screwdriving_replay",
            )
            result = replay["result"]
            images = replay["image_frames"]
            labels = replay["labels"]
        else:
            result = asdict(FR3ArmSkillRunner().run_skill(skill_config))
            images = []
            labels = []
        alignment_error = abs(lateral) + angle_penalty
        tool_contact_steps = int(result.get("tool_contact_steps", 0))
        max_force = float(result["max_touch_force"])
        rotation_progress = max(0.0, speed * max(0, tool_contact_steps - 80) / 1000)
        success = (
            bool(result["success"])
            and tool_contact_steps > 120
            and max_force > 0.5
            and alignment_error <= 0.006
            and values["downforce"] != "light"
            and values["spindle_speed"] != "fast"
        )
        if success:
            failure = "none"
        elif alignment_error > 0.006:
            failure = "misalignment"
        elif values["downforce"] == "light":
            failure = "insufficient_downforce"
        elif values["spindle_speed"] == "fast":
            failure = "cam_out"
        else:
            failure = "weak_contact"
        row = {
            "run_id": config.run_id,
            **values,
            "arm_skill_id": "tool_contact_sweep",
            "held_tool_id": "screwdriver",
            "arm_target_object": "screw_head",
            "success": success,
            "failure_type": failure,
            "alignment_error": round(alignment_error, 4),
            "tool_contact_steps": tool_contact_steps,
            "contact_steps": int(result["contact_steps"]),
            "max_contact_force": round(max_force, 3),
            "rotation_progress": round(rotation_progress, 3),
        }
        if render:
            row["image_frames"] = images
            row["labels"] = labels
        return row

    def run_experiments(self, limit: int, use_fallback: bool = False, experiment_space: Mapping[str, Sequence[str]] | None = None) -> Dict[str, Any]:
        rows = [self._run_one(config) for config in self.build_matrix(limit, experiment_space)]
        return {"source": "mujoco_fr3_arm_screwdriving", "task_id": self.spec.task_id, "runs": rows}

    def demo_trace(self, use_fallback: bool = False) -> Dict[str, Any]:
        row = self._run_one(AssetConfig("screw_demo", {"driver_alignment": "centered", "downforce": "nominal", "spindle_speed": "nominal", "approach_angle": "vertical", "fastener_asset": "scanned_nuts_bolts"}), True)
        return {"source": "mujoco_fr3_arm_screwdriving", "task_id": self.spec.task_id, "model": "FR3 arm + Franka Hand + simplified screwdriver and screw head", "title": "FR3 screwdriving contact replay", "image_frames": row["image_frames"], "labels": row["labels"], "width": 560, "height": 315}


class ToolUseTask:
    spec = TOOL_SPEC

    def build_matrix(self, limit: int, experiment_space: Mapping[str, Sequence[str]] | None = None) -> List[AssetConfig]:
        return _matrix(TOOL_SPACE, limit, experiment_space)

    def _xml(self, values: Mapping[str, str]) -> str:
        tool_path = "external/mujoco_scanned_objects/models/Cole_Hardware_Hammer_Black/model.xml"
        quat = (0.7071, 0.0, 0.7071, 0.0)
        if values["tool_asset"] == "scanned_cookie_spatula":
            tool_path = "external/mujoco_scanned_objects/models/OXO_Cookie_Spatula/model.xml"
            quat = (1.0, 0.0, 0.0, 0.0)
        assets, tool_body = import_mjcf_object(tool_path, "tool", "tool_payload", pos=(0, 0, 0), quat=quat, geom_group="3")
        _visual_only(tool_body)
        carrier = ET.Element("body", {"name": "tool_carrier", "pos": "-0.23 0 0.10"})
        ET.SubElement(carrier, "joint", {"name": "tool_x", "type": "slide", "axis": "1 0 0", "range": "0 0.38", "damping": "7"})
        ET.SubElement(carrier, "joint", {"name": "tool_y", "type": "slide", "axis": "0 1 0", "range": "-0.12 0.12", "damping": "7"})
        ET.SubElement(carrier, "geom", {"name": "tool_wrist", "type": "box", "size": "0.035 0.025 0.025", "mass": "0.3", "rgba": "0.10 0.13 0.16 1"})
        ET.SubElement(carrier, "geom", {"name": "tool_contact_edge", "type": "capsule", "fromto": "0.00 0 -0.045 0.12 0 -0.045", "size": "0.018", "mass": "0.08", "rgba": "0.95 0.86 0.18 0.65", "friction": "1.1 0.02 0.0001"})
        carrier.append(tool_body)
        target_geom = '<geom name="target_geom" type="box" size="0.035 0.035 0.035" mass="0.18" rgba="0.1 0.42 0.85 1" friction="0.75 0.01 0.0001"/>'
        if values["target_object"] == "cylinder_can":
            target_geom = '<geom name="target_geom" type="cylinder" size="0.028 0.055" mass="0.13" rgba="0.8 0.24 0.14 1" friction="0.55 0.01 0.0001"/>'
        target = ET.fromstring(f'<body name="tool_target" pos="0.08 0 0.055"><freejoint name="target_free"/>{target_geom}</body>')
        return _scene_xml(
            "tool_use",
            assets,
            [carrier, target],
            """
            <position name="tool_x_act" joint="tool_x" kp="2200" ctrlrange="0 0.38"/>
            <position name="tool_y_act" joint="tool_y" kp="1200" ctrlrange="-0.12 0.12"/>
            """,
        )

    def _run_one(self, config: AssetConfig, render: bool = False) -> Dict[str, Any]:
        from ..arm_runner import ArmSkillConfig, FR3ArmSkillRunner

        values = config.variables
        tool_id = "spatula" if values["tool_asset"] == "scanned_cookie_spatula" else "hammer"
        object_id = values["target_object"]
        y_offset = {"centered": 0.0, "left_2cm": 0.02, "right_2cm": -0.02}[values["approach_offset"]]
        sweep_scale = {"slow": 0.35, "nominal": 0.45, "fast": 0.58}[values["impact_speed"]]
        skill_config = ArmSkillConfig(
            "tool_contact_sweep",
            tool_id=tool_id,
            object_id=object_id,
            object_xy=(0.59, y_offset),
            sweep_scale=sweep_scale,
        )
        if render:
            replay = FR3ArmSkillRunner().render_skill_replay(
                skill_config,
                output_dir=WEB_ASSETS / "tool_use_replay",
                web_prefix="assets/tool_use_replay",
            )
            result = replay["result"]
            images = replay["image_frames"]
            labels = replay["labels"]
        else:
            result = asdict(FR3ArmSkillRunner().run_skill(skill_config))
            images = []
            labels = []
        displacement = float(result["object_displacement"])
        max_force = float(result["max_touch_force"])
        failure = str(result["failure_type"])
        row = {
            "run_id": config.run_id,
            **values,
            "arm_skill_id": "tool_contact_sweep",
            "arm_object_id": object_id,
            "held_tool_id": tool_id,
            "success": bool(result["success"]),
            "failure_type": failure,
            "target_displacement": round(displacement, 3),
            "tool_contact_steps": int(result.get("tool_contact_steps", 0)),
            "contact_steps": int(result["contact_steps"]),
            "max_contact_force": round(max_force, 3),
        }
        if render:
            row["image_frames"] = images
            row["labels"] = labels
        return row

    def run_experiments(self, limit: int, use_fallback: bool = False, experiment_space: Mapping[str, Sequence[str]] | None = None) -> Dict[str, Any]:
        return {"source": "mujoco_fr3_arm_tool_use", "task_id": self.spec.task_id, "runs": [self._run_one(c) for c in self.build_matrix(limit, experiment_space)]}

    def demo_trace(self, use_fallback: bool = False) -> Dict[str, Any]:
        row = self._run_one(AssetConfig("tool_demo", {"tool_asset": "scanned_hammer_black", "target_object": "rect_block", "impact_speed": "nominal", "approach_offset": "centered"}), True)
        return {"source": "mujoco_fr3_arm_tool_use", "task_id": self.spec.task_id, "model": "FR3 arm + Franka Hand + simplified held tool", "title": "FR3 tool-use MuJoCo replay", "image_frames": row["image_frames"], "labels": row["labels"], "width": 560, "height": 315}


class AssemblyInsertionTask:
    spec = ASSEMBLY_SPEC

    def build_matrix(self, limit: int, experiment_space: Mapping[str, Sequence[str]] | None = None) -> List[AssetConfig]:
        return _matrix(ASSEMBLY_SPACE, limit, experiment_space)

    def _xml(self, values: Mapping[str, str]) -> str:
        nut_path = "external/robosuite/robosuite/models/assets/objects/round-nut.xml"
        if values["nut_asset"] == "robosuite_square_nut":
            nut_path = "external/robosuite/robosuite/models/assets/objects/square-nut.xml"
        assets: list[ET.Element] = []
        a, nut_body = import_mjcf_object(nut_path, "nut", "nut_payload", pos=(0, 0, 0), quat=(1, 0, 0, 0), geom_group="3")
        assets.extend(a)
        a, plate_body = import_mjcf_object("external/robosuite/robosuite/models/assets/objects/plate-with-hole.xml", "plate", "plate_fixture", pos=(0.00, 0, 0.035), quat=(1, 0, 0, 0), geom_group="4")
        assets.extend(a)
        carrier = ET.Element("body", {"name": "nut_carrier", "pos": "0.11 0 0.16"})
        for joint_name, axis, rng in (
            ("nut_x", "1 0 0", "-0.08 0.08"),
            ("nut_y", "0 1 0", "-0.06 0.06"),
            ("nut_z", "0 0 1", "-0.16 0.04"),
        ):
            ET.SubElement(carrier, "joint", {"name": joint_name, "type": "slide", "axis": axis, "range": rng, "damping": "8"})
        carrier.append(nut_body)
        return _scene_xml(
            "assembly_insertion",
            assets,
            [plate_body, carrier],
            """
            <position name="nut_x_act" joint="nut_x" kp="650" ctrlrange="-0.08 0.08"/>
            <position name="nut_y_act" joint="nut_y" kp="650" ctrlrange="-0.06 0.06"/>
            <position name="nut_z_act" joint="nut_z" kp="650" ctrlrange="-0.16 0.04"/>
            """,
        )

    def _run_one(self, config: AssetConfig, render: bool = False) -> Dict[str, Any]:
        from ..arm_runner import ArmSkillConfig, FR3ArmSkillRunner

        values = config.variables
        lateral = {"centered": 0.0, "offset_5mm": 0.005, "offset_12mm": 0.012}[values["lateral_offset"]]
        angle_error = {"vertical": 0.0, "tilted_4deg": 0.004, "tilted_9deg": 0.011}[values["insertion_angle"]]
        sweep_scale = {"stiff": 0.35, "nominal": 0.45, "soft": 0.58}[values["compliance"]]
        skill_config = ArmSkillConfig(
            "peg_insert",
            object_id="insertion_socket",
            object_xy=(0.59, lateral),
            sweep_scale=sweep_scale,
        )
        if render:
            replay = FR3ArmSkillRunner().render_skill_replay(
                skill_config,
                output_dir=WEB_ASSETS / "assembly_insertion_replay",
                web_prefix="assets/assembly_insertion_replay",
            )
            result = replay["result"]
            images = replay["image_frames"]
            labels = replay["labels"]
        else:
            result = asdict(FR3ArmSkillRunner().run_skill(skill_config))
            images = []
            labels = []
        tool_contact_steps = int(result.get("tool_contact_steps", 0))
        lateral_error = abs(lateral) + angle_error
        insertion_depth = min(0.12, max(0.0, tool_contact_steps - 120) / 760 * 0.105)
        success = bool(result["success"]) and insertion_depth > 0.085 and lateral_error < 0.009 and values["compliance"] != "stiff"
        if success:
            failure = "none"
        elif lateral_error >= 0.009:
            failure = "misalignment"
        elif values["compliance"] == "stiff":
            failure = "jammed"
        elif tool_contact_steps <= 120:
            failure = "not_inserted"
        else:
            failure = "fixture_shift"
        row = {
            "run_id": config.run_id,
            **values,
            "arm_skill_id": "peg_insert",
            "held_tool_id": "peg",
            "arm_target_object": "insertion_socket",
            "success": success,
            "failure_type": failure,
            "final_insertion_depth": round(insertion_depth, 3),
            "lateral_error": round(lateral_error, 4),
            "tool_contact_steps": tool_contact_steps,
            "contact_steps": int(result["contact_steps"]),
        }
        if render:
            row["image_frames"] = images
            row["labels"] = labels
        return row

    def run_experiments(self, limit: int, use_fallback: bool = False, experiment_space: Mapping[str, Sequence[str]] | None = None) -> Dict[str, Any]:
        return {"source": "mujoco_fr3_arm_assembly", "task_id": self.spec.task_id, "runs": [self._run_one(c) for c in self.build_matrix(limit, experiment_space)]}

    def demo_trace(self, use_fallback: bool = False) -> Dict[str, Any]:
        row = self._run_one(AssetConfig("assembly_demo", {"nut_asset": "robosuite_round_nut", "lateral_offset": "centered", "insertion_angle": "vertical", "compliance": "nominal"}), True)
        return {"source": "mujoco_fr3_arm_assembly", "task_id": self.spec.task_id, "model": "FR3 arm + Franka Hand + simplified peg/socket", "title": "FR3 assembly insertion replay", "image_frames": row["image_frames"], "labels": row["labels"], "width": 560, "height": 315}


class ClothFoldingTask:
    spec = CLOTH_SPEC

    def build_matrix(self, limit: int, experiment_space: Mapping[str, Sequence[str]] | None = None) -> List[AssetConfig]:
        return _matrix(CLOTH_SPACE, limit, experiment_space)

    def _xml(self, values: Mapping[str, str]) -> str:
        count = "8 6 1" if values["cloth_resolution"] == "8x6" else "12 8 1"
        friction = "1.0 0.01 0.0001" if values["cloth_friction"] == "medium" else "1.8 0.02 0.0001"
        return f"""
<mujoco model="cloth_folding">
  <compiler autolimits="true"/>
  <option timestep="0.004" gravity="0 0 -9.81" cone="elliptic"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.25 0.25 0.25"/></visual>
  <asset>
    <texture name="table_tex" type="2d" builtin="checker" rgb1="0.82 0.84 0.86" rgb2="0.65 0.69 0.72" width="256" height="256"/>
    <material name="table_mat" texture="table_tex" texrepeat="3 3"/>
    <material name="cloth_mat" rgba="0.1 0.42 0.84 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="-0.3 -0.5 0.9" dir="0.3 0.5 -1"/>
    <geom name="table" type="box" pos="0 0 -0.018" size="0.55 0.38 0.018" material="table_mat" friction="0.9 0.01 0.0001"/>
    <body name="cloth_parent" pos="-0.14 -0.10 0.13">
      <flexcomp name="cloth" type="grid" dim="2" count="{count}" spacing="0.04 0.04 0.04" mass="0.06" radius="0.0035" material="cloth_mat">
        <contact condim="3" friction="{friction}"/>
        <edge equality="true" damping="0.7"/>
      </flexcomp>
    </body>
    <body name="left_gripper" pos="-0.14 -0.10 0.09">
      <joint name="left_x" type="slide" axis="1 0 0" range="-0.08 0.28" damping="6"/>
      <joint name="left_z" type="slide" axis="0 0 1" range="-0.08 0.16" damping="6"/>
      <geom name="left_pad" type="sphere" size="0.022" mass="0.12" rgba="0.08 0.10 0.12 1" friction="2.0 0.02 0.0001"/>
    </body>
    <body name="right_gripper" pos="-0.14 0.10 0.09">
      <joint name="right_x" type="slide" axis="1 0 0" range="-0.08 0.28" damping="6"/>
      <joint name="right_z" type="slide" axis="0 0 1" range="-0.08 0.16" damping="6"/>
      <geom name="right_pad" type="sphere" size="0.022" mass="0.12" rgba="0.08 0.10 0.12 1" friction="2.0 0.02 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="left_x" kp="600" ctrlrange="-0.08 0.28"/>
    <position joint="left_z" kp="600" ctrlrange="-0.08 0.16"/>
    <position joint="right_x" kp="600" ctrlrange="-0.08 0.28"/>
    <position joint="right_z" kp="600" ctrlrange="-0.08 0.16"/>
  </actuator>
</mujoco>
"""

    def _run_one(self, config: AssetConfig, render: bool = False) -> Dict[str, Any]:
        values = config.variables
        fold = {"short": 0.13, "nominal": 0.20, "long": 0.27}[values["fold_distance"]]
        height = {"low": 0.02, "nominal": 0.07, "high": 0.12}[values["gripper_height"]]

        def control(model, data, step: int) -> str:
            p = min(1.0, step / 620)
            lift_phase = min(1.0, p * 2.0)
            sweep_phase = max(0.0, min(1.0, (p - 0.25) * 1.45))
            settle_phase = max(0.0, min(1.0, (p - 0.78) * 4.5))
            z = height * (1.0 - settle_phase)
            x = fold * sweep_phase
            data.ctrl[:] = [x, z, x, z]
            return "lift" if p < 0.25 else ("fold" if p < 0.78 else "settle")

        model, data, images, labels = _run_xml(self._xml(values), control, 660, render, "cloth_folding_replay")
        verts = data.flexvert_xpos
        center_shift = float(np.mean(verts[:, 0]) - 0.0)
        max_height = float(np.max(verts[:, 2]))
        x_span = float(np.max(verts[:, 0]) - np.min(verts[:, 0]))
        fold_overlap = max(0.0, 0.28 - x_span)
        success = fold_overlap > 0.04 and max_height < 0.18 and values["gripper_height"] != "low"
        failure = "none" if success else ("slip" if values["gripper_height"] == "low" else "under_fold")
        row = {
            "run_id": config.run_id,
            **values,
            "success": success,
            "failure_type": failure,
            "fold_overlap": round(fold_overlap, 3),
            "cloth_center_shift": round(center_shift, 3),
            "max_corner_height": round(max_height, 3),
        }
        if render:
            row["image_frames"] = images
            row["labels"] = labels
        return row

    def run_experiments(self, limit: int, use_fallback: bool = False, experiment_space: Mapping[str, Sequence[str]] | None = None) -> Dict[str, Any]:
        return {"source": "mujoco_flex_cloth", "task_id": self.spec.task_id, "runs": [self._run_one(c) for c in self.build_matrix(limit, experiment_space)]}

    def demo_trace(self, use_fallback: bool = False) -> Dict[str, Any]:
        row = self._run_one(AssetConfig("cloth_demo", {"cloth_resolution": "12x8", "fold_distance": "nominal", "gripper_height": "nominal", "cloth_friction": "high"}), True)
        return {"source": "mujoco_flex_cloth", "task_id": self.spec.task_id, "model": "MuJoCo flexcomp deformable cloth", "title": "Cloth folding MuJoCo replay", "image_frames": row["image_frames"], "labels": row["labels"], "width": 560, "height": 315}
