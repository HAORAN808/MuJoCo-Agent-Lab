"""Dynamic MuJoCo scene composer.

Assembles valid MuJoCo scenes from robot specifications, workspace
descriptions, and object definitions. Supports all menagerie robot arms
and the project's object library.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from .object_library import ObjectSpec, get_object
from .robot_registry import RobotSpec, get_robot


# ---------------------------------------------------------------------------
# Scene description (LLM output)
# ---------------------------------------------------------------------------

@dataclass
class ObjectPlacement:
    """Describes where to place an object in the scene."""
    object_id: str
    role: str = "target"  # "target", "obstacle", "tool"
    position: Tuple[float, float, float] = (0.5, 0.0, 0.4)
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class SceneDescription:
    """What the LLM produces to describe a scene."""
    robot_id: str
    objects: List[ObjectPlacement] = field(default_factory=list)
    workspace: str = "table"  # "table", "shelf", "floor"
    workspace_params: Dict[str, Any] = field(default_factory=dict)
    gripper_id: Optional[str] = None  # for arm-only robots
    held_tool_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Workspace templates
# ---------------------------------------------------------------------------

TABLE_HEIGHT = 0.37  # default table height
TABLE_SIZE = (0.4, 0.6, 0.02)  # half-sizes
TABLE_TOP_Z = TABLE_HEIGHT + TABLE_SIZE[2]

TABLE_XML_TEMPLATE = """
    <body name="table" pos="{tx} {ty} {tz}">
      <geom name="table_top" type="box" size="{sx} {sy} {sz}"
            rgba="0.6 0.5 0.4 1" mass="50"
            contype="1" conaffinity="1"/>
      <geom name="table_leg1" type="cylinder" size="0.02 0.18"
            pos="{lx1} {ly1} -0.19" rgba="0.5 0.4 0.3 1"
            contype="1" conaffinity="1"/>
      <geom name="table_leg2" type="cylinder" size="0.02 0.18"
            pos="{lx2} {ly2} -0.19" rgba="0.5 0.4 0.3 1"
            contype="1" conaffinity="1"/>
      <geom name="table_leg3" type="cylinder" size="0.02 0.18"
            pos="{lx3} {ly3} -0.19" rgba="0.5 0.4 0.3 1"
            contype="1" conaffinity="1"/>
      <geom name="table_leg4" type="cylinder" size="0.02 0.18"
            pos="{lx4} {ly4} -0.19" rgba="0.5 0.4 0.3 1"
            contype="1" conaffinity="1"/>
    </body>
"""

FLOOR_XML = """
    <body name="floor" pos="0 0 0">
      <geom name="floor_geom" type="plane" size="2 2 0.01"
            rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1"/>
    </body>
"""


# ---------------------------------------------------------------------------
# Object geometry (from object_library)
# ---------------------------------------------------------------------------

def _object_geom_xml(obj: ObjectSpec, pos: Tuple[float, float, float]) -> str:
    """Generate XML for an object from the library."""
    x, y, z = pos

    if obj.geometry == "box":
        sx, sy, sz = obj.size_m[0], obj.size_m[1], obj.size_m[2]
        size_str = f"{sx} {sy} {sz}"
        type_str = "box"
        half_z = sz
    elif obj.geometry == "cylinder":
        r, h = obj.size_m[0], obj.size_m[1]
        size_str = f"{r} {h}"
        type_str = "cylinder"
        half_z = h
    elif obj.geometry == "sphere":
        r = obj.size_m[0]
        size_str = f"{r}"
        type_str = "sphere"
        half_z = r
    else:
        # Fallback to cube
        s = 0.025
        size_str = f"{s} {s} {s}"
        type_str = "box"
        half_z = s
    z = max(float(z), TABLE_TOP_Z + float(half_z) + 0.002)

    friction = obj.friction
    friction_str = f"{friction[0]} {friction[1]} {friction[2]}"

    return f"""
    <body name="obj_{obj.object_id}" pos="{x} {y} {z}">
      <freejoint name="obj_{obj.object_id}_joint"/>
      <geom name="obj_{obj.object_id}_geom" type="{type_str}" size="{size_str}"
            mass="{obj.mass_kg}" friction="{friction_str}"
            condim="6" contype="7" conaffinity="7"/>
    </body>
"""


# ---------------------------------------------------------------------------
# Held tool XML (from scene_builder pattern)
# ---------------------------------------------------------------------------

_TOOL_TEMPLATES = {
    "spatula": """
    <body name="held_tool" pos="0 0 0.112">
      <geom name="tool_handle" type="capsule" size="0.008 0.06"
            pos="0 0 -0.05" rgba="0.3 0.3 0.3 1"
            mass="0.05" contype="8" conaffinity="7"/>
      <geom name="tool_head" type="box" size="0.025 0.04 0.004"
            pos="0 0.03 -0.1" rgba="0.7 0.7 0.7 1"
            mass="0.03" contype="8" conaffinity="7"/>
    </body>
""",
    "screwdriver": """
    <body name="held_tool" pos="0 0 0.112">
      <geom name="tool_handle" type="capsule" size="0.01 0.07"
            pos="0 0 -0.05" rgba="0.2 0.2 0.6 1"
            mass="0.08" contype="8" conaffinity="7"/>
      <geom name="tool_shaft" type="cylinder" size="0.003 0.04"
            pos="0 0 -0.12" rgba="0.7 0.7 0.7 1"
            mass="0.05" contype="8" conaffinity="7"/>
    </body>
""",
    "hammer": """
    <body name="held_tool" pos="0 0 0.112">
      <geom name="tool_handle" type="capsule" size="0.01 0.08"
            pos="0 0 -0.05" rgba="0.4 0.25 0.1 1"
            mass="0.1" contype="8" conaffinity="7"/>
      <geom name="tool_head" type="box" size="0.02 0.025 0.02"
            pos="0 0 -0.12" rgba="0.3 0.3 0.3 1"
            mass="0.2" contype="8" conaffinity="7"/>
    </body>
""",
    "peg": """
    <body name="held_tool" pos="0 0 0.112">
      <geom name="tool_shaft" type="cylinder" size="0.006 0.06"
            pos="0 0 -0.06" rgba="0.8 0.6 0.2 1"
            mass="0.06" contype="8" conaffinity="7"/>
    </body>
""",
}


# ---------------------------------------------------------------------------
# Gripper XML templates for arm-only robots
# ---------------------------------------------------------------------------

def _franka_hand_xml() -> str:
    """Franka Hand XML for FR3/Panda attachment."""
    return """
    <body name="hand" pos="0 0 0.107" quat="0.923880 0 0 0.382683">
      <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
      <geom name="hand_vis_0" mesh="hand_0" material="off_white" type="mesh" contype="0" conaffinity="0" group="2"/>
      <geom name="hand_vis_1" mesh="hand_1" material="black" type="mesh" contype="0" conaffinity="0" group="2"/>
      <geom name="hand_vis_2" mesh="hand_2" material="black" type="mesh" contype="0" conaffinity="0" group="2"/>
      <geom name="hand_vis_3" mesh="hand_3" material="white" type="mesh" contype="0" conaffinity="0" group="2"/>
      <geom name="hand_vis_4" mesh="hand_4" material="off_white" type="mesh" contype="0" conaffinity="0" group="2"/>
      <geom name="hand_col" mesh="hand_c" type="mesh" group="3"/>
      <site name="pinch_site" pos="0 0 0.105" size="0.008" rgba="0.1 0.9 0.7 0.7"/>
      <body name="left_finger" pos="0 0 0.0584">
        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04" armature="0.1" damping="1"/>
        <geom name="hand_left_finger_vis_0" mesh="finger_0" material="off_white" type="mesh" contype="0" conaffinity="0" group="2"/>
        <geom name="hand_left_finger_vis_1" mesh="finger_1" material="black" type="mesh" contype="0" conaffinity="0" group="2"/>
        <geom name="hand_left_finger_col" mesh="finger_0" type="mesh" group="3" contype="4" conaffinity="7" condim="6" friction="2.0 0.005 0.0001"/>
        <geom name="hand_left_pad_1" type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445" contype="4" conaffinity="7" condim="6" friction="2.0 0.005 0.0001"/>
        <geom name="hand_left_pad_2" type="box" size="0.003 0.002 0.003" pos="0.0055 0.002 0.05" contype="4" conaffinity="7" condim="6" friction="2.0 0.005 0.0001"/>
        <geom name="hand_left_pad_3" type="box" size="0.003 0.002 0.003" pos="-0.0055 0.002 0.05" contype="4" conaffinity="7" condim="6" friction="2.0 0.005 0.0001"/>
        <site name="left_finger_touch" pos="0 0.0055 0.0445" size="0.012 0.005 0.012"/>
      </body>
      <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
        <joint name="finger_joint2" type="slide" axis="0 1 0" range="0 0.04" armature="0.1" damping="1"/>
        <geom name="hand_right_finger_vis_0" mesh="finger_0" material="off_white" type="mesh" contype="0" conaffinity="0" group="2"/>
        <geom name="hand_right_finger_vis_1" mesh="finger_1" material="black" type="mesh" contype="0" conaffinity="0" group="2"/>
        <geom name="hand_right_finger_col" mesh="finger_0" type="mesh" group="3" contype="4" conaffinity="7" condim="6" friction="2.0 0.005 0.0001"/>
        <geom name="hand_right_pad_1" type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445" contype="4" conaffinity="7" condim="6" friction="2.0 0.005 0.0001"/>
        <geom name="hand_right_pad_2" type="box" size="0.003 0.002 0.003" pos="0.0055 0.002 0.05" contype="4" conaffinity="7" condim="6" friction="2.0 0.005 0.0001"/>
        <geom name="hand_right_pad_3" type="box" size="0.003 0.002 0.003" pos="-0.0055 0.002 0.05" contype="4" conaffinity="7" condim="6" friction="2.0 0.005 0.0001"/>
        <site name="right_finger_touch" pos="0 0.0055 0.0445" size="0.012 0.005 0.012"/>
      </body>
    </body>
"""


def _franka_hand_asset_xml() -> str:
    """Mesh/material assets for the Franka Hand copied from the bundled assets."""
    return """
    <material name="off_white" rgba="0.901961 0.921569 0.929412 1"/>
    <mesh name="hand_c" file="hand.stl"/>
    <mesh name="hand_0" file="hand_0.obj"/>
    <mesh name="hand_1" file="hand_1.obj"/>
    <mesh name="hand_2" file="hand_2.obj"/>
    <mesh name="hand_3" file="hand_3.obj"/>
    <mesh name="hand_4" file="hand_4.obj"/>
    <mesh name="finger_0" file="finger_0.obj"/>
    <mesh name="finger_1" file="finger_1.obj"/>
"""


def _franka_hand_contact_xml() -> str:
    return """
  <contact>
    <exclude body1="hand" body2="left_finger"/>
    <exclude body1="hand" body2="right_finger"/>
  </contact>
"""


def _franka_hand_tendon_xml() -> str:
    return """
  <tendon>
    <fixed name="split">
      <joint joint="finger_joint1" coef="0.5"/>
      <joint joint="finger_joint2" coef="0.5"/>
    </fixed>
  </tendon>
"""


def _franka_hand_equality_xml() -> str:
    return """
  <equality>
    <joint joint1="finger_joint1" joint2="finger_joint2" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>
"""


# ---------------------------------------------------------------------------
# Scene composer
# ---------------------------------------------------------------------------

class DynamicSceneComposer:
    """Composes MuJoCo scenes from robot specs and scene descriptions."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self._menagerie_root = self.project_root / "external" / "mujoco_menagerie"
        self._franka_hand_assets_dir = self.project_root / "mujoco_bridge" / "assets" / "franka_hand" / "assets"

    def _uses_auto_franka_hand(self, spec: RobotSpec, scene_desc: SceneDescription) -> bool:
        """FR3 is arm-only in menagerie, so this demo mounts the bundled Franka Hand."""
        return spec.robot_id == "franka_fr3" and (scene_desc.gripper_id in {None, "", "franka_hand"})

    def compose(
        self,
        scene_desc: SceneDescription,
        load_model: bool = True,
    ) -> Dict[str, Any]:
        """Compose a scene and optionally load the MuJoCo model.

        Returns dict with keys: xml, assets, model, data, object_body_ids,
        robot_spec.
        """
        spec = get_robot(scene_desc.robot_id)

        # Build the scene XML
        xml = self._build_xml(spec, scene_desc)

        # Load mesh assets
        assets = self._load_assets(spec, scene_desc)

        # Filter assets to only those referenced in the XML
        filtered_assets = self._filter_referenced_assets(xml, assets)

        result: Dict[str, Any] = {
            "xml": xml,
            "assets": filtered_assets,
            "robot_spec": spec,
            "object_body_ids": {},
        }

        if load_model:
            model, data = self._load_mujoco_model(xml, filtered_assets, spec)
            mujoco.mj_forward(model, data)

            # Resolve object body IDs
            for obj_placement in scene_desc.objects:
                body_name = f"obj_{obj_placement.object_id}"
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                result["object_body_ids"][obj_placement.object_id] = bid

            result["model"] = model
            result["data"] = data

        return result

    def _build_xml(self, spec: RobotSpec, scene_desc: SceneDescription) -> str:
        """Build the complete scene XML string."""
        # Start with a MuJoCo root
        parts = ['<mujoco model="dynamic_scene">']
        parts.append('  <compiler angle="radian" meshdir="assets"/>')
        parts.append('  <option integrator="implicitfast" timestep="0.002"/>')

        # Include robot's default definitions
        robot_defaults = self._extract_robot_defaults(spec)
        parts.append(robot_defaults)

        # Assets from robot MJCF (meshes, textures, materials)
        robot_assets_xml = self._extract_robot_assets(spec)
        if robot_assets_xml:
            parts.append(robot_assets_xml)
        else:
            parts.append('  <asset/>')

        # Worldbody
        parts.append('  <worldbody>')

        # Light
        parts.append('    <light name="top_light" pos="0 0 2.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>')
        parts.append('    <light name="side_light" pos="1 1 1.5" dir="-0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>')

        # Ground plane
        parts.append(FLOOR_XML)

        # Workspace
        table_pos = scene_desc.workspace_params.get("position", [0.5, 0.0, TABLE_HEIGHT])
        if scene_desc.workspace == "table":
            parts.append(self._table_xml(table_pos))

        # Robot body (inline from robot MJCF)
        robot_xml = self._extract_robot_body_xml(spec)
        if self._uses_auto_franka_hand(spec, scene_desc):
            robot_xml = self._attach_franka_hand_xml(robot_xml)
        if scene_desc.held_tool_id:
            robot_xml = self._attach_held_tool_xml(robot_xml, spec, scene_desc.held_tool_id)
        parts.append(robot_xml)

        # Objects
        for obj_placement in scene_desc.objects:
            obj = get_object(obj_placement.object_id)
            parts.append(_object_geom_xml(obj, obj_placement.position))

        parts.append('  </worldbody>')

        if self._uses_auto_franka_hand(spec, scene_desc):
            parts.append(_franka_hand_contact_xml())
            parts.append(_franka_hand_tendon_xml())
            parts.append(_franka_hand_equality_xml())

        # Sensors
        parts.append('  <sensor>')
        for obj_placement in scene_desc.objects:
            parts.append(f'    <framepos name="obj_{obj_placement.object_id}_pos" objtype="body" objname="obj_{obj_placement.object_id}"/>')
        if self._uses_auto_franka_hand(spec, scene_desc):
            parts.append('    <touch name="left_touch" site="left_finger_touch"/>')
            parts.append('    <touch name="right_touch" site="right_finger_touch"/>')
            parts.append('    <jointpos name="finger1_pos" joint="finger_joint1"/>')
            parts.append('    <jointpos name="finger2_pos" joint="finger_joint2"/>')
        parts.append('  </sensor>')

        # Actuators (from robot spec)
        parts.append(self._actuators_xml(spec, scene_desc))

        # Keyframe (arm joints only)
        if spec.has_keyframe and spec.keyframe_qpos:
            n_arm = spec.dof
            arm_qpos = spec.keyframe_qpos[:n_arm]
            qpos_str = " ".join(str(q) for q in arm_qpos)
            ctrl_str = qpos_str
            parts.append(f'  <keyframe>')
            parts.append(f'    <key name="home" qpos="{qpos_str}" ctrl="{ctrl_str}"/>')
            parts.append(f'  </keyframe>')

        parts.append('</mujoco>')
        return "\n".join(parts)

    def _table_xml(self, pos: Sequence[float]) -> str:
        """Generate table XML."""
        x, y, z = pos
        sx, sy, sz = TABLE_SIZE
        lx1, ly1 = x - sx + 0.05, y - sy + 0.05
        lx2, ly2 = x + sx - 0.05, y - sy + 0.05
        lx3, ly3 = x - sx + 0.05, y + sy - 0.05
        lx4, ly4 = x + sx - 0.05, y + sy - 0.05
        return TABLE_XML_TEMPLATE.format(
            tx=x, ty=y, tz=z,
            sx=sx, sy=sy, sz=sz,
            lx1=lx1, ly1=ly1,
            lx2=lx2, ly2=ly2,
            lx3=lx3, ly3=ly3,
            lx4=lx4, ly4=ly4,
        )

    def _extract_robot_body_xml(self, spec: RobotSpec) -> str:
        """Extract the robot body tree from the menagerie MJCF.

        This parses the robot XML and extracts just the kinematic chain
        (bodies, joints, geoms, sites) for embedding in the scene XML.
        """
        mjcf_path = Path(spec.mjcf_path)
        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        # Resolve includes
        self._resolve_includes(root, mjcf_path.parent)

        # Extract worldbody content
        worldbody = root.find(".//worldbody")
        if worldbody is None:
            raise ValueError(f"No worldbody in {spec.mjcf_path}")

        # Add base position (robot sits on table or floor)
        base_pos = self._robot_base_pos(spec)

        # Build the robot body wrapper
        lines = [f'    <body name="robot_base" pos="{base_pos[0]} {base_pos[1]} {base_pos[2]}">']

        # Inline all bodies from the robot's worldbody
        for child in worldbody:
            lines.extend(self._element_to_lines(child, indent=3))

        lines.append('    </body>')
        return "\n".join(lines)

    def _attach_held_tool_xml(self, robot_xml: str, spec: RobotSpec, tool_id: str) -> str:
        """Attach a simple held tool to the robot end-effector body."""
        tool_xml = _TOOL_TEMPLATES.get(tool_id)
        if not tool_xml:
            return robot_xml

        try:
            wrapper = ET.fromstring(f"<root>\n{robot_xml}\n</root>")
            tool_elem = ET.fromstring(tool_xml)
        except ET.ParseError:
            return robot_xml

        target_body = None
        if spec.end_effector_body:
            for body in wrapper.iter("body"):
                if body.get("name") == spec.end_effector_body:
                    target_body = body
                    break

        if target_body is None:
            bodies = list(wrapper.iter("body"))
            target_body = bodies[-1] if bodies else None
        if target_body is None:
            return robot_xml

        target_body.append(tool_elem)

        lines: List[str] = []
        for child in wrapper:
            lines.extend(self._element_to_lines(child, indent=2))
        return "\n".join(lines)

    def _attach_franka_hand_xml(self, robot_xml: str) -> str:
        """Attach the bundled Franka Hand to the FR3 attachment site body."""
        try:
            wrapper = ET.fromstring(f"<root>\n{robot_xml}\n</root>")
            hand_elem = ET.fromstring(_franka_hand_xml())
        except ET.ParseError:
            return robot_xml

        target_body = None
        for body in wrapper.iter("body"):
            for site in body.findall("site"):
                if site.get("name") == "attachment_site":
                    target_body = body
                    break
            if target_body is not None:
                break

        if target_body is None:
            for body in wrapper.iter("body"):
                if body.get("name") == "fr3_link7":
                    target_body = body
                    break
        if target_body is None:
            return robot_xml

        if not any(child.get("name") == "hand" for child in target_body.findall("body")):
            target_body.append(hand_elem)

        lines: List[str] = []
        for child in wrapper:
            lines.extend(self._element_to_lines(child, indent=2))
        return "\n".join(lines)

    def _extract_robot_defaults(self, spec: RobotSpec) -> str:
        """Extract <default> definitions from the robot MJCF."""
        mjcf_path = Path(spec.mjcf_path)
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        self._resolve_includes(root, mjcf_path.parent)

        defaults = root.find("default")
        if defaults is None:
            return "  <default/>"

        lines = ["  <default>"]
        for child in defaults:
            lines.extend(self._element_to_lines(child, indent=2))
        lines.append("  </default>")
        return "\n".join(lines)

    def _extract_robot_assets(self, spec: RobotSpec) -> str:
        """Extract <asset> definitions from the robot MJCF."""
        mjcf_path = Path(spec.mjcf_path)
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        self._resolve_includes(root, mjcf_path.parent)

        assets = root.find("asset")
        if assets is None:
            return ""

        lines = ["  <asset>"]
        for child in assets:
            lines.extend(self._element_to_lines(child, indent=2))
        if spec.robot_id == "franka_fr3":
            hand_root = ET.fromstring(f"<root>{_franka_hand_asset_xml()}</root>")
            for child in hand_root:
                lines.extend(self._element_to_lines(child, indent=2))
        lines.append("  </asset>")
        return "\n".join(lines)

    def _robot_base_pos(self, spec: RobotSpec) -> Tuple[float, float, float]:
        """Compute robot base position based on workspace."""
        # Default: robot base at origin, table in front
        return (0.0, 0.0, 0.0)

    def _resolve_includes(self, root: ET.Element, base_dir: Path) -> None:
        """Recursively inline <include> elements."""
        includes = root.findall(".//include")
        for inc in includes:
            file_attr = inc.get("file", "")
            inc_path = base_dir / file_attr
            if not inc_path.exists():
                continue
            inc_tree = ET.parse(inc_path)
            inc_root = inc_tree.getroot()
            self._resolve_includes(inc_root, inc_path.parent)

            parent_map = {c: p for p in root.iter() for c in p}
            parent = parent_map.get(inc)
            if parent is None:
                idx = list(root).index(inc)
                root.remove(inc)
                for i, child in enumerate(inc_root):
                    root.insert(idx + i, child)
            else:
                idx = list(parent).index(inc)
                parent.remove(inc)
                for i, child in enumerate(inc_root):
                    parent.insert(idx + i, child)

    def _element_to_lines(self, elem: ET.Element, indent: int = 2) -> List[str]:
        """Convert an XML element to indented string lines."""
        prefix = "  " * indent
        tag = elem.tag
        attrs = " ".join(f'{k}="{v}"' for k, v in elem.attrib.items())

        children = list(elem)
        if not children and elem.text is None:
            return [f"{prefix}<{tag} {attrs}/>"]

        lines = [f"{prefix}<{tag} {attrs}>"]
        for child in children:
            lines.extend(self._element_to_lines(child, indent + 1))
        lines.append(f"{prefix}</{tag}>")
        return lines

    def _actuators_xml(self, spec: RobotSpec, scene_desc: SceneDescription) -> str:
        """Extract actuator XML from the robot's original MJCF."""
        mjcf_path = Path(spec.mjcf_path)
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        self._resolve_includes(root, mjcf_path.parent)

        actuators = root.find("actuator")
        if actuators is None:
            return "  <actuator/>"

        lines = ["  <actuator>"]
        for child in actuators:
            lines.extend(self._element_to_lines(child, indent=2))
        if self._uses_auto_franka_hand(spec, scene_desc):
            lines.append('    <general name="actuator8" tendon="split" forcerange="-100 100" ctrlrange="0 255"')
            lines.append('             gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>')
        lines.append("  </actuator>")

        # Also extract tendons if present
        tendons = root.find("tendon")
        if tendons is not None:
            lines.append("  <tendon>")
            for child in tendons:
                lines.extend(self._element_to_lines(child, indent=2))
            lines.append("  </tendon>")

        # Extract equality constraints
        equality = root.find("equality")
        if equality is not None:
            lines.append("  <equality>")
            for child in equality:
                lines.extend(self._element_to_lines(child, indent=2))
            lines.append("  </equality>")

        return "\n".join(lines)

    @staticmethod
    def _load_mujoco_model(
        xml: str, assets: Dict[str, bytes], spec: RobotSpec
    ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Load MuJoCo model, falling back to temp file for complex assets."""
        import tempfile

        try:
            model = mujoco.MjModel.from_xml_string(xml, assets)
            data = mujoco.MjData(model)
            return model, data
        except Exception:
            pass

        # Fallback: write XML and assets to temp dir, load from path
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            xml_path = tmpdir / "scene.xml"
            xml_path.write_text(xml, encoding="utf-8")

            # Write asset files
            assets_dir = tmpdir / "assets"
            assets_dir.mkdir(exist_ok=True)
            for key, data_bytes in assets.items():
                asset_path = assets_dir / key
                asset_path.parent.mkdir(parents=True, exist_ok=True)
                asset_path.write_bytes(data_bytes)

            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            return model, data

    @staticmethod
    def _filter_referenced_assets(xml: str, assets: Dict[str, bytes]) -> Dict[str, bytes]:
        """Filter assets dict to only include files referenced in the XML."""
        import re
        # Find all file="..." attributes in the XML
        referenced = set(re.findall(r'file="([^"]+)"', xml))
        # Also check for file='...'
        referenced.update(re.findall(r"file='([^']+)'", xml))

        filtered: Dict[str, bytes] = {}
        for key, value in assets.items():
            # Normalize path separators
            normalized = key.replace("\\", "/")
            if normalized in referenced:
                filtered[normalized] = value

        return filtered

    def _load_assets(self, spec: RobotSpec, scene_desc: SceneDescription) -> Dict[str, bytes]:
        """Load mesh assets from robot and object directories."""
        assets: Dict[str, bytes] = {}

        # Robot mesh assets (recursive, handle subdirs like visual/ collision/)
        robot_assets = Path(spec.assets_dir)
        if robot_assets.exists():
            for f in robot_assets.rglob("*"):
                if f.is_file() and f.suffix.lower() in ('.stl', '.obj', '.png', '.jpg'):
                    # Use relative path from assets dir as key
                    rel = f.relative_to(robot_assets)
                    key = str(rel).replace("\\", "/")  # normalize path separators
                    assets[key] = f.read_bytes()
        if self._uses_auto_franka_hand(spec, scene_desc) and self._franka_hand_assets_dir.exists():
            for f in self._franka_hand_assets_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ('.stl', '.obj', '.png', '.jpg'):
                    assets[f.name] = f.read_bytes()

        return assets


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compose_scene(
    robot_id: str,
    objects: Optional[List[Dict[str, Any]]] = None,
    workspace: str = "table",
    **kwargs,
) -> Dict[str, Any]:
    """Quick scene composition helper.

    Parameters
    ----------
    robot_id : str
        ID of the robot to use.
    objects : list of dict, optional
        Each dict has keys: object_id, role, position.
    workspace : str
        Workspace type.

    Returns
    -------
    dict
        With keys: model, data, xml, assets, robot_spec, object_body_ids.
    """
    placements = []
    for obj in (objects or []):
        placements.append(ObjectPlacement(
            object_id=obj["object_id"],
            role=obj.get("role", "target"),
            position=tuple(obj.get("position", [0.5, 0.0, 0.4])),
        ))

    scene_desc = SceneDescription(
        robot_id=robot_id,
        objects=placements,
        workspace=workspace,
        **kwargs,
    )

    composer = DynamicSceneComposer()
    return composer.compose(scene_desc)
