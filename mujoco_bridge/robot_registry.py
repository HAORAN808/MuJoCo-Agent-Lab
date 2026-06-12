"""Universal robot abstraction layer for MuJoCo menagerie arms.

Parses arbitrary menagerie robot MJCF files and extracts a uniform
``RobotSpec`` dataclass that downstream components (IK solver, motion
primitives, scene composer) can consume without knowing robot-specific
XML details.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotSpec:
    """Robot-agnostic specification extracted from MJCF."""

    robot_id: str
    name: str
    mjcf_path: str
    assets_dir: str
    dof: int
    joint_names: Tuple[str, ...]
    joint_ranges: Tuple[Tuple[float, float], ...]
    actuator_names: Tuple[str, ...]
    actuator_type: str  # "position" or "general"
    end_effector_site: str
    end_effector_body: str
    has_gripper: bool
    gripper_joint_names: Tuple[str, ...]
    gripper_actuator_names: Tuple[str, ...]
    gripper_type: str  # "parallel", "none"
    gripper_ctrl_open: float
    gripper_ctrl_closed: float
    base_body: str
    meshdir: str
    has_keyframe: bool
    keyframe_qpos: Optional[Tuple[float, ...]]
    keyframe_ctrl: Optional[Tuple[float, ...]]
    # Extra metadata
    manufacturer: str = ""
    family: str = ""


# ---------------------------------------------------------------------------
# MJCF parser
# ---------------------------------------------------------------------------

class RobotParser:
    """Extracts a ``RobotSpec`` from a menagerie MJCF model file."""

    # Menagerie gripper models that can be attached to arm-only robots
    ATTACHABLE_GRIPPERS: Dict[str, Path] = {}

    def parse(self, robot_id: str, mjcf_path: str | Path) -> RobotSpec:
        mjcf_path = Path(mjcf_path).resolve()
        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir", "assets") if compiler is not None else "assets"
        assets_dir = str(mjcf_path.parent / meshdir)

        # Resolve include directives (menagerie uses <include file="..."/>)
        self._resolve_includes(root, mjcf_path.parent)

        # Parse body tree to find joints and end-effector
        worldbody = root.find(".//worldbody")
        if worldbody is None:
            raise ValueError(f"No <worldbody> found in {mjcf_path}")

        joint_names, joint_ranges, base_body, body_map = self._parse_joints(worldbody)
        ee_site, ee_body = self._find_end_effector(worldbody, body_map)

        # Parse actuators
        actuator_names, actuator_type, gripper_actuator_names = self._parse_actuators(
            root, joint_names
        )

        # Parse gripper
        has_gripper, gripper_joint_names, gripper_type = self._parse_gripper(
            root, joint_names, worldbody
        )

        # Determine gripper ctrl range
        gripper_ctrl_open, gripper_ctrl_closed = self._gripper_ctrl_range(
            root, gripper_actuator_names, gripper_type
        )

        # Parse keyframe
        has_keyframe, kf_qpos, kf_ctrl = self._parse_keyframe(root, len(joint_names) + len(gripper_joint_names))

        dof = len(joint_names)

        return RobotSpec(
            robot_id=robot_id,
            name=self._human_name(robot_id),
            mjcf_path=str(mjcf_path),
            assets_dir=assets_dir,
            dof=dof,
            joint_names=tuple(joint_names),
            joint_ranges=tuple(joint_ranges),
            actuator_names=tuple(actuator_names),
            actuator_type=actuator_type,
            end_effector_site=ee_site,
            end_effector_body=ee_body,
            has_gripper=has_gripper,
            gripper_joint_names=tuple(gripper_joint_names),
            gripper_actuator_names=tuple(gripper_actuator_names),
            gripper_type=gripper_type,
            gripper_ctrl_open=gripper_ctrl_open,
            gripper_ctrl_closed=gripper_ctrl_closed,
            base_body=base_body,
            meshdir=meshdir,
            has_keyframe=has_keyframe,
            keyframe_qpos=kf_qpos,
            keyframe_ctrl=kf_ctrl,
        )

    # -- internal helpers ---------------------------------------------------

    def _resolve_includes(self, root: ET.Element, base_dir: Path) -> None:
        """Recursively inline <include file="..."/> elements."""
        includes = root.findall(".//include")
        for inc in includes:
            file_attr = inc.get("file", "")
            inc_path = base_dir / file_attr
            if not inc_path.exists():
                continue
            inc_tree = ET.parse(inc_path)
            inc_root = inc_tree.getroot()
            # Resolve nested includes first
            self._resolve_includes(inc_root, inc_path.parent)
            # Replace the include element with the parsed content
            parent_map = {c: p for p in root.iter() for c in p}
            parent = parent_map.get(inc)
            if parent is None:
                # include is direct child of root
                idx = list(root).index(inc)
                root.remove(inc)
                for i, child in enumerate(inc_root):
                    root.insert(idx + i, child)
            else:
                idx = list(parent).index(inc)
                parent.remove(inc)
                for i, child in enumerate(inc_root):
                    parent.insert(idx + i, child)

    def _parse_joints(
        self, worldbody: ET.Element
    ) -> Tuple[List[str], List[Tuple[float, float]], str, Dict[str, str]]:
        """Extract arm joint names, ranges, base body, and body->parent map."""
        joint_names: List[str] = []
        joint_ranges: List[Tuple[float, float]] = []
        base_body = "world"
        body_map: Dict[str, str] = {}  # body_name -> parent_body_name

        first_joint_found = False

        for body in worldbody.iter("body"):
            body_name = body.get("name", "")
            for joint in body.findall("joint"):
                jname = joint.get("name", "")
                jrange_str = joint.get("range", "")
                if jrange_str:
                    parts = jrange_str.split()
                    jrange = (float(parts[0]), float(parts[1]))
                else:
                    jrange = (-6.28319, 6.28319)  # default unlimited

                # Skip gripper joints (they have "finger" or "driver" in name)
                if self._is_gripper_joint(jname):
                    continue

                joint_names.append(jname)
                joint_ranges.append(jrange)

                if not first_joint_found:
                    # The parent of the first arm joint is the base
                    # Walk up body_map to find it
                    first_joint_found = True

            # Track body hierarchy
            for child_body in body.findall("body"):
                child_name = child_body.get("name", "")
                if child_name:
                    body_map[child_name] = body_name

        # Find base body (parent of first joint's body)
        if joint_names:
            for body in worldbody.iter("body"):
                for joint in body.findall("joint"):
                    if joint.get("name") == joint_names[0]:
                        base_body = body.get("name", "world")
                        break
                if base_body != "world":
                    break

        return joint_names, joint_ranges, base_body, body_map

    def _is_gripper_joint(self, name: str) -> bool:
        gripper_keywords = [
            "finger", "driver", "knuckle", "gripper",
            "left_finger", "right_finger",
        ]
        name_lower = name.lower()
        return any(kw in name_lower for kw in gripper_keywords)

    def _find_end_effector(
        self, worldbody: ET.Element, body_map: Dict[str, str]
    ) -> Tuple[str, str]:
        """Find the end-effector site and its parent body."""
        # Priority: attachment_site > pinch_site > link_tcp
        ee_candidates = ["attachment_site", "pinch_site", "link_tcp"]

        for body in worldbody.iter("body"):
            for site in body.findall("site"):
                sname = site.get("name", "")
                if sname in ee_candidates:
                    return sname, body.get("name", "")

        # Fallback: look for a body named "hand" (Franka Panda pattern)
        for body in worldbody.iter("body"):
            bname = body.get("name", "")
            if bname == "hand":
                return "", bname

        # Fallback: find the last non-gripper body in the kinematic chain
        all_bodies = set()
        parent_bodies = set()
        for body in worldbody.iter("body"):
            bname = body.get("name", "")
            if bname:
                all_bodies.add(bname)
                for child in body.findall("body"):
                    cname = child.get("name", "")
                    if cname:
                        parent_bodies.add(bname)

        leaf_bodies = all_bodies - parent_bodies
        if leaf_bodies:
            for lb in sorted(leaf_bodies):
                if not self._is_gripper_joint(lb) and "finger" not in lb.lower() and "gripper" not in lb.lower():
                    return "", lb

        return "", ""

    def _parse_actuators(
        self, root: ET.Element, arm_joint_names: List[str]
    ) -> Tuple[List[str], str, List[str]]:
        """Parse actuators, separating arm actuators from gripper actuators."""
        actuator_names: List[str] = []
        gripper_actuator_names: List[str] = []
        actuator_type = "general"
        arm_set = set(arm_joint_names)

        actuators = root.find("actuator")
        if actuators is None:
            return actuator_names, actuator_type, gripper_actuator_names

        for act in actuators:
            aname = act.get("name", "")
            tag = act.tag

            # Determine which joints this actuator controls
            target_joint = act.get("joint", "")
            target_tendon = act.get("tendon", "")

            # Use joint name as actuator name if name is missing (e.g. Lite6)
            if not aname and target_joint:
                aname = target_joint

            if target_joint and target_joint in arm_set:
                actuator_names.append(aname)
                if tag == "position":
                    actuator_type = "position"
            elif target_tendon:
                # Tendon-based actuator is typically gripper
                gripper_actuator_names.append(aname)
            elif target_joint and self._is_gripper_joint(target_joint):
                gripper_actuator_names.append(aname)
            elif target_joint:
                # Might be an arm joint not detected as arm
                actuator_names.append(aname)
                if tag == "position":
                    actuator_type = "position"

        return actuator_names, actuator_type, gripper_actuator_names

    def _parse_gripper(
        self, root: ET.Element, arm_joint_names: List[str], worldbody: ET.Element
    ) -> Tuple[bool, List[str], str]:
        """Detect gripper presence and type."""
        gripper_joints: List[str] = []
        arm_set = set(arm_joint_names)

        for body in worldbody.iter("body"):
            for joint in body.findall("joint"):
                jname = joint.get("name", "")
                if jname and jname not in arm_set and self._is_gripper_joint(jname):
                    gripper_joints.append(jname)

        # Check for tendon-based gripper (Franka Hand, xArm7)
        tendons = root.find("tendon")
        has_tendon_gripper = False
        if tendons is not None:
            for fixed in tendons.findall("fixed"):
                if fixed.get("name", "") == "split":
                    has_tendon_gripper = True

        if gripper_joints:
            return True, gripper_joints, "parallel"

        # Check for actuator-on-tendon gripper without explicit gripper joints
        # (e.g. Panda with split tendon)
        if has_tendon_gripper:
            # Find joints referenced by the tendon
            for fixed in tendons.findall("fixed"):
                if fixed.get("name", "") == "split":
                    for joint_elem in fixed.findall("joint"):
                        jname = joint_elem.get("joint", "")
                        if jname and jname not in arm_set:
                            gripper_joints.append(jname)
            if gripper_joints:
                return True, gripper_joints, "parallel"

        return False, [], "none"

    def _gripper_ctrl_range(
        self,
        root: ET.Element,
        gripper_actuator_names: List[str],
        gripper_type: str,
    ) -> Tuple[float, float]:
        """Determine gripper open/closed ctrl values."""
        if gripper_type == "none":
            return 0.0, 0.0

        actuators = root.find("actuator")
        if actuators is None:
            return 0.0, 255.0

        for act in actuators:
            aname = act.get("name", "")
            if aname in gripper_actuator_names:
                ctrlrange = act.get("ctrlrange", "0 255")
                parts = ctrlrange.split()
                return float(parts[0]), float(parts[1])

        return 0.0, 255.0

    def _parse_keyframe(
        self, root: ET.Element, total_qpos_len: int
    ) -> Tuple[bool, Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
        """Parse the first 'home' keyframe."""
        keyframe = root.find(".//keyframe")
        if keyframe is None:
            return False, None, None

        for key in keyframe.findall("key"):
            kname = key.get("name", "")
            if kname == "home" or not kname:
                qpos_str = key.get("qpos", "")
                ctrl_str = key.get("ctrl", "")
                qpos = tuple(float(x) for x in qpos_str.split()) if qpos_str else None
                ctrl = tuple(float(x) for x in ctrl_str.split()) if ctrl_str else None
                return True, qpos, ctrl

        # Take first keyframe if no "home" found
        first = keyframe.find("key")
        if first is not None:
            qpos_str = first.get("qpos", "")
            ctrl_str = first.get("ctrl", "")
            qpos = tuple(float(x) for x in qpos_str.split()) if qpos_str else None
            ctrl = tuple(float(x) for x in ctrl_str.split()) if ctrl_str else None
            return True, qpos, ctrl

        return False, None, None

    @staticmethod
    def _human_name(robot_id: str) -> str:
        names = {
            "franka_fr3": "Franka FR3",
            "franka_emika_panda": "Franka Emika Panda",
            "universal_robots_ur5e": "Universal Robots UR5e",
            "universal_robots_ur10e": "Universal Robots UR10e",
            "kinova_gen3": "Kinova Gen3",
            "kuka_iiwa_14": "KUKA iiwa 14",
            "ufactory_xarm7": "UFACTORY xArm7",
            "ufactory_lite6": "UFACTORY Lite6",
        }
        return names.get(robot_id, robot_id.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Robot registry — pre-populated from menagerie
# ---------------------------------------------------------------------------

_MENAGERIE_ROOT = Path(__file__).resolve().parent.parent / "external" / "mujoco_menagerie"

_ROBOT_PATHS: Dict[str, str] = {
    "franka_fr3": "franka_fr3/fr3.xml",
    "franka_emika_panda": "franka_emika_panda/panda.xml",
    "universal_robots_ur5e": "universal_robots_ur5e/ur5e.xml",
    "universal_robots_ur10e": "universal_robots_ur10e/ur10e.xml",
    "kinova_gen3": "kinova_gen3/gen3.xml",
    "kuka_iiwa_14": "kuka_iiwa_14/iiwa14.xml",
    "ufactory_xarm7": "ufactory_xarm7/xarm7.xml",
    "ufactory_lite6": "ufactory_lite6/lite6.xml",
}

_MANUFACTURER = {
    "franka_fr3": "Franka Emika",
    "franka_emika_panda": "Franka Emika",
    "universal_robots_ur5e": "Universal Robots",
    "universal_robots_ur10e": "Universal Robots",
    "kinova_gen3": "Kinova",
    "kuka_iiwa_14": "KUKA",
    "ufactory_xarm7": "UFACTORY",
    "ufactory_lite6": "UFACTORY",
}

_FAMILY = {
    "franka_fr3": "franka",
    "franka_emika_panda": "franka",
    "universal_robots_ur5e": "ur",
    "universal_robots_ur10e": "ur",
    "kinova_gen3": "kinova",
    "kuka_iiwa_14": "kuka",
    "ufactory_xarm7": "ufactory",
    "ufactory_lite6": "ufactory",
}

_registry: Dict[str, RobotSpec] = {}

_ROBOT_ALIASES: Dict[str, str] = {
    "fr3": "franka_fr3",
    "franka": "franka_fr3",
    "panda": "franka_emika_panda",
    "franka_panda": "franka_emika_panda",
    "ur5e": "universal_robots_ur5e",
    "ur10e": "universal_robots_ur10e",
    "kinova": "kinova_gen3",
    "gen3": "kinova_gen3",
    "kuka": "kuka_iiwa_14",
    "iiwa": "kuka_iiwa_14",
    "iiwa14": "kuka_iiwa_14",
    "xarm": "ufactory_xarm7",
    "xarm7": "ufactory_xarm7",
    "lite6": "ufactory_lite6",
    "ufactory": "ufactory_lite6",
}


def _ensure_loaded() -> None:
    if _registry:
        return
    parser = RobotParser()
    for robot_id, rel_path in _ROBOT_PATHS.items():
        mjcf_path = _MENAGERIE_ROOT / rel_path
        if mjcf_path.exists():
            try:
                spec = parser.parse(robot_id, str(mjcf_path))
                if robot_id == "franka_fr3":
                    object.__setattr__(spec, "has_gripper", True)
                    object.__setattr__(spec, "gripper_joint_names", ("finger_joint1", "finger_joint2"))
                    object.__setattr__(spec, "gripper_actuator_names", ("actuator8",))
                    object.__setattr__(spec, "gripper_type", "parallel")
                    object.__setattr__(spec, "gripper_ctrl_open", 255.0)
                    object.__setattr__(spec, "gripper_ctrl_closed", 0.0)
                    object.__setattr__(spec, "end_effector_site", "pinch_site")
                    object.__setattr__(spec, "end_effector_body", "hand")
                    if spec.keyframe_qpos and len(spec.keyframe_qpos) == spec.dof:
                        object.__setattr__(spec, "keyframe_qpos", spec.keyframe_qpos + (0.0, 0.0))
                    if spec.keyframe_ctrl and len(spec.keyframe_ctrl) == spec.dof:
                        object.__setattr__(spec, "keyframe_ctrl", spec.keyframe_ctrl + (255.0,))
                # Enrich with metadata
                object.__setattr__(spec, "manufacturer", _MANUFACTURER.get(robot_id, ""))
                object.__setattr__(spec, "family", _FAMILY.get(robot_id, ""))
                _registry[robot_id] = spec
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to parse {robot_id} from {mjcf_path}: {e}")


def get_robot(robot_id: str) -> RobotSpec:
    """Get a RobotSpec by robot_id. Raises ValueError if not found."""
    _ensure_loaded()
    robot_id = _ROBOT_ALIASES.get(robot_id.strip().lower(), robot_id)
    if robot_id not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise ValueError(f"Unknown robot_id '{robot_id}'. Available: {available}")
    return _registry[robot_id]


def list_robots() -> List[RobotSpec]:
    """Return all registered robots."""
    _ensure_loaded()
    return list(_registry.values())


def list_robot_ids() -> List[str]:
    """Return all registered robot IDs."""
    _ensure_loaded()
    return sorted(_registry.keys())


def get_robot_spec(robot_id: str) -> RobotSpec:
    """Alias for get_robot() for compatibility with asset_library."""
    return get_robot(robot_id)


def reload_registry() -> None:
    """Force re-parsing of all robots (useful for testing)."""
    _registry.clear()
    _ensure_loaded()
