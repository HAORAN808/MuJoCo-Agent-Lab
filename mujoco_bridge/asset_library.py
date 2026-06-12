from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MENAGERIE_ROOT = PROJECT_ROOT / "external" / "mujoco_menagerie"


@dataclass(frozen=True)
class RobotAsset:
    asset_id: str
    name: str
    category: str
    path: str
    primary_xml: str
    status: str
    tags: List[str]


ROBOT_ASSETS: List[RobotAsset] = [
    RobotAsset("franka_fr3", "Franka FR3 + Franka Hand", "arm_gripper", "franka_fr3", "fr3.xml + franka_hand", "downloaded", ["arm", "7dof", "manipulation", "parallel_gripper"]),
    RobotAsset("franka_emika_panda", "Franka Emika Panda", "arm_gripper", "franka_emika_panda", "panda.xml", "downloaded", ["arm", "7dof", "parallel_gripper"]),
    RobotAsset("universal_robots_ur5e", "Universal Robots UR5e", "arm", "universal_robots_ur5e", "ur5e.xml", "downloaded", ["arm", "6dof", "industrial"]),
    RobotAsset("universal_robots_ur10e", "Universal Robots UR10e", "arm", "universal_robots_ur10e", "ur10e.xml", "downloaded", ["arm", "6dof", "industrial"]),
    RobotAsset("kinova_gen3", "Kinova Gen3", "arm", "kinova_gen3", "gen3.xml", "downloaded", ["arm", "7dof", "assistive"]),
    RobotAsset("kuka_iiwa_14", "KUKA iiwa 14", "arm", "kuka_iiwa_14", "iiwa14.xml", "downloaded", ["arm", "7dof", "industrial"]),
    RobotAsset("ufactory_xarm7", "UFactory xArm7", "arm_gripper", "ufactory_xarm7", "xarm7.xml", "downloaded", ["arm", "7dof", "parallel_gripper"]),
    RobotAsset("ufactory_lite6", "UFactory Lite6", "arm_gripper", "ufactory_lite6", "lite6.xml", "downloaded", ["arm", "6dof", "parallel_gripper"]),
    RobotAsset("robotiq_2f85", "Robotiq 2F-85", "gripper", "robotiq_2f85", "2f85.xml", "downloaded", ["parallel_gripper", "grasping"]),
    RobotAsset("robotiq_2f85_v4", "Robotiq 2F-85 v4", "gripper", "robotiq_2f85_v4", "2f85.xml", "downloaded", ["parallel_gripper", "grasping"]),
    RobotAsset("shadow_hand", "Shadow Dexterous Hand", "dexterous_hand", "shadow_hand", "right_hand.xml", "downloaded", ["dexterous", "in_hand", "tool_use"]),
    RobotAsset("leap_hand", "LEAP Hand", "dexterous_hand", "leap_hand", "right_hand.xml", "downloaded", ["dexterous", "in_hand", "low_cost"]),
    RobotAsset("wonik_allegro", "Wonik Allegro Hand", "dexterous_hand", "wonik_allegro", "right_hand.xml", "downloaded", ["dexterous", "in_hand"]),
    RobotAsset("aloha", "ALOHA dual-arm platform", "dual_arm", "aloha", "aloha.xml", "downloaded", ["dual_arm", "teleoperation", "bimanual"]),
    RobotAsset("realsense_d435i", "Intel RealSense D435i", "sensor", "realsense_d435i", "d435i.xml", "downloaded", ["camera", "rgbd", "perception"]),
]


TASK_BLUEPRINTS: List[Dict[str, Any]] = [
    {
        "blueprint_id": "pick_place_general",
        "name_zh": "通用抓取放置",
        "name_en": "general pick-and-place",
        "status": "implemented",
        "implemented_task_id": "fr3_pick_place",
        "required_assets": ["arm", "parallel_gripper", "graspable_object"],
    },
    {
        "blueprint_id": "tabletop_push_general",
        "name_zh": "桌面推动",
        "name_en": "tabletop pushing",
        "status": "implemented",
        "implemented_task_id": "tabletop_push",
        "required_assets": ["arm_or_pusher", "pushable_object", "target_region"],
    },
    {
        "blueprint_id": "screwdriving",
        "name_zh": "拧螺丝",
        "name_en": "screwdriving",
        "status": "implemented",
        "implemented_task_id": "screwdriving",
        "required_assets": ["arm", "wrist_tool", "scanned_screwdriver_phillips", "screw", "threaded_hole"],
    },
    {
        "blueprint_id": "cloth_folding",
        "name_zh": "叠衣服/布料折叠",
        "name_en": "cloth folding",
        "status": "implemented",
        "implemented_task_id": "cloth_folding",
        "required_assets": ["dual_arm_or_dexterous_gripper", "dynamic_cloth_folding_templates", "scanned_towel_assets"],
    },
    {
        "blueprint_id": "tool_use",
        "name_zh": "工具使用",
        "name_en": "tool use",
        "status": "implemented",
        "implemented_task_id": "tool_use",
        "required_assets": ["arm", "scanned_hammer_black", "scanned_cookie_spatula", "robosuite_door_lock"],
    },
    {
        "blueprint_id": "assembly_insertion",
        "name_zh": "装配/插入",
        "name_en": "assembly and insertion",
        "status": "implemented",
        "implemented_task_id": "assembly_insertion",
        "required_assets": ["arm", "robosuite_round_nut", "robosuite_square_nut", "robosuite_plate_with_hole"],
    },
]


def _exists(asset: RobotAsset) -> bool:
    return (MENAGERIE_ROOT / asset.path).exists()


def list_robot_assets() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for asset in ROBOT_ASSETS:
        row = asdict(asset)
        root = MENAGERIE_ROOT / asset.path
        row["available"] = _exists(asset)
        row["absolute_path"] = str(root)
        row["xml_path"] = str(root / asset.primary_xml)
        row["xml_exists"] = (root / asset.primary_xml).exists()
        rows.append(row)
    return rows


def list_task_blueprints() -> List[Dict[str, Any]]:
    return TASK_BLUEPRINTS


def list_asset_registry() -> Dict[str, Any]:
    return {
        "menagerie_root": str(MENAGERIE_ROOT),
        "robot_assets": list_robot_assets(),
        "task_blueprints": list_task_blueprints(),
    }
