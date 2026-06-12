from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    name_zh: str
    name_en: str
    geometry: str
    size_m: List[float]
    mass_kg: float
    friction: List[float]
    tags: List[str]
    xml_kind: str = "primitive"
    source: str = "local"
    asset_path: str = ""


OBJECT_LIBRARY: List[ObjectSpec] = [
    ObjectSpec(
        object_id="cube_5cm",
        name_zh="5cm 立方体",
        name_en="5 cm cube",
        geometry="box",
        size_m=[0.025, 0.025, 0.025],
        mass_kg=0.12,
        friction=[0.55, 0.005, 0.0001],
        tags=["graspable", "pushable", "stackable", "box"],
    ),
    ObjectSpec(
        object_id="cube_7cm",
        name_zh="7cm 立方体",
        name_en="7 cm cube",
        geometry="box",
        size_m=[0.035, 0.035, 0.035],
        mass_kg=0.2,
        friction=[0.6, 0.005, 0.0001],
        tags=["graspable", "pushable", "box"],
    ),
    ObjectSpec(
        object_id="rect_block",
        name_zh="长方体积木",
        name_en="rectangular block",
        geometry="box",
        size_m=[0.05, 0.025, 0.02],
        mass_kg=0.16,
        friction=[0.55, 0.005, 0.0001],
        tags=["graspable", "pushable", "orientation_sensitive"],
    ),
    ObjectSpec(
        object_id="cylinder_can",
        name_zh="圆柱罐",
        name_en="cylindrical can",
        geometry="cylinder",
        size_m=[0.025, 0.06],
        mass_kg=0.11,
        friction=[0.5, 0.004, 0.0001],
        tags=["graspable", "pushable", "roll_risk"],
    ),
    ObjectSpec(
        object_id="small_sphere",
        name_zh="小球",
        name_en="small sphere",
        geometry="sphere",
        size_m=[0.03],
        mass_kg=0.08,
        friction=[0.35, 0.003, 0.0001],
        tags=["pushable", "roll_risk", "hard_to_grasp"],
    ),
    ObjectSpec(
        object_id="flat_puck",
        name_zh="扁圆盘",
        name_en="flat puck",
        geometry="cylinder",
        size_m=[0.035, 0.012],
        mass_kg=0.09,
        friction=[0.45, 0.004, 0.0001],
        tags=["pushable", "sliding", "low_profile"],
    ),
    ObjectSpec(
        object_id="screw_head",
        name_zh="简化螺丝头",
        name_en="simplified screw head",
        geometry="cylinder",
        size_m=[0.018, 0.006],
        mass_kg=0.18,
        friction=[1.25, 0.02, 0.0001],
        tags=["fastener", "screwdriving", "contact_target", "primitive"],
    ),
    ObjectSpec(
        object_id="insertion_socket",
        name_zh="简化插入座",
        name_en="simplified insertion socket",
        geometry="box",
        size_m=[0.035, 0.035, 0.012],
        mass_kg=0.45,
        friction=[1.4, 0.02, 0.0001],
        tags=["assembly", "fixture", "socket", "primitive"],
    ),
    ObjectSpec(
        object_id="button_target",
        name_zh="简化按钮",
        name_en="simplified button target",
        geometry="cylinder",
        size_m=[0.022, 0.008],
        mass_kg=0.10,
        friction=[0.9, 0.01, 0.0001],
        tags=["button", "press", "contact_target", "primitive"],
    ),
    ObjectSpec(
        object_id="scanned_screwdriver_phillips",
        name_zh="扫描版十字螺丝刀",
        name_en="scanned Phillips screwdriver",
        geometry="mesh",
        size_m=[],
        mass_kg=0.18,
        friction=[0.7, 0.005, 0.0001],
        tags=["tool", "screwdriving", "graspable", "mesh"],
        xml_kind="mjcf_mesh",
        source="mujoco_scanned_objects",
        asset_path="external/mujoco_scanned_objects/models/Craftsman_Grip_Screwdriver_Phillips_Cushion/model.xml",
    ),
    ObjectSpec(
        object_id="scanned_nuts_bolts",
        name_zh="扫描版螺母螺栓套件",
        name_en="scanned nuts and bolts set",
        geometry="mesh",
        size_m=[],
        mass_kg=0.22,
        friction=[0.8, 0.006, 0.0001],
        tags=["fastener", "screwdriving", "assembly", "mesh"],
        xml_kind="mjcf_mesh",
        source="mujoco_scanned_objects",
        asset_path="external/mujoco_scanned_objects/models/NUTS_BOLTS/model.xml",
    ),
    ObjectSpec(
        object_id="scanned_hammer_black",
        name_zh="扫描版锤子",
        name_en="scanned hammer",
        geometry="mesh",
        size_m=[],
        mass_kg=0.35,
        friction=[0.7, 0.005, 0.0001],
        tags=["tool", "hammer", "tool_use", "mesh"],
        xml_kind="mjcf_mesh",
        source="mujoco_scanned_objects",
        asset_path="external/mujoco_scanned_objects/models/Cole_Hardware_Hammer_Black/model.xml",
    ),
    ObjectSpec(
        object_id="scanned_cookie_spatula",
        name_zh="扫描版饼干铲",
        name_en="scanned cookie spatula",
        geometry="mesh",
        size_m=[],
        mass_kg=0.12,
        friction=[0.6, 0.005, 0.0001],
        tags=["tool", "spatula", "tool_use", "mesh"],
        xml_kind="mjcf_mesh",
        source="mujoco_scanned_objects",
        asset_path="external/mujoco_scanned_objects/models/OXO_Cookie_Spatula/model.xml",
    ),
    ObjectSpec(
        object_id="scanned_dish_towel_blue",
        name_zh="扫描版蓝色洗碗巾",
        name_en="scanned blue dish towel",
        geometry="mesh",
        size_m=[],
        mass_kg=0.06,
        friction=[0.8, 0.01, 0.0001],
        tags=["cloth_like", "towel", "folding", "mesh"],
        xml_kind="mjcf_mesh",
        source="mujoco_scanned_objects",
        asset_path="external/mujoco_scanned_objects/models/Cole_Hardware_Dishtowel_Blue/model.xml",
    ),
    ObjectSpec(
        object_id="scanned_kitchen_towel",
        name_zh="扫描版厨房毛巾",
        name_en="scanned kitchen towel",
        geometry="mesh",
        size_m=[],
        mass_kg=0.08,
        friction=[0.8, 0.01, 0.0001],
        tags=["cloth_like", "towel", "folding", "mesh"],
        xml_kind="mjcf_mesh",
        source="mujoco_scanned_objects",
        asset_path="external/mujoco_scanned_objects/models/Room_Essentials_Kitchen_Towels_16_x_26_2_count/model.xml",
    ),
    ObjectSpec(
        object_id="robosuite_round_nut",
        name_zh="robosuite 圆螺母",
        name_en="robosuite round nut",
        geometry="composite",
        size_m=[],
        mass_kg=0.1,
        friction=[0.95, 0.3, 0.1],
        tags=["assembly", "nut", "peg_in_hole", "mjcf"],
        xml_kind="mjcf",
        source="robosuite",
        asset_path="external/robosuite/robosuite/models/assets/objects/round-nut.xml",
    ),
    ObjectSpec(
        object_id="robosuite_square_nut",
        name_zh="robosuite 方螺母",
        name_en="robosuite square nut",
        geometry="composite",
        size_m=[],
        mass_kg=0.1,
        friction=[0.95, 0.3, 0.1],
        tags=["assembly", "nut", "peg_in_hole", "mjcf"],
        xml_kind="mjcf",
        source="robosuite",
        asset_path="external/robosuite/robosuite/models/assets/objects/square-nut.xml",
    ),
    ObjectSpec(
        object_id="robosuite_plate_with_hole",
        name_zh="robosuite 带孔板",
        name_en="robosuite plate with hole",
        geometry="composite",
        size_m=[],
        mass_kg=0.4,
        friction=[0.8, 0.01, 0.0001],
        tags=["assembly", "fixture", "hole", "mjcf"],
        xml_kind="mjcf",
        source="robosuite",
        asset_path="external/robosuite/robosuite/models/assets/objects/plate-with-hole.xml",
    ),
    ObjectSpec(
        object_id="robosuite_door_lock",
        name_zh="robosuite 门锁机构",
        name_en="robosuite door lock mechanism",
        geometry="articulated",
        size_m=[],
        mass_kg=1.0,
        friction=[1.0, 1.0, 1.0],
        tags=["tool_use", "articulated", "lock", "mjcf"],
        xml_kind="mjcf",
        source="robosuite",
        asset_path="external/robosuite/robosuite/models/assets/objects/door_lock.xml",
    ),
]


def list_objects() -> List[Dict[str, Any]]:
    return [asdict(obj) for obj in OBJECT_LIBRARY]


def get_object(object_id: str) -> ObjectSpec:
    for obj in OBJECT_LIBRARY:
        if obj.object_id == object_id:
            return obj
    supported = ", ".join(obj.object_id for obj in OBJECT_LIBRARY)
    raise ValueError(f"Unknown object_id '{object_id}'. Supported objects: {supported}")
