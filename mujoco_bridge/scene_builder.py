"""Build a combined MuJoCo scene: FR3 arm + Franka Hand + workspace.

Assembles the scene by loading all mesh assets into memory as bytes and
passing them to ``MjModel.from_xml_string()`` via the *assets* dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from .object_library import get_object

_ASSETS = Path(__file__).resolve().parent / "assets"
_FR3_ROOT = Path(__file__).resolve().parent.parent / "external" / "mujoco_menagerie" / "franka_fr3"
_HAND_ROOT = _ASSETS / "franka_hand"


def _load_mesh_assets() -> Dict[str, bytes]:
    """Read all mesh files into a ``{filename: bytes}`` dict."""
    assets: Dict[str, bytes] = {}
    fr3_assets = _FR3_ROOT / "assets"
    hand_assets = _HAND_ROOT / "assets"
    for src_dir in (fr3_assets, hand_assets):
        for f in src_dir.iterdir():
            if f.is_file():
                assets[f.name] = f.read_bytes()
    return assets


def _object_geom_xml(object_id: str) -> tuple[str, float]:
    obj = get_object(object_id)
    if obj.xml_kind != "primitive":
        obj = get_object("cube_5cm")
    geom_type = obj.geometry if obj.geometry in {"box", "sphere", "cylinder"} else "box"
    size = " ".join(str(v) for v in obj.size_m)
    if geom_type == "box":
        half_z = obj.size_m[2]
    elif geom_type == "sphere":
        half_z = obj.size_m[0]
    elif geom_type == "cylinder":
        half_z = obj.size_m[1]
    else:
        half_z = 0.025
    friction = " ".join(str(v) for v in obj.friction)
    geom_xml = (
        f'<geom name="cube" type="{geom_type}" size="{size}" material="cube_mat" '
        f'mass="{obj.mass_kg}" condim="6" contype="7" conaffinity="7" friction="{friction}"/>'
    )
    return geom_xml, 0.37 + half_z


def _held_tool_xml(tool_id: str | None) -> str:
    if not tool_id:
        return ""
    if tool_id == "spatula":
        return """
                        <body name="held_tool" pos="0 0 0.112">
                          <geom name="held_tool_handle" type="capsule" fromto="-0.055 0 0 0.065 0 0" size="0.008" mass="0.045"
                                rgba="0.10 0.11 0.12 1" contype="8" conaffinity="7" friction="1.1 0.01 0.0001"/>
                          <geom name="held_tool_blade" type="box" pos="0.092 0 -0.002" size="0.032 0.020 0.004" mass="0.025"
                                rgba="0.82 0.84 0.86 1" contype="8" conaffinity="7" friction="1.1 0.01 0.0001"/>
                        </body>"""
    if tool_id == "screwdriver":
        return """
                        <body name="held_tool" pos="0 0 0.112">
                          <geom name="held_tool_handle" type="capsule" fromto="-0.055 0 0 0.030 0 0" size="0.014" mass="0.060"
                                rgba="0.08 0.10 0.12 1" contype="8" conaffinity="7" friction="1.2 0.01 0.0001"/>
                          <geom name="held_tool_tip" type="capsule" fromto="0.030 0 0 0.105 0 0" size="0.0055" mass="0.020"
                                rgba="0.86 0.80 0.18 1" contype="8" conaffinity="7" friction="1.2 0.01 0.0001"/>
                        </body>"""
    if tool_id == "peg":
        return """
                        <body name="held_tool" pos="0 0 0.112">
                          <geom name="held_tool_shank" type="capsule" fromto="-0.040 0 0 0.105 0 0" size="0.010" mass="0.090"
                                rgba="0.70 0.72 0.74 1" contype="8" conaffinity="7" friction="1.4 0.02 0.0001"/>
                          <geom name="held_tool_tip" type="sphere" pos="0.112 0 0" size="0.011" mass="0.020"
                                rgba="0.90 0.84 0.20 1" contype="8" conaffinity="7" friction="1.4 0.02 0.0001"/>
                        </body>"""
    return """
                        <body name="held_tool" pos="0 0 0.112">
                          <geom name="held_tool_handle" type="capsule" fromto="-0.060 0 0 0.040 0 0" size="0.012" mass="0.070"
                                rgba="0.12 0.14 0.16 1" contype="8" conaffinity="7" friction="1.0 0.01 0.0001"/>
                          <geom name="held_tool_head" type="box" pos="0.072 0 0" size="0.024 0.018 0.018" mass="0.090"
                                rgba="0.72 0.74 0.76 1" contype="8" conaffinity="7" friction="1.0 0.01 0.0001"/>
                        </body>"""


def _scene_xml(object_id: str = "cube_5cm", held_tool: str | None = None) -> str:
    """Return the combined scene XML string (meshdir placeholder)."""
    object_geom, object_z = _object_geom_xml(object_id)
    held_tool_xml = _held_tool_xml(held_tool)
    return rf"""<mujoco model="fr3_pick_place">
  <compiler angle="radian" meshdir="." autolimits="true"/>
  <option integrator="implicitfast" timestep="0.002"/>

  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
    <rgba haze="0.78 0.84 0.90 1"/>
    <global azimuth="135" elevation="-22"/>
  </visual>

  <default>
    <default class="fr3">
      <joint damping="0.21" armature="0.195" frictionloss="1.137"/>
      <position inheritrange="1"/>
      <default class="visual">
        <geom type="mesh" group="2" contype="0" conaffinity="0"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" mass="0" density="0"/>
      </default>
      <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
    </default>
    <default class="panda">
      <material specular="0.5" shininess="0.25"/>
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
      <default class="panda_finger">
        <joint axis="0 1 0" type="slide" range="0 0.04"/>
      </default>
      <default class="panda_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="panda_collision">
        <geom type="mesh" group="3"/>
        <default class="panda_fingertip_1">
          <geom type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445" contype="2" conaffinity="2" condim="6" friction="2.0 0.005 0.0001"/>
        </default>
        <default class="panda_fingertip_2">
          <geom type="box" size="0.003 0.002 0.003" pos="0.0055 0.002 0.05" contype="2" conaffinity="2" condim="6" friction="2.0 0.005 0.0001"/>
        </default>
        <default class="panda_fingertip_3">
          <geom type="box" size="0.003 0.002 0.003" pos="-0.0055 0.002 0.05" contype="2" conaffinity="2" condim="6" friction="2.0 0.005 0.0001"/>
        </default>
        <default class="panda_fingertip_4">
          <geom type="box" size="0.003 0.002 0.0035" pos="0.0055 0.002 0.0395" contype="2" conaffinity="2" condim="6" friction="2.0 0.005 0.0001"/>
        </default>
        <default class="panda_fingertip_5">
          <geom type="box" size="0.003 0.002 0.0035" pos="-0.0055 0.002 0.0395" contype="2" conaffinity="2" condim="6" friction="2.0 0.005 0.0001"/>
        </default>
      </default>
    </default>
    <default class="workspace">
      <geom rgba="0.6 0.6 0.5 1" contype="1" conaffinity="1"/>
    </default>
  </default>

  <asset>
    <!-- FR3 materials -->
    <material name="black" rgba=".2 .2 .2 1"/>
    <material name="white" rgba="1 1 1 1"/>
    <material name="red" rgba="1 0.072272 0.039546 1"/>
    <material name="gray" rgba="0.863156 0.863156 0.863157 1"/>
    <material name="button_green" rgba="0.102241 0.571125 0.102242 1"/>
    <material name="button_red" rgba="0.520996 0.008023 0.013702 1"/>
    <material name="button_blue" rgba="0.024157 0.445201 0.737911 1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.80 0.88 0.96" rgb2="0.98 0.99 1.00" width="512" height="3072"/>
    <texture type="2d" name="floor_tex" builtin="checker" mark="edge" rgb1="0.82 0.86 0.90" rgb2="0.70 0.76 0.82" markrgb="0.95 0.95 0.95" width="300" height="300"/>
    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="8 8" reflectance="0.08"/>

    <!-- Hand materials -->
    <material class="panda" name="off_white" rgba="0.901961 0.921569 0.929412 1"/>

    <!-- FR3 collision meshes -->
    <mesh name="link0_coll" file="link0.stl"/>
    <mesh name="link1_coll" file="link1.stl"/>
    <mesh name="link2_coll" file="link2.stl"/>
    <mesh name="link3_coll" file="link3.stl"/>
    <mesh name="link4_coll" file="link4.stl"/>
    <mesh name="link5_coll" file="link5.stl"/>
    <mesh name="link6_coll" file="link6.stl"/>
    <mesh name="link7_coll" file="link7.stl"/>

    <!-- FR3 visual meshes -->
    <mesh file="link0_0.obj"/>
    <mesh file="link0_1.obj"/>
    <mesh file="link0_2.obj"/>
    <mesh file="link0_3.obj"/>
    <mesh file="link0_4.obj"/>
    <mesh file="link0_5.obj"/>
    <mesh file="link0_6.obj"/>
    <mesh file="link1.obj"/>
    <mesh file="link2.obj"/>
    <mesh file="link3_0.obj"/>
    <mesh file="link3_1.obj"/>
    <mesh file="link4_0.obj"/>
    <mesh file="link4_1.obj"/>
    <mesh file="link5_0.obj"/>
    <mesh file="link5_1.obj"/>
    <mesh file="link5_2.obj"/>
    <mesh file="link6_0.obj"/>
    <mesh file="link6_1.obj"/>
    <mesh file="link6_2.obj"/>
    <mesh file="link6_3.obj"/>
    <mesh file="link6_4.obj"/>
    <mesh file="link6_5.obj"/>
    <mesh file="link6_6.obj"/>
    <mesh file="link6_7.obj"/>
    <mesh file="link7_0.obj"/>
    <mesh file="link7_1.obj"/>
    <mesh file="link7_2.obj"/>
    <mesh file="link7_3.obj"/>

    <!-- Hand collision mesh -->
    <mesh name="hand_c" file="hand.stl"/>

    <!-- Hand visual meshes -->
    <mesh file="hand_0.obj"/>
    <mesh file="hand_1.obj"/>
    <mesh file="hand_2.obj"/>
    <mesh file="hand_3.obj"/>
    <mesh file="hand_4.obj"/>
    <mesh file="finger_0.obj"/>
    <mesh file="finger_1.obj"/>

    <!-- Cube texture -->
    <texture name="cube_tex" type="cube" builtin="gradient" rgb1="0.8 0.2 0.1" rgb2="0.6 0.1 0.05" width="64" height="64"/>
    <material name="cube_mat" texture="cube_tex" rgba="0.8 0.2 0.1 1" specular="0.3" shininess="0.1"/>
  </asset>

  <worldbody>
    <light name="replay_key_light" pos="0 -1.2 1.8" dir="0 0 -1" directional="true"/>
    <light name="replay_fill_light" pos="-0.8 0.8 1.3" dir="0.4 -0.4 -1" directional="true"/>
    <camera name="replay_camera" pos="0.88 -1.08 0.76" xyaxes="0.78 0.63 0 -0.25 0.31 0.92"/>

    <!-- Ground plane -->
    <geom name="floor" type="plane" size="1 1 0.01" material="floor_mat" contype="1" conaffinity="1"/>

    <!-- Table -->
    <body name="table" pos="0.5 0 0.18">
      <geom name="table_top" type="box" size="0.4 0.3 0.01" pos="0 0 0.18" class="workspace" mass="20"/>
      <geom name="table_leg1" type="cylinder" size="0.025 0.09" pos="0.35 0.25 0.09" class="workspace" mass="1"/>
      <geom name="table_leg2" type="cylinder" size="0.025 0.09" pos="-0.35 0.25 0.09" class="workspace" mass="1"/>
      <geom name="table_leg3" type="cylinder" size="0.025 0.09" pos="0.35 -0.25 0.09" class="workspace" mass="1"/>
      <geom name="table_leg4" type="cylinder" size="0.025 0.09" pos="-0.35 -0.25 0.09" class="workspace" mass="1"/>
    </body>

    <!-- FR3 Arm mounted above table -->
    <body name="base" childclass="fr3" pos="-0.05 0 0.63">
      <body name="fr3_link0">
        <geom mesh="link0_0" material="black" class="visual"/>
        <geom mesh="link0_1" material="white" class="visual"/>
        <geom mesh="link0_2" material="white" class="visual"/>
        <geom mesh="link0_3" material="white" class="visual"/>
        <geom mesh="link0_4" material="white" class="visual"/>
        <geom mesh="link0_5" material="red" class="visual"/>
        <geom mesh="link0_6" material="black" class="visual"/>
        <geom name="fr3_link0_collision" mesh="link0_coll" class="collision"/>
        <body name="fr3_link1" pos="0 0 0.333">
          <inertial pos="4.128e-07 -0.0181251 -0.0386036" quat="0.998098 -0.0605364 0.00380499 0.0110109" mass="2.92747"
            diaginertia="0.0239286 0.0227246 0.00610634"/>
          <joint name="fr3_joint1" axis="0 0 1" range="-2.7437 2.7437" actuatorfrcrange="-87 87"/>
          <geom name="fr3_link1_collision" class="collision" mesh="link1_coll"/>
          <geom material="white" mesh="link1" class="visual"/>
          <body name="fr3_link2" quat="1 -1 0 0">
            <inertial pos="0.00318289 -0.0743222 0.00881461" quat="0.502599 0.584437 -0.465998 0.434366" mass="2.93554"
              diaginertia="0.0629567 0.0411924 0.0246371"/>
            <joint name="fr3_joint2" axis="0 0 1" range="-1.7837 1.7837" actuatorfrcrange="-87 87"/>
            <geom material="white" mesh="link2" class="visual"/>
            <geom name="fr3_link2_collision" class="collision" mesh="link2_coll"/>
            <body name="fr3_link3" pos="0 -0.316 0" quat="1 1 0 0">
              <inertial pos="0.0407016 -0.00482006 -0.0289731" quat="0.921025 -0.244161 0.155272 0.260745" mass="2.2449"
                diaginertia="0.0267409 0.0189869 0.0171587"/>
              <joint name="fr3_joint3" axis="0 0 1" range="-2.9007 2.9007" actuatorfrcrange="-87 87"/>
              <geom mesh="link3_0" material="white" class="visual"/>
              <geom mesh="link3_1" material="black" class="visual"/>
              <geom name="fr3_link3_collision" class="collision" mesh="link3_coll"/>
              <body name="fr3_link4" pos="0.0825 0 0" quat="1 1 0 0">
                <inertial pos="-0.0459101 0.0630493 -0.00851879" quat="0.438018 0.803311 0.00937812 0.403414"
                  mass="2.6156" diaginertia="0.05139 0.0372717 0.0160047"/>
                <joint name="fr3_joint4" axis="0 0 1" range="-3.0421 -0.1518" actuatorfrcrange="-87 87"/>
                <geom mesh="link4_0" material="white" class="visual"/>
                <geom mesh="link4_1" material="black" class="visual"/>
                <geom name="fr3_link4_collision" class="collision" mesh="link4_coll"/>
                <body name="fr3_link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                  <inertial pos="-0.00160396 0.0292536 -0.0972966" quat="0.919031 0.125604 0.0751531 -0.366003"
                    mass="2.32712" diaginertia="0.0579335 0.0449144 0.0130634"/>
                  <joint name="fr3_joint5" axis="0 0 1" range="-2.8065 2.8065" actuatorfrcrange="-12 12"
                    armature="0.074" frictionloss="0.763"/>
                  <geom mesh="link5_0" material="white" class="visual"/>
                  <geom mesh="link5_1" material="white" class="visual"/>
                  <geom mesh="link5_2" material="black" class="visual"/>
                  <geom name="fr3_link5_collision" class="collision" mesh="link5_coll"/>
                  <body name="fr3_link6" quat="1 1 0 0">
                    <inertial pos="0.0597131 -0.0410295 -0.0101693" quat="0.621301 0.552665 0.510011 0.220081"
                      mass="1.81704" diaginertia="0.0175039 0.0161123 0.00193529"/>
                    <joint name="fr3_joint6" axis="0 0 1" range="0.5445 4.5169" actuatorfrcrange="-12 12"
                      armature="0.074" frictionloss="0.44"/>
                    <geom mesh="link6_0" material="button_green" class="visual"/>
                    <geom mesh="link6_1" material="white" class="visual"/>
                    <geom mesh="link6_2" material="white" class="visual"/>
                    <geom mesh="link6_3" material="gray" class="visual"/>
                    <geom mesh="link6_4" material="button_red" class="visual"/>
                    <geom mesh="link6_5" material="white" class="visual"/>
                    <geom mesh="link6_6" material="black" class="visual"/>
                    <geom mesh="link6_7" material="button_blue" class="visual"/>
                    <geom name="fr3_link6_collision" class="collision" mesh="link6_coll"/>
                    <body name="fr3_link7" pos="0.088 0 0" quat="1 1 0 0">
                      <inertial pos="0.00452258 0.00862619 -0.0161633" quat="0.727579 0.0978688 -0.24906 0.63168"
                        mass="0.627143" diaginertia="0.000223836 0.000223642 5.64132e-07"/>
                      <joint name="fr3_joint7" axis="0 0 1" range="-3.0159 3.0159" actuatorfrcrange="-12 12"
                        armature="0.074" frictionloss="0.248"/>
                      <geom mesh="link7_0" material="black" class="visual"/>
                      <geom mesh="link7_1" material="white" class="visual"/>
                      <geom mesh="link7_2" material="white" class="visual"/>
                      <geom mesh="link7_3" material="black" class="visual"/>
                      <geom name="fr3_link7_collision" class="collision" mesh="link7_coll"/>
                      <site name="attachment_site" pos="0 0 0.107"/>

                      <!-- Franka Hand at attachment_site -->
                      <body name="hand" childclass="panda" pos="0 0 0.107" quat="0.923880 0 0 0.382683">
                        <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                        <geom mesh="hand_0" material="off_white" class="panda_visual"/>
                        <geom mesh="hand_1" material="black" class="panda_visual"/>
                        <geom mesh="hand_2" material="black" class="panda_visual"/>
                        <geom mesh="hand_3" material="white" class="panda_visual"/>
                        <geom mesh="hand_4" material="off_white" class="panda_visual"/>
                        <geom mesh="hand_c" class="panda_collision"/>
                        {held_tool_xml}
                        <body name="left_finger" pos="0 0 0.0584">
                          <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                          <joint name="finger_joint1" class="panda_finger"/>
                          <geom mesh="finger_0" material="off_white" class="panda_visual"/>
                          <geom mesh="finger_1" material="black" class="panda_visual"/>
                          <geom mesh="finger_0" class="panda_collision" contype="4" conaffinity="4"/>
                          <geom class="panda_fingertip_1"/>
                          <geom class="panda_fingertip_2"/>
                          <geom class="panda_fingertip_3"/>
                          <geom class="panda_fingertip_4"/>
                          <geom class="panda_fingertip_5"/>
                          <site name="left_finger_touch" pos="0 0.0055 0.0445" size="0.012 0.005 0.012"/>
                        </body>
                        <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                          <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                          <joint name="finger_joint2" class="panda_finger"/>
                          <geom mesh="finger_0" material="off_white" class="panda_visual"/>
                          <geom mesh="finger_1" material="black" class="panda_visual"/>
                          <geom mesh="finger_0" class="panda_collision" contype="4" conaffinity="4"/>
                          <geom class="panda_fingertip_1"/>
                          <geom class="panda_fingertip_2"/>
                          <geom class="panda_fingertip_3"/>
                          <geom class="panda_fingertip_4"/>
                          <geom class="panda_fingertip_5"/>
                          <site name="right_finger_touch" pos="0 0.0055 0.0445" size="0.012 0.005 0.012"/>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Grasp target object. Kept as cube_body/cube for compatibility with existing runners. -->
    <body name="cube_body" pos="0.5 0 {object_z:.5f}">
      <joint name="cube_free" type="free" damping="0.001"/>
      {object_geom}
      <site name="cube_center" pos="0 0 0"/>
    </body>

    <!-- Target indicator (visual only) -->
    <body name="target_indicator" pos="0.60 0 {object_z:.5f}">
      <geom name="target_marker" type="cylinder" size="0.025 0.001" rgba="0.1 0.8 0.1 0.3"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <contact>
    <exclude body1="fr3_link0" body2="fr3_link1"/>
    <exclude body1="hand" body2="left_finger"/>
    <exclude body1="hand" body2="right_finger"/>
  </contact>

  <tendon>
    <fixed name="split">
      <joint joint="finger_joint1" coef="0.5"/>
      <joint joint="finger_joint2" coef="0.5"/>
    </fixed>
  </tendon>

  <equality>
    <joint joint1="finger_joint1" joint2="finger_joint2" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>

  <sensor>
    <touch name="left_touch" site="left_finger_touch"/>
    <touch name="right_touch" site="right_finger_touch"/>
    <jointpos name="finger1_pos" joint="finger_joint1"/>
    <jointpos name="finger2_pos" joint="finger_joint2"/>
    <jointvel name="finger1_vel" joint="finger_joint1"/>
    <jointvel name="finger2_vel" joint="finger_joint2"/>
    <framepos name="cube_pos" objtype="site" objname="cube_center"/>
  </sensor>

  <actuator>
    <!-- FR3 arm position actuators (very high gains for gravity compensation) -->
    <position class="fr3" name="fr3_joint1" joint="fr3_joint1" kp="20000" kv="2000"/>
    <position class="fr3" name="fr3_joint2" joint="fr3_joint2" kp="20000" kv="2000"/>
    <position class="fr3" name="fr3_joint3" joint="fr3_joint3" kp="15000" kv="1500"/>
    <position class="fr3" name="fr3_joint4" joint="fr3_joint4" kp="15000" kv="1500"/>
    <position class="fr3" name="fr3_joint5" joint="fr3_joint5" kp="10000" kv="1000"/>
    <position class="fr3" name="fr3_joint6" joint="fr3_joint6" kp="10000" kv="1000"/>
    <position class="fr3" name="fr3_joint7" joint="fr3_joint7" kp="10000" kv="1000"/>
    <!-- Hand gripper actuator (tendon-driven, ctrlrange 0-255) -->
    <general class="panda" name="actuator8" tendon="split" forcerange="-100 100" ctrlrange="0 255"
      gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>
  </actuator>

  <keyframe>
    <key name="home"
      qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0 0 0.5 0 {object_z:.5f} 1 0 0 0"
      ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255"/>
  </keyframe>
</mujoco>
"""


def build_scene(mujoco_module, object_id: str = "cube_5cm", held_tool: str | None = None) -> Tuple:
    """Build and return (model, data) for the FR3+Hand pick-place scene.

    Loads all mesh assets into memory and passes them to
    ``MjModel.from_xml_string()`` via the *assets* dict.

    Parameters
    ----------
    mujoco_module : module
        The ``mujoco`` Python module (imported externally to give a clear
        error if missing).

    Returns
    -------
    (model, data) : tuple
        MuJoCo model and data objects for the scene.
    """
    assets = _load_mesh_assets()
    xml = _scene_xml(object_id, held_tool=held_tool)
    model = mujoco_module.MjModel.from_xml_string(xml, assets)
    data = mujoco_module.MjData(model)
    return model, data


def get_cube_body_id(mujoco_module, model) -> int:
    """Return the body ID of ``cube_body``."""
    return mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, "cube_body")


def get_touch_sensor_ids(mujoco_module, model) -> Tuple[int, int]:
    """Return sensor IDs for left and right fingertip touch sensors."""
    left = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SENSOR, "left_touch")
    right = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SENSOR, "right_touch")
    return left, right


def get_cube_pos_sensor_id(mujoco_module, model) -> int:
    """Return sensor ID for the cube position sensor."""
    return mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SENSOR, "cube_pos")
