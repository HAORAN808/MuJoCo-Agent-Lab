from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from ..protocols import SkillStep
from .base import SkillDefinition, SkillFailure


def _failure(failure_type: str, description: str, retry_hint: str) -> SkillFailure:
    return SkillFailure(failure_type=failure_type, description=description, retry_hint=retry_hint)


SKILL_LIBRARY: Dict[str, SkillDefinition] = {
    "reach": SkillDefinition(
        name="reach",
        description="Move the end effector to a target pose using task-space IK.",
        input_params=["target_pos", "target_quat", "steps"],
        preconditions=["Target pose is inside the robot workspace."],
        expected_observations=["End-effector distance to target decreases."],
        success_criteria=["Final end-effector error is below tolerance."],
        failure_modes=[
            _failure("ik_unreachable", "IK cannot reach the requested pose.", "Choose a nearer target or different robot."),
            _failure("trajectory_error", "Trajectory error remains above tolerance.", "Increase steps or relax pose constraint."),
        ],
        tunable_params={"steps": [120, 240, 360], "tolerance": [0.01, 0.02, 0.05]},
        implemented_by="mujoco_bridge.motion_primitives.MotionPrimitive.reach",
    ),
    "grasp": SkillDefinition(
        name="grasp",
        description="Close the gripper around a graspable object.",
        input_params=["force_scale", "object_id"],
        preconditions=["Robot has a gripper.", "Object has a graspable tag or grasp point."],
        expected_observations=["Gripper closes and object remains near end effector."],
        success_criteria=["Object is retained during lift or transport."],
        failure_modes=[
            _failure("grasp_slip", "Object slips after gripper closure.", "Increase force or adjust grasp pose."),
            _failure("no_gripper", "Selected robot has no compatible gripper.", "Select a robot with a gripper."),
        ],
        tunable_params={"force_scale": [0.5, 0.75, 1.0]},
        implemented_by="mujoco_bridge.motion_primitives.MotionPrimitive.grasp",
    ),
    "lift": SkillDefinition(
        name="lift",
        description="Raise the end effector or grasped object by a target height.",
        input_params=["height", "steps"],
        preconditions=["Object is grasped or end effector is clear of obstacles."],
        expected_observations=["Object z position or end-effector z position increases."],
        success_criteria=["Lifted height exceeds the requested threshold."],
        failure_modes=[
            _failure("lift_failed", "Object height did not increase enough.", "Check grasp quality or reduce payload."),
            _failure("collision", "Lift path collided with scene geometry.", "Add a clearance waypoint."),
        ],
        tunable_params={"height": [0.04, 0.08, 0.12], "steps": [120, 240, 360]},
        implemented_by="mujoco_bridge.motion_primitives.MotionPrimitive.lift",
    ),
    "place": SkillDefinition(
        name="place",
        description="Move a grasped object to a target placement pose.",
        input_params=["target_pos", "steps"],
        preconditions=["Object is grasped.", "Target pose is reachable."],
        expected_observations=["Object approaches placement region."],
        success_criteria=["Object final distance to target is below tolerance."],
        failure_modes=[
            _failure("placement_error", "Object final pose is outside tolerance.", "Reduce speed or add alignment waypoint."),
            _failure("object_dropped", "Object was released too early.", "Adjust gripper timing."),
        ],
        tunable_params={"steps": [180, 300, 420], "tolerance": [0.02, 0.04, 0.06]},
        implemented_by="mujoco_bridge.motion_primitives.MotionPrimitive.place",
    ),
    "push": SkillDefinition(
        name="push",
        description="Apply lateral contact motion to displace an object.",
        input_params=["direction", "distance", "steps"],
        preconditions=["Object is pushable.", "Contact point is reachable."],
        expected_observations=["Object displacement increases along push direction."],
        success_criteria=["Object displacement crosses target threshold."],
        failure_modes=[
            _failure("missed_contact", "End effector did not establish contact.", "Lower approach height or adjust lateral offset."),
            _failure("insufficient_displacement", "Object moved less than expected.", "Increase push distance or contact force."),
        ],
        tunable_params={"distance": [0.04, 0.08, 0.12], "steps": [120, 240, 360]},
        implemented_by="mujoco_bridge.motion_primitives.MotionPrimitive.push",
    ),
    "pull": SkillDefinition(
        name="pull",
        description="Apply a controlled contact or grasp motion that moves an object toward the robot.",
        input_params=["direction", "distance", "grasp_required"],
        preconditions=["Object has a pullable feature or is grasped."],
        expected_observations=["Object displacement is opposite the approach direction."],
        success_criteria=["Object moves toward the desired region."],
        failure_modes=[
            _failure("feature_not_found", "No pullable feature is available.", "Switch to grasp or push strategy."),
            _failure("slip", "Contact slips during pull.", "Increase normal force or use gripper."),
        ],
        tunable_params={"distance": [0.03, 0.06, 0.1]},
        implemented_by="planned_skill_library",
    ),
    "press": SkillDefinition(
        name="press",
        description="Apply downward contact force on a button or contact target.",
        input_params=["target_pos", "force", "duration_steps"],
        preconditions=["Target is a pressable contact object."],
        expected_observations=["Contact steps and normal force exceed threshold."],
        success_criteria=["Contact duration and force satisfy button criterion."],
        failure_modes=[
            _failure("no_contact", "Press motion did not touch the target.", "Reduce approach height or improve alignment."),
            _failure("insufficient_force", "Contact force was too low.", "Increase downforce or dwell time."),
        ],
        tunable_params={"force": ["light", "nominal", "heavy"], "duration_steps": [40, 80, 120]},
        implemented_by="mujoco_bridge.arm_runner.FR3ArmSkillRunner.button_press",
    ),
    "insert": SkillDefinition(
        name="insert",
        description="Move a peg, tool, or object along an insertion axis into a fixture.",
        input_params=["target_pos", "axis", "depth_m", "compliance"],
        preconditions=["Fixture and inserted part are aligned within tolerance."],
        expected_observations=["Sustained contact without large lateral displacement."],
        success_criteria=["Insertion depth or contact pattern satisfies criterion."],
        failure_modes=[
            _failure("alignment_error", "Part is laterally or angularly misaligned.", "Retry with smaller lateral offset and alignment step."),
            _failure("jammed", "Insertion contact stalls before depth target.", "Reduce speed or add compliance."),
        ],
        tunable_params={"compliance": ["stiff", "nominal", "soft"], "depth_m": [0.015, 0.03, 0.045]},
        implemented_by="mujoco_bridge.arm_runner.FR3ArmSkillRunner.peg_insert",
    ),
    "rotate": SkillDefinition(
        name="rotate",
        description="Rotate a tool or object around a target axis.",
        input_params=["axis", "angle_deg", "torque_hint"],
        preconditions=["Tool or object can be rotated without violating constraints."],
        expected_observations=["Target orientation changes along requested axis."],
        success_criteria=["Orientation error is below tolerance."],
        failure_modes=[
            _failure("slip", "Tool slips instead of transmitting rotation.", "Improve tool alignment or normal force."),
            _failure("torque_limit", "Required torque exceeds safe limit.", "Lower angle step or reject real execution."),
        ],
        tunable_params={"angle_deg": [15, 30, 60], "torque_hint": ["low", "nominal", "high"]},
        implemented_by="planned_skill_library",
    ),
    "screw": SkillDefinition(
        name="screw",
        description="Use a screwdriver-like tool to contact and rotate a fastener.",
        input_params=["tool_id", "downforce", "approach_angle", "rotation"],
        preconditions=["A screwdriver tool is attached.", "Target is a screw-like object."],
        expected_observations=["Tool contact is sustained at the fastener head."],
        success_criteria=["Contact steps and alignment metrics satisfy screwdriving criterion."],
        failure_modes=[
            _failure("driver_slip", "Tool loses contact with screw head.", "Increase downforce or improve alignment."),
            _failure("tilt_error", "Approach angle is too far from the screw axis.", "Retry with vertical approach."),
        ],
        tunable_params={"downforce": ["light", "nominal", "heavy"], "approach_angle": ["vertical", "tilted_5deg"]},
        implemented_by="mujoco_bridge.arm_runner.FR3ArmSkillRunner.tool_contact_sweep",
    ),
    "tool_contact": SkillDefinition(
        name="tool_contact",
        description="Use an attached tool to establish controlled contact with a target object.",
        input_params=["tool_id", "object_id", "contact_path"],
        preconditions=["Tool is attached and target is reachable."],
        expected_observations=["Tool-target contact steps are recorded."],
        success_criteria=["Tool contact duration and displacement satisfy task criterion."],
        failure_modes=[
            _failure("tool_missed_target", "Tool path did not contact the target.", "Adjust object pose or contact path."),
            _failure("excessive_force", "Contact force exceeds the allowed range.", "Reduce speed or downforce."),
        ],
        tunable_params={"contact_path": ["short", "nominal", "long"]},
        implemented_by="mujoco_bridge.arm_runner.FR3ArmSkillRunner.tool_contact_sweep",
    ),
    "sweep": SkillDefinition(
        name="sweep",
        description="Sweep the end effector or tool across a target surface or object.",
        input_params=["direction", "distance", "tool_id"],
        preconditions=["Sweep path is collision-free except intended contact."],
        expected_observations=["Contact steps occur over a path interval."],
        success_criteria=["Target object moves or surface contact metric is satisfied."],
        failure_modes=[
            _failure("path_blocked", "Sweep path collides with unexpected geometry.", "Use a shorter path or higher clearance."),
            _failure("weak_contact", "Sweep contact is too brief.", "Lower path height or increase distance."),
        ],
        tunable_params={"distance": [0.04, 0.08, 0.12]},
        implemented_by="mujoco_bridge.arm_runner.FR3ArmSkillRunner.contact_sweep",
    ),
    "align": SkillDefinition(
        name="align",
        description="Add a pose adjustment step before contact-rich manipulation.",
        input_params=["axis", "target_pos", "tolerance"],
        preconditions=["Target reference pose is known."],
        expected_observations=["Pose error decreases before contact action."],
        success_criteria=["Alignment error is inside tolerance."],
        failure_modes=[
            _failure("alignment_error", "Pose remains outside tolerance.", "Add visual estimate or reduce task difficulty."),
            _failure("reference_missing", "Target reference is missing.", "Provide fixture or object pose."),
        ],
        tunable_params={"tolerance": [0.005, 0.01, 0.02]},
        implemented_by="planned_skill_library",
    ),
    "release": SkillDefinition(
        name="release",
        description="Open the gripper or detach the held object/tool at the desired moment.",
        input_params=["steps"],
        preconditions=["Robot has a gripper or detachable tool state."],
        expected_observations=["Gripper opens and object is no longer constrained."],
        success_criteria=["Object remains stable after release."],
        failure_modes=[
            _failure("release_instability", "Object moves unexpectedly after release.", "Lower release height or add settling time."),
        ],
        tunable_params={"steps": [40, 80, 120]},
        implemented_by="mujoco_bridge.motion_primitives.MotionPrimitive.release",
    ),
}


ACTION_TO_SKILL = {
    "reach": "reach",
    "grasp": "grasp",
    "lift": "lift",
    "place": "place",
    "push": "push",
    "insert": "insert",
    "release": "release",
    "wait": "release",
    "press": "press",
    "rotate": "rotate",
    "sweep": "sweep",
    "align": "align",
}


def list_skill_names() -> List[str]:
    return sorted(SKILL_LIBRARY)


def list_skill_definitions() -> List[Dict[str, Any]]:
    return [SKILL_LIBRARY[name].to_dict() for name in list_skill_names()]


def get_skill(name: str) -> SkillDefinition:
    try:
        return SKILL_LIBRARY[name]
    except KeyError as exc:
        supported = ", ".join(list_skill_names())
        raise ValueError(f"Unknown skill '{name}'. Supported skills: {supported}") from exc


def task_actions_to_skill_steps(actions: Iterable[Mapping[str, Any]]) -> List[SkillStep]:
    steps: List[SkillStep] = []
    for action in actions:
        action_type = str(action.get("type") or "wait")
        skill_name = ACTION_TO_SKILL.get(action_type, action_type)
        definition = get_skill(skill_name)
        params = {key: value for key, value in action.items() if key != "type"}
        steps.append(
            SkillStep(
                skill_name=skill_name,
                params=params,
                preconditions=definition.preconditions,
                expected_observations=definition.expected_observations,
            )
        )
    return steps
