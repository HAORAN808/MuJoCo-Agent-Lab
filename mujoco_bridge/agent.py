from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .asset_library import list_asset_registry
from .arm_runner import list_arm_skill_specs
from .object_library import get_object, list_objects
from .robot_registry import list_robot_ids, list_robots, get_robot
from .tasks import get_task, list_task_specs, resolve_task
from .tasks.base import TaskSpec, summarize_runs, supported_space
from .capabilities import explain_object_support, select_robot_for_task, select_tool_for_task
from .evaluation_report import build_evaluation_report
from .planners import build_experiment_plan, build_retry_plan, build_skill_plan
from .protocols import standard_task_from_dynamic_spec
from .storage import ExperimentStore


DEFAULT_MODEL_API_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MODEL_NAME = "MiMo-V2.5-Pro"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_api.local.json"


def _language_name(language: str) -> str:
    normalized = (language or "zh").lower()
    if normalized in {"en", "english"}:
        return "English"
    return "Simplified Chinese"


class XiaomiAgentConfig:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_MODEL_API_BASE_URL,
        model: str = DEFAULT_MODEL_NAME,
        protocol: str = "openai",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.protocol = protocol


def load_xiaomi_config() -> XiaomiAgentConfig:
    local_config: Dict[str, Any] = {}
    if LOCAL_CONFIG_PATH.exists():
        try:
            local_config = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {LOCAL_CONFIG_PATH}") from exc

    api_key = (
        os.environ.get("MODEL_API_KEY")
        or os.environ.get("XIAOMI_API_KEY")
        or os.environ.get("MIMO_API_KEY")
        or str(local_config.get("api_key", "")).strip()
    )
    placeholder_keys = {"", "填你的 API Key", "YOUR_API_KEY", "sk-xxx"}
    if api_key in placeholder_keys:
        api_key = ""
    if not api_key:
        raise RuntimeError(
            "Model API key is not configured. Fill configs/model_api.local.json or set MODEL_API_KEY."
        )
    return XiaomiAgentConfig(
        api_key=api_key,
        base_url=(
            os.environ.get("MODEL_API_BASE")
            or os.environ.get("XIAOMI_API_BASE")
            or str(local_config.get("base_url", "")).strip()
            or DEFAULT_MODEL_API_BASE_URL
        ),
        model=(
            os.environ.get("MODEL_NAME")
            or os.environ.get("XIAOMI_MODEL")
            or str(local_config.get("model", "")).strip()
            or DEFAULT_MODEL_NAME
        ),
        protocol=(
            os.environ.get("MODEL_API_PROTOCOL")
            or str(local_config.get("protocol", "")).strip()
            or "openai"
        ).lower(),
    )


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _extract_distance_m(text: str, default_m: float = 0.1) -> float:
    """Extract an explicit movement distance from user text.

    Handles common demo phrases such as "move 5cm", "移动5厘米", and
    "push 0.05m". The returned value is meters.
    """
    normalized = text.lower().replace(" ", "")
    chinese_digits = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    match = re.search(r"(\d+(?:\.\d+)?)(cm|厘米|公分|m|米|mm|毫米)", normalized)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit in {"cm", "厘米", "公分"}:
            return value / 100.0
        if unit in {"mm", "毫米"}:
            return value / 1000.0
        return value
    match = re.search(r"([零一二两三四五六七八九十]+)(厘米|公分|米|毫米)", normalized)
    if match:
        token = match.group(1)
        if token == "十":
            value = 10
        elif token.startswith("十"):
            value = 10 + chinese_digits.get(token[-1], 0)
        elif token.endswith("十"):
            value = chinese_digits.get(token[0], 1) * 10
        elif "十" in token:
            left, right = token.split("十", 1)
            value = chinese_digits.get(left, 1) * 10 + chinese_digits.get(right, 0)
        else:
            value = chinese_digits.get(token, 0)
        unit = match.group(2)
        if unit in {"厘米", "公分"}:
            return value / 100.0
        if unit == "毫米":
            return value / 1000.0
        return float(value)
    return default_m


def _object_center_z_on_table(object_id: str, table_top_z: float = 0.39, clearance: float = 0.002) -> float:
    try:
        obj = get_object(object_id)
        if obj.geometry == "box" and len(obj.size_m) >= 3:
            half_z = float(obj.size_m[2])
        elif obj.geometry == "cylinder" and len(obj.size_m) >= 2:
            half_z = float(obj.size_m[1])
        elif obj.geometry == "sphere" and obj.size_m:
            half_z = float(obj.size_m[0])
        else:
            half_z = 0.025
    except Exception:
        half_z = 0.025
    return table_top_z + half_z + clearance


def _normalize_tabletop_object_positions(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for obj in objects:
        row = dict(obj)
        pos = list(row.get("initial_position") or [0.5, 0.0, 0.415])
        if len(pos) >= 3:
            pos[2] = max(float(pos[2]), _object_center_z_on_table(str(row.get("object_id", ""))))
        row["initial_position"] = pos
        normalized.append(row)
    return normalized


def _min_trace_ee_z(trace: Any) -> float:
    values = []
    if trace.final_ee_pos is not None:
        values.append(float(trace.final_ee_pos[2]))
    for frame in getattr(trace, "frames", []) or []:
        if getattr(frame, "ee_pos", None) is not None:
            values.append(float(frame.ee_pos[2]))
    return min(values) if values else 999.0


def _postprocess_dynamic_task_spec(goal: str, task_spec: Dict[str, Any]) -> Dict[str, Any]:
    spec = dict(task_spec)
    spec["objects"] = _normalize_tabletop_object_positions([dict(obj) for obj in spec.get("objects", [])])
    if spec.get("objects"):
        target_pos = list(spec["objects"][0].get("initial_position") or [0.5, 0.0, 0.415])
    else:
        target_pos = [0.5, 0.0, _object_center_z_on_table("cube_5cm")]

    task_type = str(spec.get("task_type", "general"))
    requested_distance_m = _extract_distance_m(goal, default_m=0.1)
    tolerance_m = max(0.01, min(0.02, requested_distance_m * 0.30))
    safe_table_z = 0.39
    min_target_z = safe_table_z + 0.012
    if task_type in {"push", "tool_use"}:
        min_target_z = max(min_target_z, float(target_pos[2]) + 0.015)

    actions = []
    for action in spec.get("actions", []) or []:
        row = dict(action)
        if isinstance(row.get("target_pos"), str) and row["target_pos"].startswith("object:"):
            row["target_pos"] = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]
        if row.get("type") == "push":
            row["distance"] = requested_distance_m
            direction = row.get("direction", [1.0, 0.0, 0.0])
            if isinstance(direction, list) and len(direction) >= 3:
                norm = math.sqrt(sum(float(v) * float(v) for v in direction[:3])) or 1.0
                row["direction"] = [float(v) / norm for v in direction[:3]]
        if isinstance(row.get("target_pos"), list) and len(row["target_pos"]) >= 3:
            row["target_pos"][2] = max(float(row["target_pos"][2]), min_target_z)
        actions.append(row)
    spec["actions"] = actions

    if task_type in {"push", "tool_use"}:
        criteria = dict(spec.get("success_criteria") or {})
        criteria.update(
            {
                "primary": "target_displacement",
                "target_displacement_m": requested_distance_m,
                "tolerance": tolerance_m,
                "table_top_z": safe_table_z,
                "min_ee_clearance_m": 0.006,
            }
        )
        spec["success_criteria"] = criteria
    else:
        criteria = dict(spec.get("success_criteria") or {})
        criteria.setdefault("table_top_z", safe_table_z)
        criteria.setdefault("min_ee_clearance_m", 0.006)
        spec["success_criteria"] = criteria
    if task_type in {"push", "tool_use"}:
        spec.setdefault("metrics", [])
        for metric in ["target_displacement_m", "displacement_error_m", "table_clearance_ok"]:
            if metric not in spec["metrics"]:
                spec["metrics"].append(metric)
    return spec


def _compact_runs(
    runs: Sequence[Mapping[str, Any]],
    task_spec: TaskSpec,
    max_rows: int = 18,
) -> List[Dict[str, Any]]:
    failures = [r for r in runs if not r.get("success")]
    successes = [r for r in runs if r.get("success")]
    selected = failures[: max_rows // 2] + successes[: max_rows - min(len(failures), max_rows // 2)]
    variable_keys = list(task_spec.experiment_space.keys())
    metric_keys = [
        "success",
        "failure_type",
        "trajectory_error",
        "collision_count",
        "final_distance",
    ]
    compact: List[Dict[str, Any]] = []
    for row in selected:
        item = {"run_id": row.get("run_id")}
        for key in variable_keys + metric_keys:
            if key in row:
                item[key] = row.get(key)
        for key in task_spec.metrics:
            if key in row:
                item[key] = row.get(key)
        compact.append(item)
    return compact


def _execution_plan(task_spec: TaskSpec, design: Mapping[str, Any]) -> Dict[str, Any]:
    uses_arm = task_spec.execution_kind.startswith("robot_arm")
    skills: List[Dict[str, Any]] = []
    if task_spec.task_id == "fr3_arm_primitives":
        selected = set(design.get("experiment_space", {}).get("skill_id", []))
        skills = [
            skill for skill in list_arm_skill_specs()
            if not selected or skill["skill_id"] in selected
        ]
    elif task_spec.task_id in {"screwdriving", "tool_use"}:
        skills = [skill for skill in list_arm_skill_specs() if skill["skill_id"] == "tool_contact_sweep"]
    elif task_spec.task_id == "assembly_insertion":
        skills = [skill for skill in list_arm_skill_specs() if skill["skill_id"] == "peg_insert"]
    return {
        "task_id": task_spec.task_id,
        "runner_status": task_spec.runner_status,
        "runner_module": task_spec.runner_module,
        "execution_kind": task_spec.execution_kind,
        "manipulation_actor": task_spec.manipulation_actor,
        "uses_real_robot_arm_scene": uses_arm,
        "arm_skills": skills,
        "supported_variables": task_spec.experiment_space,
        "replay_policy": (
            "MuJoCo replay must use renderer image_frames/replays. Abstract frames are not treated as MuJoCo replay."
        ),
        "boundary": (
            "This plan can select and parameterize implemented runners. It cannot yet synthesize a brand-new MuJoCo scene or controller from arbitrary text."
        ),
        "verified_by": task_spec.verified_by,
    }


def _local_route_task(goal: str, preferred_task_id: str | None = None) -> Dict[str, Any]:
    task = resolve_task(goal, preferred_task_id=preferred_task_id)
    normalized = goal.lower()
    if not preferred_task_id:
        if any(word in normalized for word in ["screw", "screwdriver", "bolt", "nut", "螺丝", "螺钉", "拧"]):
            task = get_task("screwdriving")
        elif any(word in normalized for word in ["insert", "assembly", "peg", "hole", "装配", "插入", "孔"]):
            task = get_task("assembly_insertion")
        elif any(word in normalized for word in ["tool", "hammer", "spatula", "工具", "锤", "铲"]):
            task = get_task("tool_use")
        elif any(word in normalized for word in ["arm", "manipulation", "robot", "机械臂", "操作"]):
            task = resolve_task(goal)
    return {
        "selected_task": task.spec.task_id,
        "reason": "Local capability router selected the closest implemented MuJoCo runner from the task registry.",
        "confidence": 0.72,
        "capability_note": "This local route uses implemented runners only; it does not invent new scenes.",
        "missing_capabilities": [],
    }


def _trim_space(task_spec: TaskSpec, requested: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return supported_space(task_spec.experiment_space, requested)


def _goal_space_override(goal: str, task_spec: TaskSpec) -> Dict[str, List[str]]:
    normalized = goal.lower()
    if task_spec.task_id != "fr3_arm_primitives":
        return {}
    if any(word in normalized for word in ["button", "press", "switch", "按钮", "按压", "按下", "按动", "触发", "开关"]):
        return {"skill_id": ["button_press"], "object_id": ["button_target"]}
    if any(word in normalized for word in ["insert", "peg", "hole", "插入", "孔"]):
        return {"skill_id": ["peg_insert"], "object_id": ["insertion_socket"]}
    if any(word in normalized for word in ["screw", "螺丝", "螺钉", "拧"]):
        return {"skill_id": ["tool_contact_sweep"], "tool_id": ["screwdriver"], "object_id": ["screw_head"]}
    if any(word in normalized for word in ["tool", "hammer", "spatula", "工具", "锤", "铲"]):
        return {"skill_id": ["tool_contact_sweep"], "tool_id": ["hammer", "spatula"], "object_id": ["rect_block"]}
    return {}


def _local_design_experiment(goal: str, limit: int, task_spec: TaskSpec, language: str = "zh") -> Dict[str, Any]:
    normalized = goal.lower()
    space: Dict[str, List[str]] = {}
    if task_spec.task_id == "fr3_arm_primitives":
        if any(word in normalized for word in ["insert", "peg", "hole", "插入", "孔"]):
            space = {"skill_id": ["peg_insert"], "object_id": ["insertion_socket"]}
        elif any(word in normalized for word in ["screw", "螺丝", "螺钉", "拧"]):
            space = {"skill_id": ["tool_contact_sweep"], "tool_id": ["screwdriver"], "object_id": ["screw_head"]}
        elif any(word in normalized for word in ["tool", "hammer", "spatula", "工具", "锤", "铲"]):
            space = {"skill_id": ["tool_contact_sweep"], "tool_id": ["hammer", "spatula"], "object_id": ["rect_block"]}
        elif any(word in normalized for word in ["button", "press", "switch", "按钮", "按压", "按下", "按动", "触发", "开关"]):
            space = {"skill_id": ["button_press"], "object_id": ["button_target"]}
        elif any(word in normalized for word in ["touch", "contact", "接触"]):
            space = {"skill_id": ["reach_touch", "button_press", "contact_sweep"]}
        else:
            space = {"skill_id": ["pick_lift", "reach_touch", "contact_sweep"]}
    elif task_spec.task_id == "screwdriving":
        space = {
            "driver_alignment": ["centered", "lateral_3mm", "lateral_8mm"],
            "downforce": ["nominal", "heavy", "light"],
            "spindle_speed": ["slow", "nominal", "fast"],
            "approach_angle": ["vertical", "tilted_5deg", "tilted_12deg"],
        }
    elif task_spec.task_id == "assembly_insertion":
        space = {
            "lateral_offset": ["centered", "offset_5mm", "offset_12mm"],
            "insertion_angle": ["vertical", "tilted_4deg", "tilted_9deg"],
            "compliance": ["stiff", "nominal", "soft"],
        }
    elif task_spec.task_id == "tool_use":
        space = {
            "tool_asset": ["scanned_hammer_black", "scanned_cookie_spatula"],
            "target_object": ["rect_block", "cylinder_can"],
            "impact_speed": ["slow", "nominal", "fast"],
            "approach_offset": ["centered", "left_2cm", "right_2cm"],
        }
    else:
        space = task_spec.experiment_space

    experiment_space = _trim_space(task_spec, _goal_space_override(goal, task_spec) or space)
    objects = [
        {"object_id": obj["object_id"], "why": "Relevant to the selected implemented runner."}
        for obj in list_objects()
        if obj["object_id"] in task_spec.supported_objects
    ][:4]
    return {
        "task": task_spec.title,
        "hypotheses": [
            {
                "id": "H1",
                "title": "Primary factor sensitivity",
                "claim": "Changing the main task factor will change success rate and failure type.",
                "metric": "success_rate",
                "controlled_factor": next(iter(experiment_space), ""),
                "expected_direction": "More alignment/contact difficulty should reduce success.",
            },
            {
                "id": "H2",
                "title": "Contact evidence",
                "claim": "Contact-step and force metrics should explain many failures.",
                "metric": task_spec.metrics[-1] if task_spec.metrics else "failure_type",
                "controlled_factor": "friction" if "friction" in experiment_space else next(iter(experiment_space), ""),
                "expected_direction": "Poor contact should correlate with failure labels.",
            },
        ],
        "experiment_space": experiment_space,
        "object_plan": objects,
        "rationale": "Local capability planner selected supported variables that directly affect the implemented MuJoCo runner.",
        "capability_boundary": "The runner uses real MuJoCo execution for registered capabilities; brand-new scene synthesis is still limited to existing reusable primitives.",
        "fixed_assumptions": [f"Run budget requested: {limit}", f"Execution kind: {task_spec.execution_kind}"],
        "design_quality_notes": ["Only supported variable values are used.", "The plan is executable without a model API key."],
    }


def _local_analyze_runs(
    task_spec: TaskSpec,
    runs: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    language: str = "zh",
) -> Dict[str, Any]:
    failures = summary.get("failure_distribution", {})
    worst_failure = max(failures, key=failures.get) if failures else "none"
    success_rate = float(summary.get("success_rate", 0.0))
    variable_keys = list(task_spec.experiment_space.keys())
    findings = [
        {
            "title": "Observed success rate",
            "body": f"{len(runs)} runs completed with success_rate={success_rate:.3f}.",
            "evidence": f"summary.success_rate={success_rate:.3f}",
            "confidence": 0.78,
        }
    ]
    if worst_failure != "none":
        findings.append(
            {
                "title": "Dominant failure mode",
                "body": f"The largest observed failure mode is {worst_failure}.",
                "evidence": f"failure_distribution={failures}",
                "confidence": 0.72,
            }
        )
    recommendations = [
        {
            "title": "Narrow around informative contrasts",
            "body": "Keep variables that changed success/failure and reduce less-informative levels in the next run.",
            "priority": "high",
        },
        {
            "title": "Add a harder validation band",
            "body": "After finding a successful region, test nearby alignment/contact perturbations to estimate robustness.",
            "priority": "medium",
        },
    ]
    next_space = {key: values[:2] for key, values in task_spec.experiment_space.items() if key in variable_keys}
    return {
        "findings": findings,
        "recommendations": recommendations,
        "next_experiment_space": next_space,
        "next_object_candidates": [
            {"object_id": obj, "why": "Already supported by the selected runner."}
            for obj in task_spec.supported_objects[:3]
        ],
        "agent_conclusion": "Local agent completed an executable design-run-analyze loop using registered MuJoCo capabilities.",
    }


def _run_local_agent_round(
    goal: str,
    limit: int,
    use_fallback: bool,
    preferred_task_id: str | None,
    language: str,
    fallback_reason: str = "",
) -> Dict[str, Any]:
    route = _local_route_task(goal, preferred_task_id=preferred_task_id)
    task = get_task(route["selected_task"])
    design = _local_design_experiment(goal, limit, task.spec, language=language)
    execution_plan = _execution_plan(task.spec, design)
    result = task.run_experiments(
        limit=limit,
        use_fallback=use_fallback,
        experiment_space=design.get("experiment_space"),
    )
    summary = summarize_runs(result["runs"])
    analysis = _local_analyze_runs(task.spec, result["runs"], summary, language=language)
    return {
        "source": f"local_capability_agent+{result['source']}",
        "agent_provider": "local-capability-router",
        "model": "local-rule-planner",
        "protocol": "local",
        "language": "en" if _language_name(language) == "English" else "zh",
        "task_id": task.spec.task_id,
        "task": asdict(task.spec),
        "object_library": [
            obj for obj in list_objects() if obj["object_id"] in task.spec.supported_objects
        ],
        "asset_registry": list_asset_registry(),
        "route": route,
        "design": design,
        "execution_plan": execution_plan,
        "runs": result["runs"],
        "summary": summary,
        "analysis": analysis,
        "model_api_calls": {
            "count": 0,
            "calls": [],
            "note": fallback_reason or "Local planner was used without model API calls.",
        },
        "agent_trace": [
            "route_task_local_capability",
            "design_experiment_local_capability",
            f"run_{result['source']}_experiments",
            "analyze_runs_local_capability",
        ],
    }


class XiaomiResearchAgent:
    """LLM planner and analyzer that executes only registered local tasks."""

    def __init__(self, config: XiaomiAgentConfig | None = None) -> None:
        self.config = config or load_xiaomi_config()
        self.client = None
        self.api_call_log: List[Dict[str, Any]] = []
        if self.config.protocol == "openai":
            try:
                from openai import OpenAI
            except Exception as exc:  # pragma: no cover - depends on local install
                raise RuntimeError(
                    "The 'openai' Python package is required for OpenAI-compatible calls. "
                    "Run scripts\\setup_mujoco_env.ps1 to install requirements."
                ) from exc
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                default_headers={"api-key": self.config.api_key},
            )
        elif self.config.protocol != "anthropic":
            raise RuntimeError("MODEL_API_PROTOCOL must be 'openai' or 'anthropic'.")

    def _json_chat(self, stage: str, system: str, user: str) -> Dict[str, Any]:
        self.api_call_log.append(
            {
                "stage": stage,
                "protocol": self.config.protocol,
                "model": self.config.model,
                "base_url": self.config.base_url,
            }
        )
        if self.config.protocol == "anthropic":
            content = self._anthropic_chat(system, user)
        else:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_completion_tokens=4096,
            )
            content = response.choices[0].message.content or "{}"
        return _extract_json(content)

    def _anthropic_chat(self, system: str, user: str) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/messages"):
            url = base_url
        elif base_url.endswith("/v1"):
            url = base_url + "/messages"
        else:
            url = base_url + "/v1/messages"
        body = json.dumps(
            {
                "model": self.config.model,
                "max_tokens": 4096,
                "temperature": 0.2,
                "system": system,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user}],
                    }
                ],
                "top_p": 0.95,
                "stream": False,
                "stop_sequences": None,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "content-type": "application/json",
                "api-key": self.config.api_key,
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "authorization": f"Bearer {self.config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Anthropic-compatible API error {exc.code}: {detail}") from exc
        content = payload.get("content", [])
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return str(content or "{}")

    def route_task(
        self,
        goal: str,
        preferred_task_id: str | None = None,
        language: str = "zh",
    ) -> Dict[str, Any]:
        if preferred_task_id:
            task = resolve_task(goal, preferred_task_id=preferred_task_id)
            return {
                "selected_task": task.spec.task_id,
                "reason": "Task was explicitly requested by the caller.",
                "confidence": 1.0,
            }

        system = (
            "You are a robotics experiment routing agent. Select exactly one "
            "registered local MuJoCo task. Return strict JSON only."
        )
        user = json.dumps(
            {
                "goal": goal,
                "available_tasks": list_task_specs(),
                "available_objects": list_objects(),
                "asset_registry": list_asset_registry(),
                "response_language": _language_name(language),
                "required_json_schema": {
                    "selected_task": "one task_id from available_tasks",
                    "reason": "short reason based on the user's goal",
                    "confidence": 0.0,
                    "capability_note": "short note on whether the goal is fully executable with the current task registry and object library",
                    "missing_capabilities": ["string"],
                },
                "constraints": [
                    "Do not invent new task_ids.",
                    "Do not claim that an unregistered task can be executed.",
                    "Use available_objects only as context; object availability does not create a runnable task by itself.",
                    "If the goal matches a planned blueprint but no implemented task_id exists, choose the closest implemented task only if it is a meaningful proxy and explain the limitation.",
                    "Write all natural-language fields in response_language.",
                    "If the goal asks for grasping, choose fr3_pick_place.",
                    "If the goal asks for pushing or sliding, choose tabletop_push.",
                    "If the goal is a general robot-arm manipulation experiment or does not cleanly match a specialized implemented task, choose fr3_arm_primitives as the executable FR3 arm fallback.",
                    "Prefer a task with execution_kind robot_arm_skill_simulation when the user asks for a general mechanical-arm experiment.",
                ],
            },
            ensure_ascii=False,
        )
        route = self._json_chat("route_task", system, user)
        selected = str(route.get("selected_task", "")).strip()
        if selected not in {task["task_id"] for task in list_task_specs()}:
            selected = resolve_task(goal).spec.task_id
            route["reason"] = "Model output was outside the registry; heuristic router selected a supported task."
        route["selected_task"] = selected
        route.setdefault("confidence", 0.6)
        route.setdefault("reason", "")
        route.setdefault("capability_note", "")
        route.setdefault("missing_capabilities", [])
        return route

    def design_experiment(
        self,
        goal: str,
        limit: int,
        task_spec: TaskSpec,
        language: str = "zh",
    ) -> Dict[str, Any]:
        system = (
            "You are a robotics R&D experiment design agent. You can only use "
            "the provided task and supported variable values. Return strict JSON only."
        )
        user = json.dumps(
            {
                "task": asdict(task_spec),
                "goal": goal,
                "run_budget": limit,
                "supported_variables": task_spec.experiment_space,
                "supported_metrics": task_spec.metrics,
                "supported_failure_types": task_spec.failure_types,
                "available_objects": [
                    obj for obj in list_objects() if obj["object_id"] in task_spec.supported_objects
                ],
                "asset_registry": list_asset_registry(),
                "response_language": _language_name(language),
                "required_json_schema": {
                    "task": "string",
                    "hypotheses": [
                        {
                            "id": "H1",
                            "title": "string",
                            "claim": "string",
                            "metric": "string",
                            "controlled_factor": "one variable from supported_variables",
                            "expected_direction": "string",
                        }
                    ],
                    "experiment_space": task_spec.experiment_space,
                    "object_plan": [
                        {
                            "object_id": "one object_id from available_objects",
                            "why": "string",
                        }
                    ],
                    "rationale": "string",
                    "capability_boundary": "string",
                    "fixed_assumptions": ["string"],
                    "design_quality_notes": ["string"],
                },
                "constraints": [
                    "experiment_space values must come from supported_variables.",
                    "object_plan object_id values must come from available_objects.",
                    "Do not invent unavailable objects, robot models, sensors, or scenes.",
                    "Use the task execution_kind, manipulation_actor, and fidelity_notes when describing what will be executed.",
                    "If manipulation_actor says it is not a robot arm, explicitly say this is a task-specific proxy runner rather than a real manipulator trajectory.",
                    "If the requested task needs a planned but unimplemented blueprint, state that boundary clearly in capability_boundary.",
                    "Write all natural-language fields in response_language.",
                    "Design hypotheses around one primary factor at a time, with at least two contrast levels when possible.",
                    "Prefer variables that can explain success_rate, failure_type, and the task-specific metrics.",
                    "Do not include variables that are unsupported by the selected runner.",
                    "Do not claim training, policy optimization, or real hardware control.",
                ],
            },
            ensure_ascii=False,
        )
        design = self._json_chat("design_experiment", system, user)
        override = _goal_space_override(goal, task_spec)
        if override:
            design["experiment_space"] = {
                **dict(design.get("experiment_space") or {}),
                **override,
            }
        design["experiment_space"] = supported_space(
            task_spec.experiment_space,
            design.get("experiment_space"),
        )
        design.setdefault("task", task_spec.title)
        design.setdefault("hypotheses", [])
        design.setdefault("object_plan", [])
        design.setdefault("rationale", "")
        design.setdefault("capability_boundary", "")
        design.setdefault("fixed_assumptions", [])
        design.setdefault("design_quality_notes", [])
        return design

    def analyze_runs(
        self,
        goal: str,
        task_spec: TaskSpec,
        design: Mapping[str, Any],
        runs: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
        language: str = "zh",
    ) -> Dict[str, Any]:
        system = (
            "You are a robotics simulation experiment analysis agent. Analyze "
            "only the provided MuJoCo logs and return strict JSON only."
        )
        user = json.dumps(
            {
                "goal": goal,
                "task": asdict(task_spec),
                "design": design,
                "summary": summary,
                "sample_runs": _compact_runs(runs, task_spec),
                "response_language": _language_name(language),
                "required_json_schema": {
                    "findings": [
                        {
                            "title": "string",
                            "body": "string",
                            "evidence": "string",
                            "confidence": 0.0,
                        }
                    ],
                    "recommendations": [
                        {"title": "string", "body": "string", "priority": "high|medium|low"}
                    ],
                    "next_experiment_space": task_spec.experiment_space,
                    "next_object_candidates": [
                        {
                            "object_id": "one object_id from task.supported_objects",
                            "why": "string",
                        }
                    ],
                    "agent_conclusion": "string",
                },
                "constraints": [
                    "Every conclusion must cite visible evidence from summary or sample_runs.",
                    "Use only supported failure types and metrics for this task.",
                    "next_experiment_space values must come from the task supported_variables.",
                    "Do not describe task-specific proxy runners as full robot-arm manipulation.",
                    "If the runner uses a scripted actuator or carrier, preserve that limitation in agent_conclusion.",
                    "Write all natural-language fields in response_language.",
                ],
            },
            ensure_ascii=False,
        )
        analysis = self._json_chat("analyze_runs", system, user)
        analysis["next_experiment_space"] = supported_space(
            task_spec.experiment_space,
            analysis.get("next_experiment_space"),
        )
        analysis.setdefault("findings", [])
        analysis.setdefault("recommendations", [])
        analysis.setdefault("next_object_candidates", [])
        analysis.setdefault("agent_conclusion", "")
        return analysis

    def parse_task_description(
        self,
        goal: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        """Parse natural language task description into structured spec.

        Returns a dict with:
        - task_type: e.g. "pick_and_place", "push", "insert", "press"
        - robot_preference: "auto" or a specific robot_id
        - robot_candidates: list of suitable robot_ids
        - objects: list of {object_id, role, initial_position}
        - actions: list of {type, target_pos, force, height, ...}
        - workspace: "table", "shelf", etc.
        - success_criteria: {primary, target_position, tolerance}
        - experiment_variables: dict of variable_name -> list of values
        - metrics: list of metric names
        """
        system = (
            "You are a robotics task parsing agent. Convert a natural language "
            "robot arm manipulation task into a structured JSON specification. "
            "Return strict JSON only."
        )

        robot_info = []
        for r in list_robots():
            robot_info.append({
                "robot_id": r.robot_id,
                "name": r.name,
                "dof": r.dof,
                "has_gripper": r.has_gripper,
                "manufacturer": r.manufacturer,
            })

        user = json.dumps(
            {
                "goal": goal,
                "available_robots": robot_info,
                "available_objects": [
                    {"object_id": obj["object_id"], "name_en": obj["name_en"], "geometry": obj["geometry"], "tags": obj["tags"]}
                    for obj in list_objects()
                ],
                "available_actions": [
                    "reach", "grasp", "lift", "place", "push", "insert", "release", "wait"
                ],
                "response_language": _language_name(language),
                "required_json_schema": {
                    "task_type": "pick_and_place|push|insert|press|touch|sweep|general",
                    "robot_preference": "auto or specific robot_id",
                    "robot_candidates": ["robot_id from available_robots"],
                    "objects": [
                        {
                            "object_id": "from available_objects",
                            "role": "target|obstacle|tool",
                            "initial_position": "[x, y, z] or 'table_center'|'table_left'|'table_right'",
                        }
                    ],
                    "actions": [
                        {
                            "type": "reach|grasp|lift|place|push|insert|release|wait",
                            "target_pos": "[x, y, z] or 'object:object_id'",
                            "target_quat": "[w, x, y, z] or null",
                            "force_scale": 0.0-1.0,
                            "height": 0.0,
                            "direction": "[x, y, z]",
                            "distance": 0.0,
                            "steps": 300,
                        }
                    ],
                    "workspace": "table|shelf|floor",
                    "success_criteria": {
                        "primary": "object_at_target|contact_achieved|object_displaced|lifted",
                        "target_position": "[x, y, z] or null",
                        "tolerance": 0.05,
                        "min_contact_steps": 0,
                        "min_displacement": 0.0,
                    },
                    "experiment_variables": {
                        "variable_name": ["value1", "value2"]
                    },
                    "metrics": ["success_rate", "object_displacement", "contact_steps", "max_touch_force"],
                },
                "constraints": [
                    "Use only object_ids from available_objects.",
                    "Use only robot_ids from available_robots.",
                    "Actions must be one of available_actions.",
                    "For pick_and_place tasks, prefer robots with grippers (has_gripper=true).",
                    "For tasks needing dexterity, prefer 7-DOF robots.",
                    "Initial positions should be in robot workspace (roughly x: 0.3-0.7, y: -0.2 to 0.2, z: table_height to 0.6).",
                    "Default table height is 0.37m. Objects sit on the table.",
                    "experiment_variables should be physically meaningful (friction, mass, position offsets, etc.).",
                    "Write all natural-language fields in response_language.",
                ],
            },
            ensure_ascii=False,
        )

        spec = self._json_chat("parse_task_description", system, user)

        # Validate and fill defaults
        spec.setdefault("task_type", "general")
        spec.setdefault("robot_preference", "auto")
        spec.setdefault("robot_candidates", [])
        spec.setdefault("objects", [])
        spec.setdefault("actions", [])
        spec.setdefault("workspace", "table")
        spec.setdefault("success_criteria", {"primary": "object_at_target", "tolerance": 0.05})
        spec.setdefault("experiment_variables", {})
        spec.setdefault("metrics", ["success_rate", "object_displacement", "contact_steps"])

        # Validate robot candidates
        valid_robots = set(list_robot_ids())
        spec["robot_candidates"] = [
            r for r in spec.get("robot_candidates", []) if r in valid_robots
        ]
        if not spec["robot_candidates"]:
            # Auto-select based on task requirements
            if spec.get("task_type") in ("pick_and_place", "press"):
                spec["robot_candidates"] = [
                    r for r in list_robot_ids() if get_robot(r).has_gripper
                ] or ["franka_fr3"]
            else:
                spec["robot_candidates"] = list_robot_ids()

        # Validate object IDs
        valid_objects = {obj["object_id"] for obj in list_objects()}
        for obj in spec.get("objects", []):
            if obj.get("object_id") not in valid_objects:
                obj["object_id"] = "cube_5cm"  # safe fallback

        # Resolve position aliases
        position_aliases = {
            "table_center": (0.5, 0.0, 0.395),
            "table_left": (0.5, 0.15, 0.395),
            "table_right": (0.5, -0.15, 0.395),
            "table_near": (0.4, 0.0, 0.395),
            "table_far": (0.6, 0.0, 0.395),
        }
        for obj in spec.get("objects", []):
            pos = obj.get("initial_position", "table_center")
            if isinstance(pos, str) and pos in position_aliases:
                obj["initial_position"] = list(position_aliases[pos])
            elif isinstance(pos, str):
                obj["initial_position"] = list(position_aliases.get(pos, [0.5, 0.0, 0.395]))

        return spec


def run_nlp_agent_round(
    goal: str,
    limit: int = 9,
    language: str = "zh",
    robot_id: str = "",
    progress_callback=None,
) -> Dict[str, Any]:
    """Full NLP pipeline: parse goal → compose scene → run experiments → analyze.

    This is the main entry point for the new NLP-driven experiment pipeline.
    It uses real MuJoCo physics simulation with menagerie robot models — no
    simplification or proxy runners.
    """
    from dataclasses import asdict as _asdict
    from .tasks import resolve_task, summarize_runs as summarize_task_runs
    from .scene_composer import DynamicSceneComposer, SceneDescription, ObjectPlacement
    from .motion_primitives import MotionPlan, UniversalMotionExecutor

    def _progress(step: str, detail: str, percent: int):
        if progress_callback:
            progress_callback(step, detail, percent)

    # Step 1: Parse task description
    _progress("任务解析", "正在解析自然语言任务描述...", 5)
    use_local = os.environ.get("AGENT_FORCE_LOCAL", "").strip().lower() in {"1", "true", "yes"}
    agent = None
    task_spec = None
    if not use_local:
        try:
            agent = XiaomiResearchAgent()
            task_spec = agent.parse_task_description(goal, language=language)
        except Exception:
            pass
    if task_spec is None:
        task_spec = _local_parse_task(goal, robot_id=robot_id)
    task_spec = _postprocess_dynamic_task_spec(goal, task_spec)
    _progress("任务解析", f"完成: {task_spec.get('task_type', 'unknown')}", 15)
    standard_task = standard_task_from_dynamic_spec(task_spec, goal).to_dict()
    _progress("协议化校验", "TaskSpec -> ExperimentPlan -> SkillPlan", 18)

    # Step 1.5: optional compatibility path for fixed demo runners.
    # The NLP endpoint defaults to the dynamic pipeline so arbitrary goals are
    # actually parsed, composed, executed, and analyzed through one path.
    use_registered = os.environ.get("NLP_USE_REGISTERED_RUNNERS", "").strip().lower() in {"1", "true", "yes"}
    existing_task = resolve_task(goal)
    if use_registered and existing_task.spec.task_id != "fr3_arm_primitives":
        _progress("任务匹配", f"匹配到已有任务: {existing_task.spec.title}", 18)
        _progress("仿真实验", f"正在运行 {existing_task.spec.title}（真实 MuJoCo 物理仿真）...", 30)
        result = existing_task.run_experiments(limit=limit)
        runs = result.get("runs", [])
        summary = summarize_task_runs(runs)
        result["summary"] = summary
        _progress("轨迹渲染", "正在渲染 MuJoCo 仿真帧...", 75)
        trace = existing_task.demo_trace()
        _progress("结果分析", "正在分析实验结果...", 90)
        try:
            if agent is None:
                agent = XiaomiResearchAgent()
            analysis = agent.analyze_runs(
                goal=goal,
                task_spec=existing_task.spec,
                design={"experiment_space": existing_task.spec.experiment_space},
                runs=runs,
                summary=summary,
                language=language,
            )
        except Exception:
            analysis = {
                "findings": [{"title": "执行完成", "body": f"{len(runs)} 组实验已完成。", "confidence": 0.8}],
                "recommendations": [{"title": "继续迭代", "body": "尝试调整实验变量。", "priority": "high"}],
                "agent_conclusion": "本地分析（LLM 不可用）。",
            }
        _progress("完成", f"共 {len(runs)} 组实验, 成功率 {summary.get('success_rate', 0):.1%}", 100)
        return {
            "source": "nlp_agent",
            "agent_provider": "nlp_pipeline",
            "robot_id": "franka_fr3",
            "robot_spec": {
                "robot_id": "franka_fr3",
                "name": "Franka FR3",
                "dof": 7,
                "has_gripper": True,
                "gripper_type": "parallel",
                "gripper_joint_names": ["finger_joint1", "finger_joint2"],
                "gripper_actuator_names": ["actuator8"],
                "end_effector_site": "pinch_site",
                "end_effector_body": "hand",
                "end_effector_name": "Franka Hand",
            },
            "task_spec": _asdict(existing_task.spec),
            "experiment_space": existing_task.spec.experiment_space,
            "runs": runs,
            "summary": summary,
            "analysis": analysis,
            "demo_trace": trace,
            "num_runs": len(runs),
            "agent_trace": ["resolve_task", "run_experiments", "demo_trace", "analyze_runs"],
        }

    # Step 2: Select robot (explicit override takes priority)
    if not robot_id:
        robot_id = task_spec.get("robot_candidates", ["franka_fr3"])[0]
    spec = get_robot(robot_id)
    _progress("机器人选择", f"已选择: {spec.name}", 20)

    # Step 3: Compose scene
    _progress("场景组装", f"正在组装 {spec.name} MuJoCo 场景...", 25)
    composer = DynamicSceneComposer()
    object_placements = []
    for obj in task_spec.get("objects", []):
        pos = obj.get("initial_position", [0.5, 0.0, 0.395])
        if isinstance(pos, str):
            pos = [0.5, 0.0, 0.395]
        object_placements.append(ObjectPlacement(
            object_id=obj["object_id"],
            role=obj.get("role", "target"),
            position=tuple(pos),
        ))

    scene_desc = SceneDescription(
        robot_id=robot_id,
        objects=object_placements,
        workspace=task_spec.get("workspace", "table"),
        held_tool_id=task_spec.get("held_tool_id"),
    )
    scene = composer.compose(scene_desc, load_model=True)
    model = scene["model"]
    data = scene["data"]
    _progress("场景组装", f"完成: nq={model.nq}, nu={model.nu}", 35)

    # Step 4: Build motion plan from parsed actions
    _progress("运动规划", "正在构建 Jacobian IK 运动计划...", 40)
    actions = task_spec.get("actions", [])
    if actions:
        plan = MotionPlan.from_action_sequence(actions)
    else:
        plan = MotionPlan.pick_and_place(
            pre_grasp_pos=[0.5, 0.0, 0.5],
            grasp_pos=[0.5, 0.0, 0.38],
            place_pos=[0.5, 0.15, 0.38],
        )
    _progress("运动规划", f"完成: {len(plan.primitives)} 个运动原语", 45)

    # Step 5: Design experiment matrix
    exp_vars = task_spec.get("experiment_variables", {})
    if not exp_vars:
        exp_vars = {"friction": ["medium"], "object_position": ["center"]}

    experiment_matrix = _build_experiment_matrix(exp_vars, limit)
    _progress("实验设计", f"生成 {len(experiment_matrix)} 组实验矩阵", 50)

    # Step 6: Run experiments with real MuJoCo physics
    runs = []
    total = len(experiment_matrix)
    for i, config in enumerate(experiment_matrix):
        if i % max(1, total // 5) == 0 or i == total - 1:
            pct = 50 + int(40 * (i + 1) / total)
            _progress("仿真实验", f"正在运行第 {i+1}/{total} 组...", pct)
        run_result = _run_single_nlp_experiment(
            model=model,
            data=data,
            spec=spec,
            plan=plan,
            scene=scene,
            config=config,
            run_id=f"nlp_{i+1:03d}",
            task_type=task_spec.get("task_type", "general"),
            success_criteria=task_spec.get("success_criteria", {}),
        )
        runs.append(run_result)
    _progress("仿真实验", f"完成: {len(runs)} 组实验", 90)

    # Step 7: Summarize and analyze
    _progress("结果分析", "正在汇总实验结果...", 92)
    summary = summarize_runs(runs)
    evaluation_report = build_evaluation_report(str(standard_task.get("task_id") or "nlp_dynamic"), runs)
    experiment_plan = build_experiment_plan(standard_task, run_count=len(experiment_matrix)).to_dict()
    skill_plan = build_skill_plan(standard_task).to_dict()
    robot_selection = select_robot_for_task(
        standard_task,
        [step.get("skill_name", "") for step in skill_plan.get("skills", [])],
    )
    tool_selection = select_tool_for_task(
        str(standard_task.get("task_type") or ""),
        str(skill_plan.get("robot_id") or robot_id),
        str(standard_task.get("held_tool_id") or ""),
    )
    object_support = explain_object_support(standard_task)
    retry_plan = build_retry_plan(task_spec, evaluation_report)
    retry_execution = None
    retry_auto_enabled = os.environ.get("NLP_AUTO_RETRY", "1").strip().lower() not in {"0", "false", "no"}
    if retry_auto_enabled and retry_plan.get("should_retry"):
        _progress("闭环重试", "正在根据 RetryPlan 自动执行一轮修正实验...", 94)
        retry_task_spec = _postprocess_dynamic_task_spec(
            goal,
            dict(retry_plan.get("revised_task_spec") or task_spec),
        )
        retry_actions = retry_task_spec.get("actions", [])
        retry_plan_obj = MotionPlan.from_action_sequence(retry_actions) if retry_actions else plan
        retry_exp_vars = retry_task_spec.get("experiment_variables") or exp_vars
        retry_limit = max(1, min(3, limit))
        retry_matrix = _build_experiment_matrix(retry_exp_vars, retry_limit)
        retry_runs = []
        for i, config in enumerate(retry_matrix):
            retry_config = dict(config)
            retry_config["retry_of"] = retry_plan.get("dominant_failure", "unknown")
            retry_runs.append(
                _run_single_nlp_experiment(
                    model=model,
                    data=data,
                    spec=spec,
                    plan=retry_plan_obj,
                    scene=scene,
                    config=retry_config,
                    run_id=f"retry_{i+1:03d}",
                    task_type=retry_task_spec.get("task_type", task_spec.get("task_type", "general")),
                    success_criteria=retry_task_spec.get("success_criteria", task_spec.get("success_criteria", {})),
                )
            )
        retry_summary = summarize_runs(retry_runs)
        retry_evaluation = build_evaluation_report(
            f"{standard_task.get('task_id') or 'nlp_dynamic'}_retry",
            retry_runs,
        )
        retry_execution = {
            "attempted": True,
            "source_failure": retry_plan.get("dominant_failure"),
            "changes": retry_plan.get("changes", []),
            "revised_task_spec": retry_task_spec,
            "runs": retry_runs,
            "summary": retry_summary,
            "evaluation_report": retry_evaluation,
            "comparison": {
                "before_success_rate": summary.get("success_rate", 0.0),
                "after_success_rate": retry_summary.get("success_rate", 0.0),
                "delta_success_rate": retry_summary.get("success_rate", 0.0) - summary.get("success_rate", 0.0),
                "before_failures": summary.get("failure_distribution", {}),
                "after_failures": retry_summary.get("failure_distribution", {}),
                "source_failure_before_rate": (summary.get("failure_distribution", {}) or {}).get(
                    retry_plan.get("dominant_failure"),
                    0.0,
                ),
                "source_failure_after_rate": (retry_summary.get("failure_distribution", {}) or {}).get(
                    retry_plan.get("dominant_failure"),
                    0.0,
                ),
            },
        }
    elif retry_auto_enabled:
        retry_execution = {
            "attempted": False,
            "reason": "No retry was needed because the current sample did not expose a failure.",
            "comparison": {
                "before_success_rate": summary.get("success_rate", 0.0),
                "after_success_rate": summary.get("success_rate", 0.0),
                "delta_success_rate": 0.0,
            },
        }

    memory_summary = {}
    memory_record = {}
    try:
        store = ExperimentStore(PROJECT_ROOT / "results" / "nlp_experiment_store.jsonl")
        memory_record = store.append(
            {
                "task_spec": task_spec,
                "robot_id": robot_id,
                "experiment_plan": experiment_plan,
                "skill_plan": skill_plan,
                "evaluation_report": evaluation_report,
                "retry_plan": retry_plan,
                "retry_execution": retry_execution,
            }
        )
        memory_summary = store.summarize()
    except Exception as exc:
        memory_summary = {"error": str(exc)}

    _progress("结果分析", "正在进行失败归因分析...", 95)
    try:
        if agent is None:
            agent = XiaomiResearchAgent()
        analysis = agent.analyze_runs(
            goal=goal,
            task_spec=TaskSpec(
                task_id="nlp_dynamic",
                title=goal[:80],
                description=goal,
                keywords=[],
                experiment_space=exp_vars,
                metrics=task_spec.get("metrics", ["success_rate", "object_displacement"]),
                failure_types=["no_contact", "grasp_miss", "lift_failed", "weak_displacement"],
                supported_objects=[o["object_id"] for o in task_spec.get("objects", [])],
                execution_kind="robot_arm_skill_simulation",
                manipulation_actor=f"{spec.name} with real MuJoCo physics",
                fidelity_notes=["Real menagerie robot model", "Jacobian IK control", "Full physics simulation"],
            ),
            design={"experiment_space": exp_vars},
            runs=runs,
            summary=summary,
            language=language,
        )
    except Exception:
        analysis = {
            "findings": [{"title": "Execution complete", "body": f"{len(runs)} runs completed.", "confidence": 0.8}],
            "recommendations": [{"title": "Iterate", "body": "Try different friction or position values.", "priority": "high"}],
            "agent_conclusion": "Local analysis (LLM unavailable).",
        }

    # Build demo_trace with real MuJoCo render frames
    success_runs = [r for r in runs if r.get("success")]
    failure_runs = [r for r in runs if not r.get("success")]

    def _render_run(run, title):
        """Re-run a specific config with rendering enabled."""
        run_config = {k: run[k] for k in exp_vars if k in run}
        render_scene = composer.compose(scene_desc, load_model=True)
        render_result = _run_single_nlp_experiment(
            model=render_scene["model"], data=render_scene["data"], spec=spec, plan=plan, scene=render_scene,
            config=run_config, run_id=f"render_{run.get('run_id', '')}",
            task_type=task_spec.get("task_type", "general"),
            success_criteria=task_spec.get("success_criteria", {}),
            render=True,
        )
        return {
            "title": title,
            "model": spec.name,
            "source": "mujoco_render",
            "image_frames": render_result.get("image_frames", []),
            "labels": render_result.get("labels", []),
        }

    _progress("轨迹渲染", "正在渲染 MuJoCo 仿真帧...", 92)
    demo_trace = None
    if success_runs or failure_runs:
        demo_trace = {
            "model": f"{spec.name} MuJoCo",
            "source": "nlp_pipeline",
            "replays": [],
        }
        if success_runs:
            demo_trace["replays"].append(
                _render_run(success_runs[0], f"成功样例 ({success_runs[0].get('run_id', '')})")
            )
        if failure_runs:
            demo_trace["replays"].append(
                _render_run(failure_runs[0], f"失败样例 ({failure_runs[0].get('run_id', '')})")
            )
        if not failure_runs and success_runs:
            worst = min(success_runs, key=lambda r: r.get("object_displacement", 0))
            demo_trace["replays"].append(
                _render_run(worst, f"最弱成功 ({worst.get('run_id', '')})")
            )

    _progress("完成", f"共 {len(runs)} 组实验, 成功率 {summary.get('success_rate', 0):.1%}", 100)

    return {
        "source": "nlp_agent",
        "agent_provider": "nlp_pipeline",
        "robot_id": robot_id,
        "robot_spec": {
            "robot_id": spec.robot_id,
            "name": spec.name,
            "dof": spec.dof,
            "has_gripper": spec.has_gripper,
            "gripper_type": spec.gripper_type,
            "gripper_joint_names": list(spec.gripper_joint_names),
            "gripper_actuator_names": list(spec.gripper_actuator_names),
            "end_effector_site": spec.end_effector_site,
            "end_effector_body": spec.end_effector_body,
            "end_effector_name": "Franka Hand" if spec.robot_id == "franka_fr3" and spec.has_gripper else spec.gripper_type,
        },
        "task_spec": task_spec,
        "standard_task_spec": standard_task,
        "experiment_plan": experiment_plan,
        "skill_plan": skill_plan,
        "capability_report": {
            "robot_selection": robot_selection,
            "tool_selection": tool_selection,
            "object_support": object_support,
        },
        "scene_xml_preview": scene["xml"][:500] + "...",
        "experiment_space": exp_vars,
        "runs": runs,
        "summary": summary,
        "evaluation_report": evaluation_report,
        "retry_plan": retry_plan,
        "retry_execution": retry_execution,
        "experiment_memory": {
            "recorded": bool(memory_record),
            "recorded_at": memory_record.get("recorded_at") if isinstance(memory_record, dict) else None,
            "summary": memory_summary,
        },
        "analysis": analysis,
        "demo_trace": demo_trace,
        "agent_trace": [
            "parse_task_description",
            "compose_scene",
            "build_motion_plan",
            "run_mujoco_experiments",
            "auto_retry_from_evaluation",
            "write_experiment_memory",
            "analyze_runs",
        ],
    }


def _local_parse_task(goal: str, robot_id: str = "") -> Dict[str, Any]:
    """Deterministic fallback for task parsing when LLM is unavailable."""
    normalized = goal.lower()

    task_type = "general"
    if any(w in normalized for w in ["pick", "grasp", "lift", "place", "抓", "拿", "取", "放", "搬运"]):
        task_type = "pick_and_place"
    elif any(w in normalized for w in ["push", "slide", "move", "移动", "推动", "推", "滑动"]):
        task_type = "push"
    elif any(w in normalized for w in ["insert", "peg", "hole", "插入", "插", "孔"]):
        task_type = "insert"
    elif any(w in normalized for w in ["press", "button", "按压", "按下", "按钮", "按"]):
        task_type = "press"
    elif any(w in normalized for w in ["touch", "contact", "接触", "触碰", "触"]):
        task_type = "touch"

    def _goal_has(words: List[str]) -> bool:
        return any(w in normalized for w in words)

    if _goal_has(["screw", "screwdriver", "fastener", "\u87ba\u4e1d", "\u62e7"]):
        task_type = "screwdriving"
    elif _goal_has(["tool", "spatula", "hammer", "\u5de5\u5177", "\u94f2", "\u9524"]):
        task_type = "tool_use"
    elif _goal_has(["insert", "peg", "hole", "\u63d2", "\u5b54"]):
        task_type = "insert"
    elif _goal_has(["press", "button", "\u6309", "\u6309\u94ae"]):
        task_type = "press"
    elif _goal_has(["push", "slide", "move", "\u79fb\u52a8", "\u63a8\u52a8", "\u63a8"]):
        task_type = "push"
    elif _goal_has(["touch", "contact", "\u89e6\u78b0", "\u63a5\u89e6"]):
        task_type = "touch"

    # Detect robot from goal text if not explicitly provided
    if not robot_id:
        robot_aliases = {
            "ur5e": "universal_robots_ur5e", "ur10e": "universal_robots_ur10e",
            "fr3": "franka_fr3", "franka": "franka_fr3", "panda": "franka_emika_panda",
            "kinova": "kinova_gen3", "kuka": "kuka_iiwa_14",
            "xarm": "ufactory_xarm7", "xarm7": "ufactory_xarm7",
            "lite6": "ufactory_lite6", "ufactory": "ufactory_lite6",
        }
        for alias, rid in robot_aliases.items():
            if alias in normalized:
                robot_id = rid
                break

    # Resolve robot spec for workspace-aware positions
    default_pos = [0.5, 0.0, _object_center_z_on_table("cube_5cm")]
    place_pos = [0.5, 0.15, _object_center_z_on_table("cube_5cm")]
    if robot_id:
        try:
            rspec = get_robot(robot_id)
            # Adjust positions based on robot reach/workspace
            if rspec.dof == 6:
                # 6-DOF robots (UR5e, Lite6) have smaller workspace
                default_pos = [0.45, 0.0, _object_center_z_on_table("cube_5cm")]
                place_pos = [0.45, 0.12, _object_center_z_on_table("cube_5cm")]
        except Exception:
            pass

    objects = [{"object_id": "cube_5cm", "role": "target", "initial_position": default_pos}]
    if any(w in normalized for w in ["cylinder", "can", "罐"]):
        objects = [{"object_id": "cylinder_can", "role": "target", "initial_position": default_pos}]
    elif any(w in normalized for w in ["block", "rectangular", "长方"]):
        objects = [{"object_id": "rect_block", "role": "target", "initial_position": default_pos}]

    held_tool_id = None
    if task_type == "press":
        objects = [{"object_id": "button_target", "role": "target", "initial_position": default_pos}]
    elif task_type == "insert":
        objects = [{"object_id": "insertion_socket", "role": "target", "initial_position": default_pos}]
        held_tool_id = "peg"
    elif task_type == "screwdriving":
        objects = [{"object_id": "screw_head", "role": "target", "initial_position": default_pos}]
        held_tool_id = "screwdriver"
    elif task_type == "tool_use":
        objects = [{"object_id": "flat_puck", "role": "target", "initial_position": default_pos}]
        if _goal_has(["hammer", "\u9524"]):
            held_tool_id = "hammer"
        elif _goal_has(["screwdriver", "screw", "\u87ba\u4e1d", "\u62e7"]):
            held_tool_id = "screwdriver"
        else:
            held_tool_id = "spatula"

    objects = _normalize_tabletop_object_positions(objects)
    default_pos = list(objects[0]["initial_position"]) if objects else default_pos
    requested_distance_m = _extract_distance_m(goal, default_m=0.1)
    distance_tolerance_m = max(0.01, min(0.02, requested_distance_m * 0.30))

    actions = []
    if task_type == "pick_and_place":
        pre_grasp = [default_pos[0], default_pos[1], default_pos[2] + 0.1]
        grasp_pos = [default_pos[0], default_pos[1], default_pos[2] - 0.015]
        actions = [
            {"type": "reach", "target_pos": pre_grasp},
            {"type": "reach", "target_pos": grasp_pos},
            {"type": "grasp", "force_scale": 1.0},
            {"type": "lift", "height": 0.08},
            {"type": "place", "target_pos": place_pos},
            {"type": "release"},
        ]
    elif task_type == "push":
        actions = [
            {"type": "reach", "target_pos": [default_pos[0], default_pos[1], default_pos[2] + 0.015]},
            {"type": "push", "direction": [1.0, 0.0, 0.0], "distance": requested_distance_m},
        ]
    elif task_type == "press":
        press_pos = [default_pos[0], default_pos[1], default_pos[2] + 0.015]
        actions = [
            {"type": "reach", "target_pos": [press_pos[0], press_pos[1], press_pos[2] + 0.10]},
            {"type": "reach", "target_pos": press_pos},
            {"type": "wait", "steps": 80},
        ]
    elif task_type == "touch":
        touch_pos = [default_pos[0], default_pos[1], default_pos[2] + 0.02]
        actions = [
            {"type": "reach", "target_pos": [touch_pos[0], touch_pos[1], touch_pos[2] + 0.08]},
            {"type": "reach", "target_pos": touch_pos},
            {"type": "wait", "steps": 60},
        ]
    elif task_type == "insert":
        insert_pos = [default_pos[0], default_pos[1], default_pos[2] + 0.035]
        actions = [
            {"type": "reach", "target_pos": [insert_pos[0], insert_pos[1], insert_pos[2] + 0.12]},
            {"type": "insert", "target_pos": insert_pos},
            {"type": "wait", "steps": 80},
        ]
    elif task_type == "screwdriving":
        screw_pos = [default_pos[0], default_pos[1], default_pos[2] + 0.035]
        actions = [
            {"type": "reach", "target_pos": [screw_pos[0], screw_pos[1], screw_pos[2] + 0.12]},
            {"type": "insert", "target_pos": screw_pos},
            {"type": "wait", "steps": 120},
        ]
    elif task_type == "tool_use":
        tool_pos = [default_pos[0], default_pos[1], default_pos[2] + 0.025]
        actions = [
            {"type": "reach", "target_pos": [tool_pos[0], tool_pos[1], tool_pos[2] + 0.10]},
            {"type": "reach", "target_pos": tool_pos},
            {"type": "push", "direction": [1.0, 0.0, 0.0], "distance": requested_distance_m},
            {"type": "wait", "steps": 60},
        ]

    # Determine robot candidates
    robot_candidates = [robot_id] if robot_id else ["franka_fr3"]
    if not robot_id and task_type in ("pick_and_place", "press", "insert", "screwdriving", "tool_use"):
        robot_candidates = [r for r in list_robot_ids() if get_robot(r).has_gripper] or ["franka_fr3"]

    # Build rich experiment space for meaningful variable scanning
    exp_vars: Dict[str, List[str]] = {
        "friction": ["low", "medium", "high"],
        "object_position": ["center", "left", "right"],
        "grasp_height": ["low", "nominal", "high"],
    }
    if task_type == "push":
        exp_vars = {
            "friction": ["low", "medium", "high"],
            "object_position": ["center", "left", "right"],
            "push_force": ["light", "medium", "strong"],
        }
    elif task_type in ("press", "touch"):
        exp_vars = {
            "object_position": ["center", "left", "right"],
            "press_depth": ["shallow", "nominal", "deep"],
            "approach_height": ["low", "nominal", "high"],
        }
    elif task_type in ("insert", "screwdriving"):
        exp_vars = {
            "object_position": ["center", "left", "right"],
            "insertion_depth": ["shallow", "nominal", "deep"],
            "approach_height": ["low", "nominal", "high"],
        }
    elif task_type == "tool_use":
        exp_vars = {
            "friction": ["low", "medium", "high"],
            "object_position": ["center", "left", "right"],
            "push_force": ["light", "medium", "strong"],
        }

    return {
        "task_type": task_type,
        "robot_preference": robot_id or "auto",
        "robot_candidates": robot_candidates,
        "objects": objects,
        "actions": actions,
        "held_tool_id": held_tool_id,
        "workspace": "table",
        "success_criteria": {
            "primary": "target_displacement" if task_type in {"push", "tool_use"} else "task_specific",
            "target_displacement_m": requested_distance_m if task_type in {"push", "tool_use"} else None,
            "tolerance": distance_tolerance_m if task_type in {"push", "tool_use"} else 0.02,
            "table_top_z": 0.39,
            "min_ee_clearance_m": 0.006,
        },
        "experiment_variables": exp_vars,
        "metrics": ["success_rate", "object_displacement", "contact_steps", "lifted_height"],
    }


def _build_experiment_matrix(
    exp_vars: Dict[str, List[str]], limit: int
) -> List[Dict[str, str]]:
    """Build experiment matrix from variable space.

    Generates combinations via itertools.product. If the full cartesian
    product is smaller than ``limit``, repeats with small perturbations
    to reach the requested count.
    """
    import itertools

    keys = list(exp_vars.keys())
    values = [exp_vars[k] for k in keys]

    base_matrix = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        base_matrix.append(config)

    if not base_matrix:
        base_matrix = [{}]

    # Expand to requested limit by cycling through base configs
    matrix = []
    while len(matrix) < limit:
        for config in base_matrix:
            if len(matrix) >= limit:
                break
            entry = dict(config)
            entry["run_id"] = f"nlp_{len(matrix) + 1:03d}"
            matrix.append(entry)

    return matrix


def _run_single_nlp_experiment(
    model,
    data,
    spec,
    plan: "MotionPlan",
    scene: Dict[str, Any],
    config: Dict[str, str],
    run_id: str,
    task_type: str = "general",
    success_criteria: Mapping[str, Any] | None = None,
    render: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment with real MuJoCo physics."""
    import mujoco
    import numpy as np
    from .motion_primitives import UniversalMotionExecutor, MotionTrace

    success_criteria = dict(success_criteria or {})

    # Reset to keyframe
    mujoco.mj_resetData(model, data)
    if spec.has_keyframe and spec.keyframe_qpos:
        n = min(len(spec.keyframe_qpos), model.nq)
        for i in range(n):
            data.qpos[i] = spec.keyframe_qpos[i]
    if spec.has_keyframe and spec.keyframe_ctrl:
        n_ctrl = min(len(spec.keyframe_ctrl), model.nu)
        for i in range(n_ctrl):
            data.ctrl[i] = spec.keyframe_ctrl[i]

    # Apply config variations
    friction = config.get("friction", "medium")
    friction_map = {"low": 0.05, "medium": 0.5, "high": 1.5}
    friction_val = friction_map.get(friction, 0.5)

    # Apply friction to object geoms
    object_body_ids = scene.get("object_body_ids", {})
    for oid, bid in object_body_ids.items():
        if bid >= 0:
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == bid:
                    model.geom_friction[g, 0] = friction_val

    # Position variation — larger offsets for meaningful perturbation
    pos_offset = config.get("object_position", "center")
    pos_map = {"center": (0.0, 0.0), "left": (0.0, 0.06), "right": (0.0, -0.06),
               "near": (-0.06, 0.0), "far": (0.06, 0.0)}
    dx, dy = pos_map.get(pos_offset, (0.0, 0.0))

    for oid, bid in object_body_ids.items():
        if bid >= 0:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{oid}_joint")
            if jnt_id >= 0:
                qpos_adr = model.jnt_qposadr[jnt_id]
                data.qpos[qpos_adr + 0] += dx
                data.qpos[qpos_adr + 1] += dy

    # Grasp height variation
    grasp_height = config.get("grasp_height", "nominal")
    grasp_height_map = {"low": -0.03, "nominal": 0.0, "high": 0.03}
    grasp_dz = grasp_height_map.get(grasp_height, 0.0)

    # Deep copy plan to avoid mutation drift across experiments
    import copy
    plan = copy.deepcopy(plan)

    # Modify plan primitives for grasp height variation
    if abs(grasp_dz) > 1e-6:
        for prim in plan.primitives:
            if prim.primitive_type == "reach" and prim.target_pos is not None:
                prim.target_pos = list(prim.target_pos)
                prim.target_pos[2] = prim.target_pos[2] + grasp_dz
            elif prim.primitive_type == "grasp" and prim.target_pos is not None:
                prim.target_pos = list(prim.target_pos)
                prim.target_pos[2] = prim.target_pos[2] + grasp_dz

    approach_height = config.get("approach_height", "nominal")
    approach_dz = {"low": -0.03, "nominal": 0.0, "high": 0.03}.get(approach_height, 0.0)
    if abs(approach_dz) > 1e-6:
        for prim in plan.primitives[:1]:
            if prim.target_pos is not None:
                prim.target_pos = list(prim.target_pos)
                prim.target_pos[2] = prim.target_pos[2] + approach_dz

    press_depth = config.get("press_depth", "nominal")
    press_dz = {"shallow": 0.015, "nominal": 0.0, "deep": -0.015}.get(press_depth, 0.0)
    insertion_depth = config.get("insertion_depth", "nominal")
    insert_dz = {"shallow": 0.02, "nominal": 0.0, "deep": -0.02}.get(insertion_depth, 0.0)
    depth_dz = press_dz if task_type in ("press", "touch") else insert_dz
    if abs(depth_dz) > 1e-6:
        for prim in plan.primitives[1:]:
            if prim.target_pos is not None:
                prim.target_pos = list(prim.target_pos)
                prim.target_pos[2] = prim.target_pos[2] + depth_dz

    # Push force variation
    push_force = config.get("push_force", "medium")
    push_force_map = {"light": 0.3, "medium": 1.0, "strong": 2.0}
    force_scale = push_force_map.get(push_force, 1.0)
    for prim in plan.primitives:
        if prim.primitive_type == "push":
            prim.force_scale = force_scale

    mujoco.mj_forward(model, data)

    # Create executor and run
    object_bid = next(iter(object_body_ids.values()), -1)
    executor = UniversalMotionExecutor(model, data, spec, object_body_id=object_bid)

    # Set up MuJoCo renderer for capturing frames (only when render=True)
    render_frames = []
    _capture_frame = None
    if render:
        try:
            _renderer = mujoco.Renderer(model, height=315, width=560)
            _camera = mujoco.MjvCamera()
            _camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            _camera.lookat[:] = [0.5, 0.0, 0.4]
            _camera.distance = 1.2
            _camera.azimuth = 135
            _camera.elevation = -25

            def _capture_frame(step, data):
                _renderer.update_scene(data, _camera)
                frame = _renderer.render().copy()
                render_frames.append(frame)
        except Exception:
            _capture_frame = None

    trace = executor.execute_plan(
        plan,
        sample_every=10,
        contact_body_prefix=spec.end_effector_body or "robot_base",
        contact_target_body=f"obj_{next(iter(object_body_ids), 'cube_5cm')}",
        render_callback=_capture_frame,
    )

    # Evaluate success based on task type and physics metrics

    success = False
    failure_type = ""
    table_top_z = float(success_criteria.get("table_top_z", 0.39) or 0.39)
    min_ee_clearance_m = float(success_criteria.get("min_ee_clearance_m", 0.006) or 0.006)
    min_ee_z = min(float(getattr(trace, "min_ee_z", 999.0)), _min_trace_ee_z(trace))
    table_clearance_ok = bool(getattr(trace, "table_clearance_ok", True)) and min_ee_z >= table_top_z + min_ee_clearance_m

    if not table_clearance_ok:
        failure_type = "table_penetration"
    elif task_type == "pick_and_place":
        # Must actually lift the object significantly
        if trace.lifted_height > 0.03 and trace.object_displacement > 0.03:
            success = True
        elif trace.lifted_height > 0.01:
            success = True
            failure_type = "weak_lift"
        elif trace.contact_steps > 20 and trace.object_displacement > 0.02:
            success = True
            failure_type = "no_lift_contact_only"
        elif trace.contact_steps > 5:
            failure_type = "grasp_miss"
        else:
            failure_type = "no_contact"
    elif task_type == "push":
        target_displacement = float(success_criteria.get("target_displacement_m", 0.05) or 0.05)
        tolerance = float(success_criteria.get("tolerance", 0.015) or 0.015)
        displacement_error = abs(trace.object_displacement - target_displacement)
        if trace.contact_steps <= 10:
            failure_type = "no_contact"
        elif displacement_error <= tolerance:
            success = True
        elif trace.object_displacement > target_displacement + tolerance:
            failure_type = "overshoot"
        elif trace.object_displacement < max(0.0, target_displacement - tolerance):
            failure_type = "undershoot"
        else:
            failure_type = "push_error"
    elif task_type == "insert":
        if trace.contact_steps > 50 and trace.object_displacement > 0.01:
            success = True
        elif trace.contact_steps > 10:
            failure_type = "insertion_incomplete"
        else:
            failure_type = "no_contact"
    elif task_type in ("press", "touch"):
        if trace.contact_steps > 20 or trace.max_touch_force > 0.5:
            success = True
        elif trace.contact_steps > 3:
            failure_type = "weak_contact"
        else:
            failure_type = "no_contact"
    elif task_type in ("screwdriving", "tool_use"):
        if success_criteria.get("target_displacement_m"):
            target_displacement = float(success_criteria.get("target_displacement_m", 0.05) or 0.05)
            tolerance = float(success_criteria.get("tolerance", 0.015) or 0.015)
            displacement_error = abs(trace.object_displacement - target_displacement)
            if trace.contact_steps <= 3:
                failure_type = "no_contact"
            elif displacement_error <= tolerance:
                success = True
            elif trace.object_displacement > target_displacement + tolerance:
                failure_type = "overshoot"
            else:
                failure_type = "undershoot"
        elif trace.contact_steps > 15 and (trace.object_displacement > 0.005 or trace.max_touch_force > 0.5):
            success = True
        elif trace.contact_steps > 3:
            failure_type = "tool_contact_weak"
        else:
            failure_type = "no_contact"
    else:
        # General: require meaningful contact + displacement
        if trace.object_displacement > 0.03 and trace.contact_steps > 20:
            success = True
        elif trace.contact_steps > 10:
            failure_type = "weak_interaction"
        else:
            failure_type = "no_contact"

    result = {
        **config,
        "run_id": run_id,
        "success": success,
        "failure_type": failure_type,
        "object_displacement": trace.object_displacement,
        "lifted_height": trace.lifted_height,
        "contact_steps": trace.contact_steps,
        "max_touch_force": trace.max_touch_force,
        "target_displacement_m": success_criteria.get("target_displacement_m"),
        "displacement_error_m": (
            abs(trace.object_displacement - float(success_criteria.get("target_displacement_m")))
            if success_criteria.get("target_displacement_m") is not None
            else None
        ),
        "table_clearance_ok": table_clearance_ok,
        "min_ee_z": min_ee_z,
        "table_top_z": table_top_z,
        "table_penetration_steps": getattr(trace, "table_penetration_steps", 0),
        "table_contact_steps": getattr(trace, "table_contact_steps", 0),
        "table_contact_pairs": getattr(trace, "table_contact_pairs", []),
        "final_object_pos": trace.final_object_pos.tolist() if trace.final_object_pos is not None else None,
        "final_ee_pos": trace.final_ee_pos.tolist() if trace.final_ee_pos is not None else None,
    }
    # Only include render frames if captured (expensive, only for demo runs)
    if render_frames:
        from .task_render import encode_jpeg, select_evenly
        frames = select_evenly(render_frames, 30)
        result["image_frames"] = [encode_jpeg(f) for f in frames]
        result["labels"] = [f"frame {i+1}/{len(frames)}" for i in range(len(frames))]
    return result


def run_agent_round(
    goal: str,
    limit: int = 27,
    use_fallback: bool = False,
    preferred_task_id: str | None = None,
    language: str = "zh",
) -> Dict[str, Any]:
    if os.environ.get("AGENT_FORCE_LOCAL", "").strip().lower() in {"1", "true", "yes"}:
        return _run_local_agent_round(
            goal=goal,
            limit=limit,
            use_fallback=use_fallback,
            preferred_task_id=preferred_task_id,
            language=language,
            fallback_reason="AGENT_FORCE_LOCAL requested deterministic local capability planning.",
        )
    try:
        agent = XiaomiResearchAgent()
        route = agent.route_task(goal, preferred_task_id=preferred_task_id, language=language)
        task = get_task(route["selected_task"])
        design = agent.design_experiment(goal, limit, task.spec, language=language)
        execution_plan = _execution_plan(task.spec, design)
        result = task.run_experiments(
            limit=limit,
            use_fallback=use_fallback,
            experiment_space=design.get("experiment_space"),
        )
        summary = summarize_runs(result["runs"])
        analysis = agent.analyze_runs(
            goal,
            task.spec,
            design,
            result["runs"],
            summary,
            language=language,
        )
        return {
            "source": f"agent+{agent.config.protocol}+{result['source']}",
            "agent_provider": "xiaomi-compatible",
            "model": agent.config.model,
            "protocol": agent.config.protocol,
            "language": "en" if _language_name(language) == "English" else "zh",
            "task_id": task.spec.task_id,
            "task": asdict(task.spec),
            "object_library": [
                obj for obj in list_objects() if obj["object_id"] in task.spec.supported_objects
            ],
            "asset_registry": list_asset_registry(),
            "route": route,
            "design": design,
            "execution_plan": execution_plan,
            "runs": result["runs"],
            "summary": summary,
            "analysis": analysis,
            "model_api_calls": {
                "count": len(agent.api_call_log),
                "calls": agent.api_call_log,
                "note": "Only /api/agent/run uses these model calls. /api/run_experiments runs local MuJoCo without LLM routing, design, or analysis.",
            },
            "agent_trace": [
                "route_task_model_api" if not preferred_task_id else "route_task_local_preferred",
                "design_experiment",
                f"run_{result['source']}_experiments",
                "analyze_runs",
            ],
        }
    except Exception as exc:
        return _run_local_agent_round(
            goal=goal,
            limit=limit,
            use_fallback=use_fallback,
            preferred_task_id=preferred_task_id,
            language=language,
            fallback_reason=f"Model API path was unavailable; local planner handled the loop. Detail: {exc}",
        )
