from __future__ import annotations

import argparse
import json
import mimetypes
import re
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote

from .runner import run_demo_trace, run_experiments, summarize_runs
from .asset_library import list_asset_registry
from .object_library import list_objects
from .robot_registry import list_robots, list_robot_ids
from .tasks import get_task, list_task_specs
from .tasks.base import summarize_runs as summarize_task_runs
from .arm_runner import ArmSkillConfig, FR3ArmSkillRunner, list_arm_skill_specs

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
WEB_ASSETS_DIR = Path(__file__).resolve().parent.parent / "web_demo" / "assets"


def _parse_yaml_lists(text: str) -> Dict[str, Any]:
    """Minimal YAML parser for the experiment_space.yaml format.

    Handles nested dicts (up to 3 levels), scalar values, inline-list
    values like [a, b, c], and dash-style lists like ``- item``. Not a
    general-purpose YAML parser — only covers the subset used in this
    project's config files.
    """
    root: Dict[str, Any] = {}
    # Stack of (indent_level, dict) for tracking nesting
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    # Track (indent, key) for the most recent key at each indent level,
    # so dash-style list items know which key to attach to.
    last_key_at_indent: Dict[int, str] = {}

    for raw_line in text.splitlines():
        stripped = raw_line.split("#")[0].rstrip()
        if not stripped:
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = stripped.lstrip()

        # Dash-style list item: ``- value``
        dm = re.match(r"^-\s+(.*)", line)
        if dm:
            # Dash items at indent N belong to the key at indent N-2.
            # Pop until stack top indent < N-2 (the parent dict that owns the key).
            target_indent = indent - 2
            while len(stack) > 1 and stack[-1][0] >= target_indent:
                stack.pop()
            parent = stack[-1][1]
            key = last_key_at_indent.get(target_indent)
            if key and key in parent:
                if not isinstance(parent[key], list):
                    parent[key] = []
                parent[key].append(dm.group(1).strip().strip('"').strip("'"))
            continue

        m = re.match(r"^(\w[\w_]*):\s*(.*)", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()

        # Pop stack back to the parent dict for this indent level.
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]

        # Store this key at its indent level. Clear deeper entries —
        # they belonged to a previous section and are now stale.
        last_key_at_indent[indent] = key
        for d in list(last_key_at_indent):
            if d > indent:
                del last_key_at_indent[d]

        if value.startswith("[") and value.endswith("]"):
            items = [v.strip().strip('"').strip("'") for v in value[1:-1].split(",")]
            parent[key] = items
        elif value:
            parent[key] = value.strip('"').strip("'")
        else:
            # Empty value: create a child dict for potential nested content.
            # If dash items follow, they'll be converted to a list.
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _load_experiment_space() -> Dict[str, List[str]]:
    """Load experiment space variables from configs/experiment_space.yaml."""
    yaml_path = CONFIGS_DIR / "experiment_space.yaml"
    if not yaml_path.exists():
        return {}
    parsed = _parse_yaml_lists(yaml_path.read_text(encoding="utf-8"))
    variables = parsed.get("variables", {})
    # Flatten nested dicts: each variable key maps to its values list
    space: Dict[str, List[str]] = {}
    for var_name, var_def in variables.items():
        if isinstance(var_def, dict) and "values" in var_def:
            space[var_name] = var_def["values"]
    return space


def make_response(payload: Dict[str, Any], status: int = 200) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


class ExperimentHandler(BaseHTTPRequestHandler):
    use_fallback = False
    experiment_space: Dict[str, List[str]] = {}
    nlp_progress: Dict[str, Any] = {"step": "", "detail": "", "percent": 0}

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[mujoco-api]", fmt % args)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = make_response(payload, status)
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_asset(self) -> None:
        rel = unquote(self.path[len("/assets/") :]).replace("\\", "/")
        target = (WEB_ASSETS_DIR / rel).resolve()
        try:
            target.relative_to(WEB_ASSETS_DIR.resolve())
        except ValueError:
            self._send_json({"error": "invalid_asset_path"}, status=400)
            return
        if not target.is_file():
            self._send_json({"error": "asset_not_found"}, status=404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/assets/"):
            self._send_asset()
            return
        if self.path == "/health":
            self._send_json({"ok": True, "mode": "fallback" if self.use_fallback else "mujoco"})
            return
        if self.path == "/api/experiment_space":
            self._send_json({"variables": self.experiment_space})
            return
        if self.path == "/api/tasks":
            self._send_json({"tasks": list_task_specs()})
            return
        if self.path == "/api/object_library":
            self._send_json({"objects": list_objects()})
            return
        if self.path == "/api/asset_registry":
            self._send_json(list_asset_registry())
            return
        if self.path == "/api/arm/skills":
            self._send_json({"skills": list_arm_skill_specs()})
            return
        if self.path == "/api/nlp/status":
            self._send_json(self.nlp_progress)
            return
        if self.path == "/api/runner/status":
            self._send_json(
                {
                    "ok": True,
                    "task_runners": list_task_specs(),
                    "arm_skill_runners": list_arm_skill_specs(),
                    "verification": {
                        "all": "powershell -ExecutionPolicy Bypass -File .\\scripts\\verify_runners.ps1",
                        "tasks": "powershell -ExecutionPolicy Bypass -File .\\scripts\\run_smoke_all_tasks.ps1",
                        "arm_skills": "powershell -ExecutionPolicy Bypass -File .\\scripts\\run_arm_skill_smoke.ps1",
                        "agent_loop": "powershell -ExecutionPolicy Bypass -File .\\scripts\\verify_agent_loop.ps1",
                    },
                }
            )
            return
        if self.path == "/api/agent/status":
            try:
                from .agent import load_xiaomi_config

                config = load_xiaomi_config()
                self._send_json(
                    {
                        "configured": True,
                        "protocol": config.protocol,
                        "model": config.model,
                        "base_url": config.base_url,
                        "note": "This only checks local configuration. Actual model calls happen in POST /api/agent/run.",
                    }
                )
            except Exception as exc:
                self._send_json(
                    {
                        "configured": False,
                        "error": str(exc),
                        "note": "POST /api/run_experiments does not use a model API. Configure this before using /api/agent/run.",
                    }
                )
            return
        if self.path == "/api/robots":
            robots = []
            for r in list_robots():
                robots.append({
                    "robot_id": r.robot_id,
                    "name": r.name,
                    "dof": r.dof,
                    "has_gripper": r.has_gripper,
                    "gripper_type": r.gripper_type,
                    "gripper_joint_names": list(r.gripper_joint_names),
                    "gripper_actuator_names": list(r.gripper_actuator_names),
                    "actuator_type": r.actuator_type,
                    "manufacturer": r.manufacturer,
                    "end_effector_site": r.end_effector_site,
                    "end_effector_body": r.end_effector_body,
                    "end_effector_name": "Franka Hand" if r.robot_id == "franka_fr3" and r.has_gripper else r.gripper_type,
                })
            self._send_json({"robots": robots})
            return
        self._send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:
        if self.path == "/api/agent/run":
            self._handle_agent_run()
            return
        if self.path == "/api/nlp/run":
            self._handle_nlp_run()
            return
        if self.path == "/api/arm/run_skill":
            self._handle_arm_skill()
            return

        if self.path != "/api/run_experiments":
            self._send_json({"error": "not_found"}, status=404)
            return

        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(raw.decode("utf-8") or "{}")
            limit = int(payload.get("limit", 81))
            if limit not in (27, 81):
                limit = 81
            task_id = str(payload.get("task_id", "")).strip()
            if task_id:
                task = get_task(task_id)
                result = task.run_experiments(limit=limit, use_fallback=self.use_fallback)
                result["summary"] = summarize_task_runs(result["runs"])
                result["task"] = asdict(task.spec)
                result["demo_trace"] = task.demo_trace(use_fallback=self.use_fallback)
            else:
                result = run_experiments(limit=limit, use_fallback=self.use_fallback)
                result["summary"] = summarize_runs(result["runs"])
                result["demo_trace"] = run_demo_trace(use_fallback=self.use_fallback)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_agent_run(self) -> None:
        try:
            from .agent import run_agent_round

            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(raw.decode("utf-8") or "{}")
            goal = str(payload.get("goal", "")).strip() or (
                "研究机械臂在不同物体、接触条件和任务目标下的操作成功率、失败类型和下一轮实验改进方向"
            )
            limit = int(payload.get("limit", 27))
            if limit not in (27, 81):
                limit = 27
            preferred_task_id = str(payload.get("task_id", "")).strip() or None
            language = str(payload.get("language", "zh")).strip().lower()
            result = run_agent_round(
                goal=goal,
                limit=limit,
                use_fallback=self.use_fallback,
                preferred_task_id=preferred_task_id,
                language=language,
            )
            result["demo_trace"] = get_task(result["task_id"]).demo_trace(
                use_fallback=self.use_fallback
            )
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_nlp_run(self) -> None:
        try:
            from .agent import run_nlp_agent_round

            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(raw.decode("utf-8") or "{}")
            goal = str(payload.get("goal", "")).strip()
            if not goal:
                self._send_json({"error": "goal is required"}, status=400)
                return
            limit = int(payload.get("limit", 9))
            if limit < 1:
                limit = 1
            if limit > 81:
                limit = 81
            language = str(payload.get("language", "zh")).strip().lower()
            robot_id_override = str(payload.get("robot_id", "")).strip()

            def _update_progress(step: str, detail: str, percent: int):
                ExperimentHandler.nlp_progress = {"step": step, "detail": detail, "percent": percent}

            _update_progress("初始化", "正在启动 NLP 管线...", 0)
            result = run_nlp_agent_round(
                goal=goal,
                limit=limit,
                language=language,
                robot_id=robot_id_override,
                progress_callback=_update_progress,
            )
            _update_progress("完成", "NLP 管线已完成", 100)
            # Enrich summary with fields the frontend expects
            summary = result.get("summary", {})
            runs = result.get("runs", [])
            if summary and not summary.get("main_failure_type"):
                dist = summary.get("failure_distribution", {})
                if dist:
                    summary["main_failure_type"] = max(dist, key=dist.get)
                else:
                    summary["main_failure_type"] = "none"
            if summary and "explainable_rate" not in summary:
                failures = [r for r in runs if not r.get("success")]
                if failures:
                    explained = [r for r in failures if r.get("failure_type", "unknown") not in ("unknown", "timeout")]
                    summary["explainable_rate"] = len(explained) / len(failures)
                else:
                    summary["explainable_rate"] = 1.0
            result["summary"] = summary
            result["num_runs"] = summary.get("num_runs", len(runs))
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_arm_skill(self) -> None:
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(raw.decode("utf-8") or "{}")
            skill_id = str(payload.get("skill_id", "reach_touch")).strip()
            object_xy = payload.get("object_xy", [0.59, 0.0])
            if not isinstance(object_xy, (list, tuple)) or len(object_xy) != 2:
                object_xy = [0.59, 0.0]
            config = ArmSkillConfig(
                skill_id=skill_id,
                object_id=str(payload.get("object_id", "cube_5cm")).strip() or "cube_5cm",
                object_xy=(float(object_xy[0]), float(object_xy[1])),
                friction=float(payload.get("friction", 0.9)),
                grasp_height_delta=float(payload.get("grasp_height_delta", 0.0)),
                sweep_scale=float(payload.get("sweep_scale", 1.0)),
            )
            result = FR3ArmSkillRunner().run_skill(config)
            self._send_json({"result": asdict(result)})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo experiment API for the web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fallback", action="store_true", help="Use deterministic proxy runner.")
    args = parser.parse_args()

    ExperimentHandler.use_fallback = args.fallback
    ExperimentHandler.experiment_space = _load_experiment_space()
    server = ThreadingHTTPServer((args.host, args.port), ExperimentHandler)
    mode = "fallback" if args.fallback else "mujoco"
    print(f"MuJoCo experiment API running at http://{args.host}:{args.port} ({mode})")
    server.serve_forever()


if __name__ == "__main__":
    main()
