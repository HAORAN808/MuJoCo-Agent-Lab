"""Render FR3 + Franka Hand replay frames for the web demo.

Replay assets are sampled from actual MuJoCo simulation states.  The renderer
does not attach or teleport the cube for display.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def _encode_jpeg(frame: np.ndarray) -> str:
    image = Image.fromarray(frame)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _select_evenly(items: List[Tuple[np.ndarray, str]], count: int) -> List[Tuple[np.ndarray, str]]:
    if len(items) <= count:
        return items
    if count <= 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def render_physics_replay_frames(
    width: int = 560,
    height: int = 315,
    frame_count: int = 40,
    scenario: str = "success",
    output_dir: Path | None = None,
    web_prefix: str | None = None,
) -> Dict[str, object]:
    """Render replay frames from the actual MuJoCo simulation state.

    The renderer does not attach the cube to the hand in display code. The cube
    only moves if the physics simulation moves it through contact.
    """
    import mujoco  # type: ignore

    from .runner import ExperimentConfig
    from .simple_pick_place import FR3PickPlaceSim

    if scenario == "failure_grasp_miss":
        config = ExperimentConfig(
            run_id="replay_failure",
            object_offset="large",
            friction="medium",
            grasp_height_offset="-2cm",
            vision_noise="heavy",
        )
        index = 2
    else:
        config = ExperimentConfig(
            run_id="replay_success",
            object_offset="small",
            friction="medium",
            grasp_height_offset="0",
            vision_noise="none",
        )
        index = 1

    renderer = None
    captured: List[Tuple[np.ndarray, str]] = []
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.70, 0.0, 0.55)
    camera.distance = 0.80
    camera.azimuth = 70
    camera.elevation = -15

    def capture(model, data, label: str) -> None:
        nonlocal renderer
        if renderer is None:
            renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera=camera)
        captured.append((renderer.render().copy(), label))

    env = FR3PickPlaceSim()
    result, _trace = env.run_with_trace(
        config,
        index,
        collect_trace=False,
        render_callback=capture,
        render_interval=35,
    )
    if renderer is not None:
        renderer.close()

    selected = _select_evenly(captured, frame_count)
    images: List[str] = []
    labels: List[str] = []

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in output_dir.glob("frame_*.jpg"):
            old_frame.unlink()

    for frame_idx, (frame, label) in enumerate(selected):
        labels.append(label)
        if output_dir is not None:
            frame_name = f"frame_{frame_idx:03d}.jpg"
            Image.fromarray(frame).save(output_dir / frame_name, format="JPEG", quality=78, optimize=True)
            images.append(
                f"{web_prefix.rstrip('/')}/{frame_name}" if web_prefix else str(output_dir / frame_name)
            )
        else:
            images.append(_encode_jpeg(frame))

    return {
        "source": "mujoco_fr3",
        "model": "MuJoCo Menagerie franka_fr3 + Franka Hand",
        "end_effector": "Franka Hand (contact-physics replay)",
        "scenario": scenario,
        "result": {
            "success": result.success,
            "failure_type": result.failure_type,
            "final_distance": result.final_distance,
            "max_grip_force": result.max_grip_force,
        },
        "image_frames": images,
        "labels": labels,
        "width": width,
        "height": height,
    }


def generate_web_replay_assets() -> Dict[str, object]:
    web_assets = ROOT.parent / "web_demo" / "assets"
    success_dir = web_assets / "fr3_success"
    failure_dir = web_assets / "fr3_failure"
    success = render_physics_replay_frames(
        scenario="success",
        output_dir=success_dir,
        web_prefix="assets/fr3_success",
    )
    failure = render_physics_replay_frames(
        scenario="failure_grasp_miss",
        output_dir=failure_dir,
        web_prefix="assets/fr3_failure",
    )
    return {"success": success, "failure": failure}


if __name__ == "__main__":
    result = generate_web_replay_assets()
    print(
        {
            "success_frames": len(result["success"]["image_frames"]),
            "failure_frames": len(result["failure"]["image_frames"]),
        }
    )
