from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def encode_jpeg(frame: np.ndarray) -> str:
    image = Image.fromarray(frame)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=76, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def select_evenly(items: List[Tuple[np.ndarray, str]], count: int) -> List[Tuple[np.ndarray, str]]:
    if len(items) <= count:
        return items
    return [items[round(i * (len(items) - 1) / max(1, count - 1))] for i in range(count)]


def save_or_encode_frames(
    frames: List[Tuple[np.ndarray, str]],
    frame_count: int,
    output_dir: Path | None,
    web_prefix: str | None,
) -> tuple[list[str], list[str]]:
    selected = select_evenly(frames, frame_count)
    images: list[str] = []
    labels: list[str] = []

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for old in output_dir.glob("frame_*.jpg"):
            old.unlink()

    for idx, (frame, label) in enumerate(selected):
        labels.append(label)
        if output_dir is None:
            images.append(encode_jpeg(frame))
        else:
            filename = f"frame_{idx:03d}.jpg"
            Image.fromarray(frame).save(output_dir / filename, format="JPEG", quality=80, optimize=True)
            images.append(f"{web_prefix.rstrip('/')}/{filename}" if web_prefix else str(output_dir / filename))

    return images, labels


def render_controlled_scene(
    model,
    data,
    step_fn: Callable[[int], str],
    steps: int,
    interval: int = 18,
    width: int = 560,
    height: int = 315,
    frame_count: int = 40,
    output_dir: Path | None = None,
    web_prefix: str | None = None,
) -> tuple[list[str], list[str]]:
    import mujoco  # type: ignore

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.08, 0.0, 0.08)
    camera.distance = 0.72
    camera.azimuth = 55
    camera.elevation = -24
    captured: List[Tuple[np.ndarray, str]] = []
    try:
        for step in range(steps):
            label = step_fn(step)
            mujoco.mj_step(model, data)
            if step % interval == 0:
                renderer.update_scene(data, camera=camera)
                captured.append((renderer.render().copy(), label))
    finally:
        renderer.close()
    return save_or_encode_frames(captured, frame_count, output_dir, web_prefix)

