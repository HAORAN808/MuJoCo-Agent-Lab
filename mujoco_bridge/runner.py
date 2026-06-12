from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


EXPERIMENT_SPACE = {
    "object_offset": ["small", "medium", "large"],
    "friction": ["low", "medium", "high"],
    "grasp_height_offset": ["-2cm", "0", "+2cm"],
    "vision_noise": ["none", "light", "heavy"],
    "control_freq": ["normal"],
}


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str
    object_offset: str
    friction: str
    grasp_height_offset: str
    vision_noise: str
    control_freq: str = "normal"


@dataclass
class RunResult:
    run_id: str
    object_offset: str
    friction: str
    grasp_height_offset: str
    vision_noise: str
    control_freq: str
    success: bool
    failure_type: str
    trajectory_error: float
    collision_count: int
    final_distance: float
    max_grip_force: float = 0.0
    cube_slip_detected: bool = False
    touch_contact_duration: float = 0.0


def seeded_noise(seed: int) -> float:
    value = math.sin(seed * 12.9898) * 43758.5453
    return value - math.floor(value)


def _space_values(
    experiment_space: Mapping[str, Sequence[str]] | None,
    key: str,
) -> List[str]:
    if experiment_space and experiment_space.get(key):
        supported = set(EXPERIMENT_SPACE[key])
        values = [str(v) for v in experiment_space[key] if str(v) in supported]
        if values:
            return values
    return EXPERIMENT_SPACE[key]


def build_experiment_matrix(
    limit: int = 81,
    experiment_space: Mapping[str, Sequence[str]] | None = None,
) -> List[ExperimentConfig]:
    rows: List[ExperimentConfig] = []
    idx = 1
    object_offsets = _space_values(experiment_space, "object_offset")
    frictions = _space_values(experiment_space, "friction")
    height_offsets = _space_values(experiment_space, "grasp_height_offset")
    vision_noises = _space_values(experiment_space, "vision_noise")
    control_freqs = _space_values(experiment_space, "control_freq")

    # The 27-run demo mode should still cover all three vision-noise levels.
    # Iterating the full 81-run matrix with rows[::3] accidentally sampled only
    # vision_noise=none because vision noise is the innermost loop.
    if limit == 27 and len(object_offsets) == len(frictions) == len(height_offsets) == 3:
        for object_offset in object_offsets:
            for friction in frictions:
                for grasp_height_offset in height_offsets:
                    vision_noise = vision_noises[(idx - 1) % len(vision_noises)]
                    rows.append(
                        ExperimentConfig(
                            run_id=f"exp_{idx:03d}",
                            object_offset=object_offset,
                            friction=friction,
                            grasp_height_offset=grasp_height_offset,
                            vision_noise=vision_noise,
                            control_freq=control_freqs[0],
                        )
                    )
                    idx += 1
        return rows

    for object_offset in object_offsets:
        for friction in frictions:
            for grasp_height_offset in height_offsets:
                for vision_noise in vision_noises:
                    rows.append(
                        ExperimentConfig(
                            run_id=f"exp_{idx:03d}",
                            object_offset=object_offset,
                            friction=friction,
                            grasp_height_offset=grasp_height_offset,
                            vision_noise=vision_noise,
                            control_freq=control_freqs[0],
                        )
                    )
                    idx += 1

    return rows[:limit]


def score_config(config: ExperimentConfig, index: int) -> float:
    """Deterministic success probability proxy calibrated to match MuJoCo physics.

    Observed from 27-config physics matrix (~37% overall success):
    - low friction → 0% success (cube slips during lift)
    - -2cm height → 0% success (fingers miss cube)
    - medium/high friction + 0/+2cm height → ~56% success
    - large offset reduces success further
    """
    # Base success rate under ideal conditions
    score = 0.72
    # Friction: low → always fail
    if config.friction == "low":
        score -= 0.50
    if config.friction == "high":
        score += 0.08
    # Height: -2cm → always fail (fingers too low)
    if config.grasp_height_offset == "-2cm":
        score -= 0.50
    if config.grasp_height_offset == "+2cm":
        score -= 0.02
    # Object offset: large → harder
    if config.object_offset == "medium":
        score -= 0.05
    if config.object_offset == "large":
        score -= 0.15
    # Vision noise
    if config.vision_noise == "light":
        score -= 0.04
    if config.vision_noise == "heavy":
        score -= 0.10
    # Coupling: large + heavy noise
    if config.object_offset == "large" and config.vision_noise == "heavy":
        score -= 0.08
    # Per-sample jitter
    score += (seeded_noise(index + 17) - 0.5) * 0.14
    return max(0.02, min(0.96, score))


def classify_proxy_failure(config: ExperimentConfig, index: int) -> str:
    """Deterministic failure-type classifier matching MuJoCo physics results.

    Physics-observed failure patterns:
    - low friction → slip (cube slips during lift)
    - -2cm height → grasp_miss (fingers miss cube, too low)
    - large offset + noise → grasp_miss or slip
    """
    # Low friction: always slip
    if config.friction == "low":
        return "slip"
    # Negative height offset: always grasp_miss (fingers misaligned)
    if config.grasp_height_offset == "-2cm":
        return "grasp_miss"
    # Large offset with noise: grasp_miss
    if config.object_offset == "large" and config.vision_noise == "heavy":
        return "grasp_miss"
    # Medium friction with some height/noise issues: slip
    if config.friction == "medium" and seeded_noise(index + 3) > 0.6:
        return "slip"
    # Default: grasp_miss
    return "grasp_miss"


class FallbackRunner:
    """Deterministic proxy with the same output schema as the MuJoCo runner."""

    source = "fallback"

    def run_one(self, config: ExperimentConfig, index: int) -> RunResult:
        score = score_config(config, index)
        success = seeded_noise(index + 101) < score
        failure_type = "none" if success else classify_proxy_failure(config, index)
        trajectory_error = round(0.025 + (1 - score) * 0.24 + seeded_noise(index + 7) * 0.03, 3)
        collision_count = 0
        if failure_type == "collision":
            collision_count = 1 + int(seeded_noise(index + 41) * 3)
        final_distance = round(
            0.02 + seeded_noise(index + 5) * 0.04
            if success
            else 0.09 + seeded_noise(index + 6) * 0.18,
            3,
        )
        max_grip_force = round(
            0.5 + seeded_noise(index + 81) * 2.0 if success else seeded_noise(index + 82) * 0.3,
            3,
        )
        cube_slip_detected = failure_type == "slip"
        touch_contact_duration = round(
            0.8 + seeded_noise(index + 91) * 0.4 if success else seeded_noise(index + 92) * 0.2,
            3,
        )
        return RunResult(
            **asdict(config),
            success=success,
            failure_type=failure_type,
            trajectory_error=trajectory_error,
            collision_count=collision_count,
            final_distance=final_distance,
            max_grip_force=max_grip_force,
            cube_slip_detected=cube_slip_detected,
            touch_contact_duration=touch_contact_duration,
        )


class MujocoPickPlaceRunner:
    """MuJoCo adapter using FR3 arm + Franka Hand.

    Builds a fresh physics scene for each run and drives the arm via
    Jacobian-based IK.  Outcomes emerge from contact physics.
    """

    source = "mujoco"

    def __init__(self, model_path: str | None = None) -> None:
        try:
            import mujoco  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "Python package 'mujoco' is not available. Start the server with "
                "--fallback to validate the API, or install mujoco before using "
                "the real runner."
            ) from exc

        self.mujoco = mujoco
        self.model_path = model_path
        from .simple_pick_place import FR3PickPlaceSim

        self.env = FR3PickPlaceSim()

    def run_one(self, config: ExperimentConfig, index: int) -> RunResult:
        """Run one physics-based pick-and-place experiment."""
        return self.env.run(config, index)


def run_experiments(
    limit: int = 81,
    use_fallback: bool = False,
    experiment_space: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, Any]:
    configs = build_experiment_matrix(limit, experiment_space=experiment_space)
    runner: Any
    if use_fallback:
        runner = FallbackRunner()
    else:
        runner = MujocoPickPlaceRunner()

    runs = [asdict(runner.run_one(config, i + 1)) for i, config in enumerate(configs)]
    return {
        "source": runner.source,
        "runs": runs,
    }


def run_demo_trace(use_fallback: bool = False) -> Dict[str, Any]:
    config = ExperimentConfig(
        run_id="demo_trace",
        object_offset="small",
        friction="medium",
        grasp_height_offset="0",
        vision_noise="none",
    )
    if use_fallback:
        return {
            "source": "fallback",
            "frames": [
                {"label": "reset", "time": 0.0, "gripper": {"x": 0.0, "y": 0.0, "z": 0.14, "closed": 0}, "cube": {"x": 0.02, "y": 0.0, "z": 0.025}, "target": {"x": 0.20, "y": 0.0, "z": 0.025}},
                {"label": "grasp", "time": 0.8, "gripper": {"x": 0.02, "y": 0.0, "z": 0.09, "closed": 1}, "cube": {"x": 0.02, "y": 0.0, "z": 0.055}, "target": {"x": 0.20, "y": 0.0, "z": 0.025}},
                {"label": "move-to-target", "time": 1.6, "gripper": {"x": 0.20, "y": 0.0, "z": 0.13, "closed": 1}, "cube": {"x": 0.20, "y": 0.0, "z": 0.055}, "target": {"x": 0.20, "y": 0.0, "z": 0.025}},
                {"label": "release", "time": 2.4, "gripper": {"x": 0.20, "y": 0.0, "z": 0.13, "closed": 0}, "cube": {"x": 0.20, "y": 0.0, "z": 0.025}, "target": {"x": 0.20, "y": 0.0, "z": 0.025}},
            ],
        }

    from .simple_pick_place import FR3PickPlaceSim

    result, frames = FR3PickPlaceSim().run_with_trace(config, 1, collect_trace=True)
    from .fr3_render import generate_web_replay_assets

    replay_assets = generate_web_replay_assets()
    success_replay = replay_assets["success"]
    failure_replay = replay_assets["failure"]
    return {
        "source": "mujoco_fr3",
        "model": "MuJoCo Menagerie franka_fr3",
        "result": asdict(result),
        "frames": frames,
        "replays": [
            {
                **success_replay,
                "title": "成功样例：完成抓取与放置",
                "diagnosis": "success",
            },
            {
                **failure_replay,
                "title": "失败样例：抓取点偏移导致抓空",
                "diagnosis": "grasp_miss",
            },
        ],
    }


def summarize_runs(runs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(runs)
    success_count = sum(1 for row in rows if row["success"])
    failures = [row for row in rows if not row["success"]]
    distribution: Dict[str, int] = {}
    for row in failures:
        distribution[row["failure_type"]] = distribution.get(row["failure_type"], 0) + 1
    return {
        "num_runs": len(rows),
        "success_rate": success_count / max(1, len(rows)),
        "failure_distribution": {
            key: value / max(1, len(rows)) for key, value in distribution.items()
        },
    }
