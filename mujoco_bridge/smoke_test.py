from __future__ import annotations

import json

import mujoco  # type: ignore

from .simple_pick_place import run_smoke


def main() -> None:
    result = run_smoke()
    print(json.dumps({
        "mujoco_version": mujoco.__version__,
        "smoke_result": result,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
