$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (!(Test-Path $VenvPython)) {
  throw "Virtual environment not found. Run scripts\setup_mujoco_env.ps1 first."
}

& $VenvPython -m mujoco_bridge.verify_strict_constraints
