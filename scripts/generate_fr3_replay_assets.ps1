$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (!(Test-Path $Python)) {
  throw "Virtual environment not found. Run scripts\setup_mujoco_env.ps1 first."
}

Push-Location $Root
try {
  & $Python -m mujoco_bridge.fr3_render
} finally {
  Pop-Location
}
