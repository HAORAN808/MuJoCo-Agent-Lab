param(
  [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (!(Test-Path $VenvPython)) {
  python -m venv (Join-Path $Root ".venv")
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "requirements-mujoco.txt")

if (!$SkipSmokeTest) {
  Push-Location $Root
  try {
    & $VenvPython -m mujoco_bridge.smoke_test
  } finally {
    Pop-Location
  }
}

Write-Host "MuJoCo environment is ready: $VenvPython"
