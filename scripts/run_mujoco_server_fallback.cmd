@echo off
setlocal

pushd "%~dp0.."
set "ROOT=%CD%"
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Virtual environment not found.
  exit /b 1
)

"%PY%" -m mujoco_bridge.server --fallback --port 8765

popd
endlocal
