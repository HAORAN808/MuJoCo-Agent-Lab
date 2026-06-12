@echo off
setlocal

pushd "%~dp0.."
set "ROOT=%CD%"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "HTML=%ROOT%\web_demo\index.html"

if not exist "%PY%" (
  echo Virtual environment not found.
  echo Please run: .venv\Scripts\python.exe -m pip install -r requirements-mujoco.txt
  exit /b 1
)

echo Starting MuJoCo API on http://127.0.0.1:8765 ...
start "mujoco-api" /min "%PY%" -m mujoco_bridge.server --port 8765

ping 127.0.0.1 -n 4 >nul

echo Opening web demo...
start "" "%HTML%"

echo.
echo In the web page:
echo   1. Click the plan button
echo   2. Tick "Use MuJoCo API"
echo   3. Click the run button
echo.
echo To stop the API server later, close the "mujoco-api" window or kill python.exe started from this project.

popd
endlocal
