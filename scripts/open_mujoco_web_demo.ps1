$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Html = Join-Path $Root "web_demo\index.html"

if (!(Test-Path $Python)) {
  throw "Virtual environment not found. Run scripts\setup_mujoco_env.ps1 first."
}

$server = Start-Process -FilePath $Python -ArgumentList @("-m", "mujoco_bridge.server", "--port", "8765") -WorkingDirectory $Root -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 3

$edgePaths = @(
  "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
  "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
  "C:\Program Files\Google\Chrome\Application\chrome.exe",
  "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
)

$browser = $edgePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
$url = "file:///" + (($Html | Resolve-Path).Path -replace "\\", "/")

if ($browser) {
  Start-Process -FilePath $browser -ArgumentList $url
} else {
  Start-Process $url
}

Write-Host "MuJoCo API started on http://127.0.0.1:8765"
Write-Host "PID: $($server.Id)"
Write-Host "In the web page, use the NLP panel and click '一键运行 NLP 管线'."
