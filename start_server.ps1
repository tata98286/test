$ErrorActionPreference = 'Stop'

$projectPath = 'C:\its\test'
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
$appPath = Join-Path $projectPath 'app.py'
$logPath = Join-Path $projectPath 'server-autostart.log'

# Do not start a second server when port 5000 is already in use.
$listener = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    exit 0
}

Set-Location -LiteralPath $projectPath

try {
    & $pythonPath $appPath *>> $logPath
}
catch {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') startup failed: $($_.Exception.Message)" |
        Add-Content -LiteralPath $logPath -Encoding UTF8
    exit 1
}
