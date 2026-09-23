$ErrorActionPreference = 'Stop'

$projectPath = 'C:\its\test'
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
$appPath = Join-Path $projectPath 'app.py'
$outputLogPath = Join-Path $projectPath 'server-autostart.out.log'
$errorLogPath = Join-Path $projectPath 'server-autostart.err.log'

# Do not start a second server when port 5000 is already in use.
$listener = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    exit 0
}

Set-Location -LiteralPath $projectPath

try {
    $server = Start-Process -FilePath $pythonPath `
        -ArgumentList $appPath `
        -WorkingDirectory $projectPath `
        -RedirectStandardOutput $outputLogPath `
        -RedirectStandardError $errorLogPath `
        -PassThru `
        -Wait
    exit $server.ExitCode
}
catch {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') startup failed: $($_.Exception.Message)" |
        Add-Content -LiteralPath $errorLogPath -Encoding UTF8
    exit 1
}
