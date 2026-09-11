# run_observatory.ps1
# Launches the Corsica Well-being Observation Assistant (RAPTOR v10 + web UI).
#
#   .\local\observatory\run_observatory.ps1
#   .\local\observatory\run_observatory.ps1 -Port 8700 -DevCode monsecret
#
# Requires CLAUDE_API_KEY and MISTRAL_API_KEY in the repo-root .env
# (the same keys rag_v10_raptor_subq.py already uses).

param(
    [int]    $Port    = 8600,
    [string] $DevHost = "127.0.0.1",
    [string] $DevCode = "corse2026",
    [switch] $NoV11
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent (Split-Path -Parent $scriptDir)
Set-Location $repoRoot

$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$env:OBS_PORT     = "$Port"
$env:OBS_HOST     = $DevHost
$env:OBS_DEV_CODE = $DevCode
if ($NoV11) { $env:OBS_ENABLE_V11 = "0" }

Write-Host "[observatory] python : $py"
Write-Host "[observatory] UI     : http://$DevHost`:$Port/"
Write-Host "[observatory] dev    : http://$DevHost`:$Port/?dev=$DevCode"
Write-Host "[observatory] loading the RAPTOR pipeline can take ~30-60 s on first start."

& $py (Join-Path $scriptDir "server.py")
