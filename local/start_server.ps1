# start_server.ps1 — Lance llama-server avec les locks GPU.
#
# DOIT être exécuté depuis un terminal PowerShell administrateur.
# nvidia-smi -lgc/-lmc requiert les droits admin ; sans eux, les locks sont silencieusement ignorés.
#
# -lgc : verrouille le SM clock minimum à 1500 MHz (max 3105 MHz).
#   Sans ce lock, le SM peut descendre à 210 MHz en idle entre deux tokens.
#
# -lmc : lock memory clock à 8001 MHz.
#   Effet à vérifier sous charge (voir local/measure_gpu.ps1).
#   À l'idle le GPU retourne toujours en P8/405 MHz — c'est normal.
#
# Libérer les locks à l'arrêt du serveur (depuis un terminal admin) :
#   nvidia-smi -rgc   (SM)
#   nvidia-smi -rmc   (memory)

param(
    [string]$Model      = "C:\models\Qwen_Qwen3.5-9B-Q4_K_M.gguf",
    [string]$Alias      = "Qwen3.5-9B-Q4_K_M",
    [int]   $CtxSize    = 8192,
    [int]   $Parallel   = 1,
    [string]$ServerHost = "127.0.0.1",
    [int]   $Port       = 8080
)

# 1. Vérification droits admin
$isAdmin = ([Security.Principal.WindowsPrincipal] `
            [Security.Principal.WindowsIdentity]::GetCurrent() `
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "Ce script doit etre lance depuis un terminal PowerShell administrateur."
    Write-Error "Clic-droit sur PowerShell -> 'Exécuter en tant qu'administrateur', puis relancer."
    exit 1
}

# 2. Lock SM clock
Write-Host "[start_server] Lock SM clock (nvidia-smi -lgc 1500,3105)..."
nvidia-smi -lgc 1500,3105
if ($LASTEXITCODE -ne 0) {
    Write-Warning "nvidia-smi -lgc a echoue (code $LASTEXITCODE). Perf SM degradee possible."
}

# 3. Lock memory clock
Write-Host "[start_server] Lock memory clock (nvidia-smi -lmc 8001,8001)..."
nvidia-smi -lmc 8001,8001
if ($LASTEXITCODE -ne 0) {
    Write-Warning "nvidia-smi -lmc a echoue (code $LASTEXITCODE). Memory clock non verrouille."
}

# 4. Diagnostic post-lock (à l'idle — memory remontera sous charge)
$clocks = (nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,pstate,power.draw --format=csv,noheader).Trim()
Write-Host "[start_server] Etat GPU (idle) : $clocks"
Write-Host "[start_server] Cible sous charge : SM >= 1500 MHz, memory = 8001 MHz, pstate = P0"

# 5. Lancer llama-server
Write-Host "[start_server] Demarrage llama-server sur ${ServerHost}:${Port} ..."
Write-Host "[start_server] Modele : $Model"
Write-Host "[start_server] Contexte : $CtxSize tokens, parallel : $Parallel"

& "C:\tools\llama.cpp\llama-server.exe" `
    --model            $Model `
    --alias            $Alias `
    --n-gpu-layers     99 `
    --ctx-size         $CtxSize `
    --flash-attn       on `
    --cache-type-k     q8_0 `
    --cache-type-v     q8_0 `
    --cache-reuse      256 `
    --parallel         $Parallel `
    --host             $ServerHost `
    --port             $Port `
    --jinja `
    --reasoning-budget 0

# llama-server bloque jusqu'a Ctrl+C.
# Apres Ctrl+C, liberer les locks (admin requis) :
#   nvidia-smi -rgc && nvidia-smi -rmc
