# measure_gpu.ps1 — Poll nvidia-smi pendant une génération llama-server.
#
# Usage (depuis un terminal admin, pendant qu'une génération tourne) :
#   .\local\measure_gpu.ps1
#
# Sortie : une ligne toutes les 2 s avec SM clock, memory clock, pstate, power.
# Arrêt : Ctrl+C
#
# Protocole de comparaison -lmc :
#   1. Lancer start_server.ps1 SANS -lmc (commenter l'étape 3), démarrer une génération, lancer ce script.
#   2. Répéter AVEC -lmc actif.
#   3. Comparer la colonne "mem" entre les deux runs.

Write-Host "$(Get-Date -Format 'HH:mm:ss')  sm_mhz  mem_mhz  pstate  power_w"
Write-Host "-------------------------------------------------------------------"

while ($true) {
    $raw = (nvidia-smi `
        --query-gpu=clocks.current.sm,clocks.current.memory,pstate,power.draw `
        --format=csv,noheader).Trim()
    # raw = "1530 MHz, 8001 MHz, P0, 45.23 W"
    $parts = $raw -split ",\s*"
    $sm  = $parts[0] -replace " MHz",""
    $mem = $parts[1] -replace " MHz",""
    $ps  = $parts[2]
    $pw  = $parts[3] -replace " W",""
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  $($sm.PadLeft(6))  $($mem.PadLeft(7))  $($ps.PadLeft(6))  $($pw.PadLeft(7))"
    Start-Sleep -Seconds 2
}
