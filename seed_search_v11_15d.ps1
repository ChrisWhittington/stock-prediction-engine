Write-Host "AURUM-AI v11 Seed Search - forward_days=15 lookback=25"
Write-Host "======================================================="
Write-Host "Press Ctrl+C to stop"
Write-Host ""

$run = 0
while ($true) {
    $run++
    Write-Host "--- Run $run ---"
    python gold_miner_trainer.py
    Write-Host ""
    $bestFile = "aurum_output\best_run_fwd15_lb25.txt"
    if (Test-Path $bestFile) {
        Write-Host "Best result so far (15d):"
        Get-Content $bestFile
    }
    Write-Host ""
    Write-Host "Waiting 120 seconds..."
    Start-Sleep -Seconds 120
}