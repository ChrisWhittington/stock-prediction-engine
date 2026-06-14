# make_shel_candidates.ps1
# Loops shel_trainer.py to generate fresh SHEL candidate models with random seeds.
# Each run dumps to shel_output\Best_Models_PT\candidates\<seed_dir>\ if it passes
# the trainer's candidate cutoff. Press Ctrl+C to stop.

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "AURUM-AI SHEL Seed Search - forward_days=20 lookback=30" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop"
Write-Host ""

$run      = 0
$bestFile = "shel_output\best_run_fwd20_lb30.txt"

while ($true) {
    $run++
    Write-Host ""
    Write-Host "--- Run $run ---" -ForegroundColor Cyan
    Write-Host ""

    python shel_trainer.py

    Write-Host ""
    if (Test-Path $bestFile) {
        Write-Host "Best result so far (20d):" -ForegroundColor Yellow
        Get-Content $bestFile
    }
    Write-Host ""
    Start-Sleep -Seconds 5
}
