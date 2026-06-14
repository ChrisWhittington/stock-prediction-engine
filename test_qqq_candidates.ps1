# test_qqq_candidates.ps1
# Tests every candidate in qqq_output\Best_Models_PT\candidates\

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$candidatesDir = "qqq_output\Best_Models_PT\candidates"
$qqqOutputDir  = "qqq_output"
$resultsFile   = "qqq_output\candidate_results.txt"

"" | Set-Content $resultsFile

$candidates = Get-ChildItem -Path $candidatesDir -Directory | Sort-Object Name

if ($candidates.Count -eq 0) {
    Write-Host "No candidates found in $candidatesDir"
    exit
}

Write-Host "Found $($candidates.Count) candidates to test"
Write-Host ""

foreach ($candidate in $candidates) {

    Write-Host "========================================"
    Write-Host "Testing: $($candidate.Name)"
    Write-Host "========================================"

    $ptFiles = Get-ChildItem -Path $candidate.FullName -Filter "fold_*.pt"
    if ($ptFiles.Count -eq 0) {
        Write-Host "  No fold_*.pt files - skipping"
        continue
    }

    # Clean stale fold models + prediction CSVs from the output dir before each
    # candidate, so an incomplete candidate folder (e.g. training crashed before
    # save_full_predictions) can't silently inherit the previous candidate's
    # files. Without this, qqq_backtest.py would read stale predictions and
    # produce duplicate metrics. See 2026-05-27 changelog in PROJECT_NOTES.md.
    Get-ChildItem -Path $qqqOutputDir -Filter "fold_*.pt" -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    foreach ($csv in @("oof_predictions.csv", "full_predictions.csv")) {
        $dest = Join-Path $qqqOutputDir $csv
        if (Test-Path $dest) { Remove-Item $dest -Force -ErrorAction SilentlyContinue }
    }

    # Warn loudly if this candidate folder is incomplete
    if (-not (Test-Path (Join-Path $candidate.FullName "full_predictions.csv"))) {
        Write-Host "  WARNING: candidate is missing full_predictions.csv (backtest will fall back to OOF)" -ForegroundColor Yellow
    }
    if (-not (Test-Path (Join-Path $candidate.FullName "oof_predictions.csv"))) {
        Write-Host "  WARNING: candidate is missing oof_predictions.csv" -ForegroundColor Yellow
    }

    foreach ($pt in $ptFiles) {
        Copy-Item $pt.FullName -Destination $qqqOutputDir -Force
    }

    foreach ($csv in @("oof_predictions.csv", "full_predictions.csv")) {
        $src = Join-Path $candidate.FullName $csv
        if (Test-Path $src) {
            Copy-Item $src -Destination $qqqOutputDir -Force
        }
    }

    Write-Host "  Copied $($ptFiles.Count) fold models"

    $output = python -X utf8 qqq_backtest.py 2>&1
    $output | Write-Host

    $cagr    = ($output | Select-String "Annual return").Line
    $sharpe  = ($output | Select-String "Sharpe ratio").Line
    $maxdd   = ($output | Select-String "Max drawdown").Line
    $winrate = ($output | Select-String "Win rate").Line
    $longpct = ($output | Select-String "Days long").Line

    Add-Content $resultsFile "========================================"
    Add-Content $resultsFile "Candidate: $($candidate.Name)"
    Add-Content $resultsFile "$cagr"
    Add-Content $resultsFile "$sharpe"
    Add-Content $resultsFile "$maxdd"
    Add-Content $resultsFile "$winrate"
    Add-Content $resultsFile "$longpct"
    Add-Content $resultsFile ""

    Write-Host ""
}

Write-Host "========================================"
Write-Host "Done. Summary in $resultsFile"
Write-Host "========================================"
Write-Host ""
Get-Content $resultsFile