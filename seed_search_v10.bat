@echo off
echo AURUM-AI v10 Seed Search - forward_days=20 lookback=30
echo =======================================================
echo Targeting IC better than current best
echo Press Ctrl+C to stop
echo.

:loop
echo.
echo --- New run ---
python gold_miner_trainer.py
echo.
if exist aurum_output\best_run_fwd20_lb30.txt (
    echo Best result so far (20d):
    type aurum_output\best_run_fwd20_lb30.txt
)
echo.
echo Waiting 120 seconds before next run...
timeout /t 120 /nobreak >nul
goto loop
