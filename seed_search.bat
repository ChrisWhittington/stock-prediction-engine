@echo off
echo AURUM-AI Seed Search Loop
echo =========================
echo Targeting IC better than 0.3389
echo Press Ctrl+C to stop
echo.

:loop
echo.
echo --- New run ---
python gold_miner_trainer.py
echo.
if exist aurum_output\best_run.txt (
    echo Best result so far:
    type aurum_output\best_run.txt
)
echo.
timeout /t 120 /nobreak >nul
goto loop