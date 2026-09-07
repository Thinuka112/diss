@echo off
cd /d "%~dp0"
echo Running the Stage-2 localisation evaluation against the expert gold standard...
echo.
python mil_eval_gold.py
echo.
pause
