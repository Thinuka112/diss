@echo off
cd /d "%~dp0"
echo Reproducing the Stage-1 headline (gold-only ResNet-50) from the shipped predictions...
echo.
python demo_stage1.py
echo.
pause
