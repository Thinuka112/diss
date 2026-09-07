@echo off
REM ============================================================
REM  Swipe Review - double-click launcher (no command line needed)
REM
REM  Opens straight on the sign-in screen using the bundled imageset
REM  (..\HPA_small_intestine). To review a DIFFERENT imageset, add flags
REM  after swipe_review.py below, e.g.:
REM     python swipe_review.py --images "C:\path\to\my_images"
REM     python swipe_review.py --manifest "C:\path\to\manifest.csv" --path-col tile_path
REM ============================================================
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
    python swipe_review.py %*
) else (
    py -3 swipe_review.py %*
)

REM Keep the window open only if something went wrong, so errors are visible.
if errorlevel 1 (
    echo.
    echo Swipe Review exited with an error. See the messages above.
    pause
)
