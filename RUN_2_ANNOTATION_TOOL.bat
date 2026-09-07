@echo off
cd /d "%~dp0"
echo Opening the annotation tool on the bundled sample slides...
echo (Sign in with any name, e.g. "examiner". Arrow keys to judge, Esc to quit.)
python review_tool\swipe_review.py --images sample_images
pause
