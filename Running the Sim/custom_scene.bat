@echo off
rem Double-click in File Explorer: runs the room laid out in scripts\custom_scene.py (EDIT HERE block).
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\activate.bat" (
  echo Run setup_windows.bat first
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python scripts\custom_scene.py --speed 2
pause
