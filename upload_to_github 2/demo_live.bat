@echo off
rem Double-click in File Explorer: opens the live mission window (run setup_windows.bat once first).
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\activate.bat" (
  echo Run setup_windows.bat first
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python scripts\view_mission.py --seed 101 --speed 2
pause
