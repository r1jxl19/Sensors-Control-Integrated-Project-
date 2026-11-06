@echo off
rem Double-click in File Explorer: sets up the project and checks it runs from a clean start (Windows).
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set PYTHONUTF8=1
echo == 41014 Track A project: setup and clean-start check ==
set "PY="
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo RESULT: NO_PYTHON - Python 3.10 or newer not found.
  echo Install Python 3.12 from python.org, tick "Add python.exe to PATH", then double-click this file again.
  start "" "https://www.python.org/downloads/windows/"
  pause
  exit /b 1
)
echo Using !PY!
if not exist ".venv\Scripts\python.exe" (
  echo == Creating virtual environment .venv ==
  !PY! -m venv .venv || (echo RESULT: VENV_FAILED & pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"
echo == Installing packages, first time takes a few minutes ==
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q || (echo RESULT: PIP_FAILED & pause & exit /b 1)
echo == Unit tests ==
python -m pytest -q tests || (echo RESULT: TESTS_FAILED & pause & exit /b 1)
echo == Nominal mission, seed 101, about 30 s ==
python scripts\run_mission.py --seed 101 || (echo RESULT: MISSION_FAILED & pause & exit /b 1)
echo RESULT: ALL_OK
echo To watch it live, double-click demo_live.bat
pause
