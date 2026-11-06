#!/bin/bash
# Double-click in Finder: sets up the project and checks it runs from a clean start.
cd "$(dirname "$0")"
exec > >(tee viva_check.log) 2>&1
echo "== 41014 Track A project: setup and clean-start check =="
date
PY=""
for c in /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
         /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
         /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
         /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 \
         /opt/homebrew/bin/python3 /usr/local/bin/python3 \
         python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print(sys.version_info[0]*100 + sys.version_info[1])' 2>/dev/null)
    if [ -n "$v" ] && [ "$v" -ge 310 ]; then PY="$(command -v "$c")"; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "RESULT: NO_PYTHON - Python 3.10 or newer not found (macOS's built-in python3 is 3.9)."
  echo "Install Python 3.12 from python.org (opening the page now), then double-click this file again."
  open "https://www.python.org/downloads/macos/"
  read -p "Press Enter to close"; exit 1
fi
echo "Using $PY ($("$PY" --version))"
if [ ! -x .venv/bin/python ]; then
  echo "== Creating virtual environment .venv =="
  "$PY" -m venv .venv || { echo "RESULT: VENV_FAILED"; read -p "Press Enter"; exit 1; }
fi
source .venv/bin/activate
echo "== Installing packages (first time takes a few minutes) =="
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q || { echo "RESULT: PIP_FAILED"; read -p "Press Enter"; exit 1; }
echo "== Unit tests =="
python -m pytest -q tests || { echo "RESULT: TESTS_FAILED"; read -p "Press Enter"; exit 1; }
echo "== Nominal mission (seed 101, about 30 s) =="
python scripts/run_mission.py --seed 101 || { echo "RESULT: MISSION_FAILED"; read -p "Press Enter"; exit 1; }
echo "RESULT: ALL_OK"
echo "To watch it live, double-click demo_live.command"
read -p "Press Enter to close"
