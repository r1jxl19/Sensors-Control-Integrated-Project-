#!/bin/bash
# Double-click in Finder: opens the live mission window (run setup_mac.command once first).
cd "$(dirname "$0")"
source .venv/bin/activate || { echo "Run setup_mac.command first"; read -p "Press Enter"; exit 1; }
python scripts/view_mission.py --seed 101 --speed 2
