#!/bin/bash
# Double-click in Finder: runs the room you laid out in scripts/custom_scene.py (EDIT HERE block).
cd "$(dirname "$0")"
source .venv/bin/activate || { echo "Run setup_mac.command first"; read -p "Press Enter"; exit 1; }
python scripts/custom_scene.py --speed 2
read -p "Press Enter to close"
