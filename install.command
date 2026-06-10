#!/bin/bash
# Run this once after cloning the repo on Mac.
# It installs all Python dependencies.

cd "$(dirname "$0")"
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Done. You can now double-click start.command to launch the dashboard."
echo ""
read -p "Press Enter to close..."
