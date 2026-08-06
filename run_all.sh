#!/usr/bin/env bash
# Convenience wrapper: install deps (once) and reproduce all analyses.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running full pipeline..."
cd code
python run_all.py "$@"
