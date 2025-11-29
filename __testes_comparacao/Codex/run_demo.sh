#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv || python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

python sample_data/MAKE_SYNTHETIC.py
python cli.py --demo --trials 120

echo "Artifacts in: $(realpath output)"

