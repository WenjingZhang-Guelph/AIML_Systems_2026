#!/usr/bin/env bash
# Start the gallery without Docker.
#
# The whole reason this script is more than two lines: nibi's python3 comes from
# CVMFS, and `python3 -m venv` against a CVMFS interpreter produces a venv that
# looks fine and then can't import anything. So we can't just create a venv and
# get on with it — we have to work out which machine we're on first and pick a
# strategy. Hence the four-branch cascade below.
#
#   ./run.sh             # http://localhost:8000
#   PORT=9000 ./run.sh

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ ! -f graphs_manifest.json ]; then
    echo "ERROR: graphs_manifest.json not found."
    echo "Run this from clinical_preprocessing_toolbox/gallery/ with the sibling ../figures/ directory present."
    exit 1
fi

if [ ! -d ../figures ]; then
    echo "ERROR: ../figures/ not found — the gallery reads graph images from the sibling figures/ directory."
    exit 1
fi

# --- Find a Python that can import Flask ---
# Ordered cheapest-and-least-surprising first: reuse what's already there before
# building anything, and never create a venv on nibi.

LOCAL_VENV="$DIR/.venv"
# Reusing the MIMIC_Extract stats venv rather than maintaining a separate one —
# it already has Flask and it's the venv the comparison notebook runs under.
NIBI_VENV="/home/ccampb47/work/clinical_preprocessing_toolbox/pipelines/MIMIC_Extract/.venv_stats"

PYTHON=""

# Already activated something that works? Use it and don't ask questions.
if python3 -c "import flask" 2>/dev/null; then
    PYTHON="$(which python3)"

elif [ -f "$LOCAL_VENV/bin/python3" ] && "$LOCAL_VENV/bin/python3" -c "import flask" 2>/dev/null; then
    PYTHON="$LOCAL_VENV/bin/python3"

# A /cvmfs/ interpreter means we're on Compute Canada. Bail out to .venv_stats
# instead of trying to build a venv, which would appear to succeed and then fail
# at import time.
elif [[ "$(python3 -c 'import sys; print(sys.executable)')" == /cvmfs/* ]]; then
    if [ -f "$NIBI_VENV/bin/python3" ] && "$NIBI_VENV/bin/python3" -c "import flask" 2>/dev/null; then
        PYTHON="$NIBI_VENV/bin/python3"
    else
        echo "ERROR: On nibi but .venv_stats Flask not found."
        echo "Run: source $NIBI_VENV/bin/activate"
        exit 1
    fi

# Ordinary machine, nothing set up yet — build the venv for them.
elif python3 -m venv "$LOCAL_VENV" && [ -f "$LOCAL_VENV/bin/pip" ]; then
    echo "Installing Flask into .venv/ ..."
    "$LOCAL_VENV/bin/pip" install -r "$DIR/requirements.txt" --quiet
    PYTHON="$LOCAL_VENV/bin/python3"

else
    echo ""
    echo "ERROR: Could not set up Python with Flask."
    echo "Please create a venv manually and install dependencies:"
    echo "  python3 -m venv .venv && .venv/bin/pip install flask && .venv/bin/python3 app/server.py"
    exit 1
fi

PORT="${PORT:-8000}"
echo "Starting Graph Gallery on http://localhost:${PORT}"
PORT="${PORT}" "$PYTHON" app/server.py
