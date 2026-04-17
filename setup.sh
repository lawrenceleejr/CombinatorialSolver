#!/usr/bin/env bash
# setup.sh – create a virtual environment and install dependencies
set -euo pipefail

VENV_DIR=".venv"

# ── Python version check ────────────────────────────────────────────────────
PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: '$PYTHON' not found. Install Python 3.9+ and retry, or set the PYTHON env var." >&2
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print('%d.%d' % sys.version_info[:2])")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info[0])")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info[1])")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    echo "ERROR: Python 3.9+ required (found $PY_VERSION)." >&2
    exit 1
fi
echo "✓ Python $PY_VERSION"

# ── Create / reuse virtual environment ─────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "✓ Virtual environment '$VENV_DIR' already exists – reusing it"
else
    echo "→ Creating virtual environment in '$VENV_DIR' …"
    "$PYTHON" -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Upgrade pip ──────────────────────────────────────────────────────────────
echo "→ Upgrading pip …"
pip install --quiet --upgrade pip

# ── Install dependencies ─────────────────────────────────────────────────────
echo "→ Installing requirements …"
pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo " Environment ready!  Activate it with:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " Quick-start:"
echo "   # Generate mock data"
echo "   python scripts/generate_mock_data.py --output data/mock_data.h5 --n-events 10000"
echo ""
echo "   # Train"
echo "   python -m src.train --config configs/default.yaml --data 'data/*.h5'"
echo ""
echo "   # Evaluate"
echo "   python -m src.evaluate --checkpoint checkpoints/best_model.pt \\"
echo "       --data 'data/test*.h5' --output results"
echo "══════════════════════════════════════════════════════"
