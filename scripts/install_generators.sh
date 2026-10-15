#!/usr/bin/env bash
# Install what scripts/generate_pythia.py needs: Pythia 8 (python bindings) + FastJet.
#   conda available:  conda-forge has both prebuilt (fastest)
#   otherwise:        FastJet from PyPI wheels; Pythia built from the pythia8mc
#                     sdist with a parallel compile (scripts/build_pythia_wheel.py)
set -euo pipefail
JOBS=${JOBS:-$(nproc)}
here=$(cd "$(dirname "$0")" && pwd)

if python -c "import pythia8mc" 2>/dev/null || python -c "import pythia8" 2>/dev/null; then
  echo "Pythia python bindings already importable"
elif command -v conda >/dev/null && [ "${USE_CONDA:-1}" = 1 ]; then
  conda install -y -c conda-forge pythia8 fastjet
else
  pip install h5py numpy scipy
  pip install fastjet
  work=$(mktemp -d)
  python "$here/build_pythia_wheel.py" --jobs "$JOBS" --workdir "$work"
  pip install "$work"/pythia8mc-*/dist/*.whl
fi
python -c "import fastjet" 2>/dev/null || pip install fastjet
python - <<'PY'
try:
    import pythia8mc as p
except ImportError:
    import pythia8 as p
import fastjet
p.Pythia("", False)
print("ok: pythia + fastjet")
PY
