#!/bin/bash
# OSPool GPU wrapper for the F5 audit trainings (after /scratch/llee/pairnet/train_auto2.sh): unpack the
# conda-packed env, the repo tarball and the data tarballs, run train_pairnet.py with HTCondor
# self-checkpointing.  The training exits with code 85 by itself after the last epoch that fits in
# its --time-budget (so no epoch is lost); `timeout` at LIMIT seconds is the safety net.
# args: <seconds before forced checkpoint exit> <train_pairnet.py args...>
LIMIT=$1; shift
W=${_CONDOR_SCRATCH_DIR:-$PWD}; cd $W
mkdir -p env repo && tar -xzf train_env_v1.tar.gz -C env && rm -f train_env_v1.tar.gz && tar -xzf repo_*.tar.gz -C repo && rm -f repo_*.tar.gz
for t in data_*.tar*; do
  [ -f "$t" ] || continue
  tar -xf "$t" && rm -f "$t"
done
source env/bin/activate && conda-unpack
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
echo "host $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "no usable GPU (or env missing): checkpoint exit so HTCondor restarts elsewhere"; exit 85; }
AMP=$(python -c "import torch; print('--amp' if torch.cuda.get_device_capability(0)[0] >= 8 else '')")
OUT=$(python - "$@" <<'PY'
import sys; a = sys.argv[1:]; print(a[a.index("--out") + 1])
PY
)
mkdir -p "$(dirname "$OUT")"
EXTRA=""; [ -f "$OUT/last.pt" ] && EXTRA="--resume" && echo "resuming from $OUT/last.pt"
echo "cmd: python repo/scripts/train_pairnet.py $* $AMP $EXTRA"
timeout --signal=TERM $LIMIT python -u repo/scripts/train_pairnet.py "$@" $AMP $EXTRA 2>&1 | tee -a $OUT.train.log
rc=${PIPESTATUS[0]}
rm -rf repo env cache f5
if [ $rc -eq 124 ] || [ $rc -eq 85 ]; then echo "time slice over, checkpointing"; exit 85; fi
exit $rc
