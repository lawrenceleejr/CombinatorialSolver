#!/bin/bash
# OSPool worker wrapper (F4 audit): labelled signal generation. args: <jobs file> <line number (1-based)> <output .h5>
set -e
JOBS=$1; N=$2; OUT=$3
W=${_CONDOR_SCRATCH_DIR:-$PWD}; cd $W
export TMPDIR=$W/tmp; mkdir -p $TMPDIR env
tar -xzf gen_env.tar.gz -C env
source env/bin/activate
conda-unpack
export PYTHIA8DATA=$W/env/share/Pythia8/xmldoc OMP_NUM_THREADS=1
CMD=$(sed -n "${N}p" $JOBS)
echo "host $(hostname)  cmd: $CMD"
eval "$CMD" 2>&1 | tail -3
test -s $OUT || { echo "no output"; exit 1; }
rm -rf env gen_env.tar.gz tmp 2>/dev/null || true
