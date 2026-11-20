#!/bin/bash
# OSPool worker: re-score one tar bundle of h5 files with the r14A ONNX keeping interpretation-level outputs.
# args: <bundle name> <kind qcd|sig>
set -e
NAME=$1; KIND=$2
W=${_CONDOR_SCRATCH_DIR:-$PWD}; cd $W
export TMPDIR=$W/tmp; mkdir -p $TMPDIR env pyonnx
tar -xzf gen_env.tar.gz -C env
tar -xzf pyonnx.tar.gz -C pyonnx
source env/bin/activate
conda-unpack
export OMP_NUM_THREADS=1 PYTHONPATH=$W/pyonnx
tar -xf $NAME.tar
echo "host $(hostname) $(ls in | wc -l) files"
python rescore.py $KIND $NAME.npz 0.5 in
test -s $NAME.npz || { echo "no output"; exit 1; }
rm -rf env pyonnx in tmp gen_env.tar.gz pyonnx.tar.gz $NAME.tar 2>/dev/null || true
