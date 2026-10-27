#!/bin/bash
# OSPool worker wrapper: generate (line N of the jobs file), then score the output with the PairNet ONNX models.
# args: <jobs file> <line number (1-based)> <output basename (.h5)>
set -e
JOBS=$1; N=$2; OUT=$3
W=${_CONDOR_SCRATCH_DIR:-$PWD}; cd $W
export TMPDIR=$W/tmp; mkdir -p $TMPDIR
mkdir -p env repo pyonnx
tar -xzf gen_env.tar.gz -C env
tar -xzf repo_tail.tar.gz -C repo
tar -xzf pyonnx.tar.gz -C pyonnx
source env/bin/activate
conda-unpack
export PYTHIA8DATA=$W/env/share/Pythia8/xmldoc OMP_NUM_THREADS=1 PYTHONPATH=$W/pyonnx
CMD=$(sed -n "${N}p" $JOBS)
CMD=${CMD//\/scratch\/llee\/envs\/pairnet\/bin\/python/python}
CMD=${CMD//\/scratch\/llee\/CombinatorialSolver/$W\/repo}
CMD=${CMD//\/scratch\/llee\/pairnet\/prod\//$W\/}
echo "host $(hostname)  cmd: $CMD"
eval "$CMD" || { echo "failed rc=$?"; tail -5 $OUT.log 2>/dev/null; exit 1; }
tail -2 $OUT.log
test -s ${OUT%.h5}.npz || { echo "no output"; exit 1; }
rm -rf env repo pyonnx gen_env.tar.gz repo_tail.tar.gz pyonnx.tar.gz tmp 2>/dev/null || true
