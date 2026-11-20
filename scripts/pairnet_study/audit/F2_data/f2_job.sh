#!/bin/bash
# F2 audit OSPool wrapper: gen_f2.py (line N of jobs file) -> score with r14A (no filter) -> append pthat/sym_pass
set -e
JOBS=$1; N=$2; NAME=$3
W=${_CONDOR_SCRATCH_DIR:-$PWD}; cd $W
export TMPDIR=$W/tmp; mkdir -p $TMPDIR
mkdir -p env repo pyonnx
tar -xzf gen_env.tar.gz -C env
tar -xzf repo_tail.tar.gz -C repo
tar -xzf pyonnx.tar.gz -C pyonnx
source env/bin/activate
conda-unpack
export PYTHIA8DATA=$W/env/share/Pythia8/xmldoc OMP_NUM_THREADS=1 PYTHONPATH=$W/pyonnx
ARGS=$(sed -n "${N}p" $JOBS)
echo "host $(hostname) args: $ARGS"
eval "python gen_f2.py $ARGS --out $NAME.h5 --etamax 3.0 --ptmin 30.0" > $NAME.log 2>&1 || { echo "gen failed"; tail -5 $NAME.log; exit 1; }
python repo/scripts/score_lowmass.py $NAME.h5 $NAME.npz r14A=$W/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx >> $NAME.log 2>&1
python f2_post.py $NAME.h5 $NAME.npz >> $NAME.log 2>&1
grep -v "^#" $NAME.log | tail -3
test -s $NAME.npz
[ "$4" = keep ] || rm -f $NAME.h5; rm -rf env repo pyonnx *.tar.gz tmp 2>/dev/null || true
