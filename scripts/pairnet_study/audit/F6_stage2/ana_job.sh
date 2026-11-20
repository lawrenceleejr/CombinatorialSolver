#!/bin/bash
# OSPool worker: run one F6 analysis script. args: <data tar> <script> <out json> [script args...]; <env K=V:K=V or -> <threads>
set -e
DATA=$1; SCRIPT=$2; OUT=$3; ENVS=$4; OMP=$5; shift 5
W=${_CONDOR_SCRATCH_DIR:-$PWD}; cd $W
export TMPDIR=$W/tmp; mkdir -p $TMPDIR env pysk out
tar -xzf gen_env.tar.gz -C env; tar -xzf pysk.tar.gz -C pysk
source env/bin/activate; conda-unpack
export PYTHONPATH=$W/pysk OMP_NUM_THREADS=${OMP:-4} MPLBACKEND=Agg
tar -xf $DATA
[ "$ENVS" = "-" ] || for kv in ${ENVS//:/ }; do export "$kv"; done
echo "host $(hostname) cpus $(nproc) data $(ls ${F6_RES:-rescored} | wc -l) files"
python -u $SCRIPT out/$OUT "$@"
tar -cf ${OUT%.json}.tar out
