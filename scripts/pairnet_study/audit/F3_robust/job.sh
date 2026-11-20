#!/bin/bash
# OSPool wrapper: args <chunk name>
set -e
W=${_CONDOR_SCRATCH_DIR:-$PWD}; cd $W
export TMPDIR=$W/tmp; mkdir -p $TMPDIR env pyonnx
tar -xzf gen_env.tar.gz -C env; tar -xzf pyonnx.tar.gz -C pyonnx
source env/bin/activate; conda-unpack
export OMP_NUM_THREADS=1 PYTHONPATH=$W/pyonnx F3_ONNX=$W/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx
echo "host $(hostname)"
python scan.py $1.npz variants.txt $1_scan.npz
rm -rf env pyonnx gen_env.tar.gz pyonnx.tar.gz tmp
