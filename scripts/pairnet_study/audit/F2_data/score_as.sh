#!/bin/bash
# score one h5 with r14A (no filter): score_as.sh in.h5 out.npz
export OMP_NUM_THREADS=1 PYTHONPATH=/scratch/llee/pairnet/stage/pyonnx
[ -s $2 ] || /scratch/llee/envs/pairnet/bin/python ~/CombinatorialSolver/scripts/score_lowmass.py $1 $2 r14A=/scratch/llee/pairnet/deliverables/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx > ${2%.npz}.log 2>&1
