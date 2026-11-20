#!/bin/bash
# score a local gen_f2 h5 with r14A (unfiltered) and append pthat/sym_pass: loc_score.sh file.h5
cd /scratch/llee/pairnet/audit/F2_data
export OMP_NUM_THREADS=1 PYTHONPATH=/scratch/llee/pairnet/stage/pyonnx; PY=/scratch/llee/envs/pairnet/bin/python
o=${1%.h5}.npz
$PY ~/CombinatorialSolver/scripts/score_lowmass.py $1 $o r14A=/scratch/llee/pairnet/deliverables/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx > ${1%.h5}.score.log 2>&1 && $PY f2_post.py $1 $o >> ${1%.h5}.score.log 2>&1
