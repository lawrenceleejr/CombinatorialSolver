#!/bin/bash
# local signal-variation run: sv_local.sh name seed mass n "extra gen options"  -> loc/sv_<name>.h5/.npz (r14A scores, no filter)
cd /scratch/llee/pairnet/audit/F2_data
export PYTHIA8DATA=/scratch/llee/envs/pairnet/share/Pythia8/xmldoc OMP_NUM_THREADS=1 PYTHONPATH=/scratch/llee/pairnet/stage/pyonnx
PY=/scratch/llee/envs/pairnet/bin/python
$PY gen_f2.py --process signal --mass $3 --n $4 --seed $2 --min-jets 6 --truth-quarks before_fsr $5 --out loc/sv_$1.h5 --etamax 3.0 --ptmin 30.0 > loc/sv_$1.log 2>&1 && \
$PY ~/CombinatorialSolver/scripts/score_lowmass.py loc/sv_$1.h5 loc/sv_$1.npz r14A=/scratch/llee/pairnet/deliverables/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx >> loc/sv_$1.log 2>&1
