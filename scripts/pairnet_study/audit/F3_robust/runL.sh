#!/bin/bash
# usage: runL.sh <indir> <variants> <outdir> name...   (local, one thread; skips finished outputs)
cd /scratch/llee/pairnet/audit/F3_robust
export OMP_NUM_THREADS=1 PYTHONPATH=/scratch/llee/pairnet/stage/pyonnx
I=$1; V=$2; O=$3; shift 3
for n in "$@"; do
  [ -s $O/$n.npz ] && continue
  /scratch/llee/envs/gen/bin/python scan.py $I/$n.npz $V $O/tmp_$n.npz > logs/L/$n.log 2>&1 && mv $O/tmp_$n.npz $O/$n.npz
done
