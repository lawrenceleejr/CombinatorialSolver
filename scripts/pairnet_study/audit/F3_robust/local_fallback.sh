#!/bin/bash
# local fallback while the pool is busy: same script, same output names; skips chunks that already have an output
cd /scratch/llee/pairnet/audit/F3_robust
export OMP_NUM_THREADS=1 PYTHONPATH=/scratch/llee/pairnet/stage/pyonnx
for n in "$@"; do
  [ -s out/scan/${n}_scan.npz ] && continue
  /scratch/llee/envs/gen/bin/python scan.py cache/chunks/$n.npz variants.txt out/local_$n.npz > logs/local_$n.log 2>&1 && [ ! -s out/scan/${n}_scan.npz ] && mv out/local_$n.npz out/scan/${n}_scan.npz
  rm -f out/local_$n.npz
done
