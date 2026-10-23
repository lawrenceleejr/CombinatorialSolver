#!/bin/bash
# Mass-window counting with a 10% background uncertainty (user's background estimate), r14A, held-out MC.
S=/tmp/claude-12421/-scratch-llee/a35d08e7-9d87-5923-8644-0d53c5b0f767/scratchpad
ST=$(ls -d /home/llee/.claude/skills/synced/*/tufte-plots/assets | head -1)
PY=/scratch/llee/envs/pairnet/bin/python; C=/scratch/llee/pairnet/cuts_study
E=/ospool/uc-shared/project/futurecolliders/llee/pairnet/eval_r14
run() {  # trig peakmax
  local T=$1 PK=$2; local O=/scratch/llee/pairnet/lowmass14win10_pk$PK; mkdir -p $O
  local env="SEED_GRID=1 MODEL=r14A LMDIR=$E OLD_NPZ=$C/big_vars_r14A_pn.npz HT_NPZ_FILE=$C/big_vars_ht_r14A_ge1400.npz NO_LM=1 SIG_CLASSES=16,17,18,19 FTAG=f0.1 DUMP_SPECTRA=$O/dump"
  [ $PK != none ] && env="$env PEAK_MAX=$PK"
  env $env $PY $S/lowmass_study4.py $ST $O eval 0.1 $T > $O/eval_t$T.log 2>&1
}
set -- 450:0.75 450:none 400:0.75 500:0.75 600:0.75 600:none 1050:none
while [ $# -gt 0 ]; do
  for k in 1 2 3 4; do [ $# -gt 0 ] && { IFS=: read t p <<< "$1"; run $t $p & shift; }; done
  wait
done
echo "$(date) done"
