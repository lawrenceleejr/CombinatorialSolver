#!/bin/bash
# Mass-window width scan at 600-800 GeV with all targeted MC (10% on B, P categories, half-sample tuning).
S=/tmp/claude-12421/-scratch-llee/a35d08e7-9d87-5923-8644-0d53c5b0f767/scratchpad
ST=$(ls -d /home/llee/.claude/skills/synced/*/tufte-plots/assets | head -1)
PY=/scratch/llee/envs/pairnet/bin/python; C=/scratch/llee/pairnet/cuts_study; P=/scratch/llee/pairnet
E=/ospool/uc-shared/project/futurecolliders/llee/pairnet/eval_r14; T=/ospool/uc-shared/project/futurecolliders/llee/pairnet/tail; T7=/ospool/uc-shared/project/futurecolliders/llee/pairnet/t700
run() { local W=$1; local O=$P/winscan700/w${W/,/_}; mkdir -p $O
  env WIN=$W TAILDIR=$T TAIL700DIR=$T7 EXTRA_SIG_DIR=$T7 CATS=3 CAT_PMIN=0.8 PEAK_MAX=0.75 MODEL=r14A LMDIR=$E OLD_NPZ=$C/big_vars_r14A_pn.npz HT_NPZ_FILE=$C/big_vars_ht_r14A_ge1400.npz NO_LM=1 SIG_CLASSES=16,17,18,19 FTAG=f0.1 $PY $S/lowmass_study4.py $ST $O eval 0.1 450 650,700,750 > $O/eval.log 2>&1; }
for W in 0.92,1.02 0.93,1.01 0.94,1.04; do run $W & done; wait
for W in 0.9,1.02 0.95,1.03 0.91,1.03; do run $W & done; wait
echo "$(date) done"
