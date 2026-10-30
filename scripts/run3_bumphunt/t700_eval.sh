#!/bin/bash
# 600-800 GeV mass-window search, 10% on B, P categories: (a) existing QCD + extra signal, (b) + targeted 700-GeV QCD.
# usage: t700_eval.sh TAG   (output dirs lowmass14t700{a,b}_TAG)
TAG=$1
S=/tmp/claude-12421/-scratch-llee/a35d08e7-9d87-5923-8644-0d53c5b0f767/scratchpad
ST=$(ls -d /home/llee/.claude/skills/synced/*/tufte-plots/assets | head -1)
PY=/scratch/llee/envs/pairnet/bin/python; C=/scratch/llee/pairnet/cuts_study; P=/scratch/llee/pairnet
E=/ospool/uc-shared/project/futurecolliders/llee/pairnet/eval_r14; T=/ospool/uc-shared/project/futurecolliders/llee/pairnet/tail; T7=/ospool/uc-shared/project/futurecolliders/llee/pairnet/t700
echo "$(date) $TAG: $(ls $T7/qcd700T_c[0-9]*.npz 2>/dev/null | wc -l) qcd700T, $(ls $T7/sig*X_*_c*.npz 2>/dev/null | wc -l) signal files"
run() {  # variant trig
  local V=$1 Tr=$2; local O=$P/lowmass14t700${V}_$TAG; mkdir -p $O
  local env="TAILDIR=$T EXTRA_SIG_DIR=$T7 CATS=3 CAT_PMIN=0.8 SEED_GRID=1 PEAK_MAX=0.75 MODEL=r14A LMDIR=$E OLD_NPZ=$C/big_vars_r14A_pn.npz HT_NPZ_FILE=$C/big_vars_ht_r14A_ge1400.npz NO_LM=1 SIG_CLASSES=16,17,18,19 FTAG=f0.1 DUMP_SPECTRA=$O/dump"
  [ $V = b ] && env="$env TAIL700DIR=$T7"
  [ $V = a ] && env="$env TAIL700DIR=/nonexistent"
  env $env $PY $S/lowmass_study4.py $ST $O eval 0.1 $Tr 600,650,700,750,800 > $O/eval_t$Tr.log 2>&1
}
for V in a b; do for Tr in 450 500 600; do run $V $Tr & done; wait; done
echo "$(date) $TAG done"
