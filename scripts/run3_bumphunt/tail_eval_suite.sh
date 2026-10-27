#!/bin/bash
# Re-evaluate the 10% mass-window search with the enriched-tail QCD merged (held-out tail files), incl. P categories.
# (production finished)

S=/tmp/claude-12421/-scratch-llee/a35d08e7-9d87-5923-8644-0d53c5b0f767/scratchpad
ST=$(ls -d /home/llee/.claude/skills/synced/*/tufte-plots/assets | head -1)
PY=/scratch/llee/envs/pairnet/bin/python; C=/scratch/llee/pairnet/cuts_study; P=/scratch/llee/pairnet
E=/ospool/uc-shared/project/futurecolliders/llee/pairnet/eval_r14; T=/ospool/uc-shared/project/futurecolliders/llee/pairnet/tail
echo "$(date) start: $(ls $T/qcdvlmT_c*.npz | wc -l) vlmT, $(ls $T/qcdhtT_c*.npz | wc -l) htT"
run() {  # trig peakmax
  local Tr=$1 PK=$2; local O=$P/lowmass14win10t_pk$PK; mkdir -p $O
  local env="TAILDIR=$T CATS=3 CAT_PMIN=0.8 SEED_GRID=1 MODEL=r14A LMDIR=$E OLD_NPZ=$C/big_vars_r14A_pn.npz HT_NPZ_FILE=$C/big_vars_ht_r14A_ge1400.npz NO_LM=1 SIG_CLASSES=16,17,18,19 FTAG=f0.1 DUMP_SPECTRA=$O/dump"
  [ $PK != none ] && env="$env PEAK_MAX=$PK"
  env $env $PY $S/lowmass_study4.py $ST $O eval 0.1 $Tr > $O/eval_t$Tr.log 2>&1
}
for pk in 0.75 99; do TAILDIR=$T NEFF_MIN=20 COUNT_F=0.1 PEAK_MAX=$pk FTAG=_win10t_pk$pk $PY $S/fisher_high.py $C/ht_all.npy $P/xs_study/xs_study.json $C/big_vars.npz $C/big_vars_r14A_pn.npz $C/big_vars_ht_r14A_ge1400.npz $P/fisher_high > $P/fisher_high/run_win10t_pk$pk.log 2>&1 & done
run 450 0.75 & run 500 0.75 & run 600 0.75 & wait
run 450 none
echo "$(date) done"
