#!/bin/bash
# Re-generate the nominal signal files with the SAME seeds (identical events)
# but jets matched to the decay quarks before their own FSR (--truth-quarks
# before_fsr); both truth definitions are stored. s1 = training, s2 = test.
#   bash scripts/regen_truth_before_fsr.sh [JOBS]
cd "$(dirname "$0")/.." && mkdir -p data/pythia_bf
J=${1:-4}
for s in 1 2; do for M in 500 750 1000 1250 1500 2000; do
  out=data/pythia_bf/sig_m${M}_s${s}.h5
  [ -e "$out" ] || echo "nice -n 5 python scripts/generate_pythia.py --process signal --mass $M --n 10000 --seed $((M + s)) --truth-quarks before_fsr --out $out > $out.log 2>&1"
done; done | xargs -P "$J" -I{} bash -c "{}"
