#!/bin/bash
# Samples for scripts/robustness.py: a different decay chain and +-10% shower alpha_s
# (Monash 0.1365). Runs JOBS generator processes at a time; finished files are skipped.
#   bash scripts/robustness_samples.sh [JOBS]
cd "$(dirname "$0")/.." && mkdir -p data/robust
J=${1:-4}
cat <<'LIST' | while read -r line; do
--process signal --mass 1000 --decay chain --squark-mass 500 --n 20000 --seed 301 --out data/robust/sig_m1000_chain500.h5
--process signal --mass 1000 --alphas-fsr 0.150 --alphas-isr 0.150 --n 20000 --seed 302 --out data/robust/sig_m1000_as150.h5
--process signal --mass 1000 --alphas-fsr 0.123 --alphas-isr 0.123 --n 20000 --seed 303 --out data/robust/sig_m1000_as123.h5
--process qcd --n 6000 --min-jets 6 --pthat-min 200 --alphas-fsr 0.150 --alphas-isr 0.150 --seed 311 --out data/robust/qcd_as150_a.h5
--process qcd --n 6000 --min-jets 6 --pthat-min 200 --alphas-fsr 0.150 --alphas-isr 0.150 --seed 312 --out data/robust/qcd_as150_b.h5
--process qcd --n 6000 --min-jets 6 --pthat-min 200 --alphas-fsr 0.123 --alphas-isr 0.123 --seed 313 --out data/robust/qcd_as123_a.h5
--process qcd --n 6000 --min-jets 6 --pthat-min 200 --alphas-fsr 0.123 --alphas-isr 0.123 --seed 314 --out data/robust/qcd_as123_b.h5
LIST
  out=${line##*--out }
  [ -e "$out" ] || echo "nice -n 19 python scripts/generate_pythia.py $line > $out.log 2>&1"
done | xargs -P "$J" -I{} bash -c "{}"
