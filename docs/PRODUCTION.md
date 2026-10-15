# Producing samples and training on a GPU

## 1. Install the generators (on the CPU machine that makes events)

```bash
pip install -r requirements.txt
JOBS=32 scripts/install_generators.sh   # conda-forge if conda exists, else a parallel build of pythia8mc
```

## 2. Produce samples

```bash
# signal: 21 masses 400-2400 GeV x 100k events, QCD: 1M kept events (>= 5 jets)
python scripts/produce_samples.py --outdir data/prod \
    --masses 400:2400:100 --sig-events 100000 --qcd-events 1000000 --cores 64
```

- Files hold 10k events each: `sig_m{M}_c{i}.h5`, `qcd_c{i}.h5`. Chunk `i`
  decides the split: `i % 10` in 0–7 is train, 8 is validation, 9 is test.
- The driver is resumable: finished files are skipped. Every file has its own
  fixed seed, and `--seed-offset` makes an independent production.
- For Condor or Slurm, add `--joblist jobs.txt` to get one command per line,
  then run `parallel -j 64 < jobs.txt` or submit each line as a job.
- Defaults match the review's testbed:
  - jets: anti-kT R = 0.4, pT > 30 GeV, |η| < 3, calorimeter-like smearing;
  - signal: g̃ → uds, width 1% of the mass;
  - QCD: p̂T > 200 GeV with a p̂T² bias (weights stored).

Measured throughput per core:
- signal: ~55 events/s, about 5 core-hours per million events;
- QCD: ~9 kept events/s, about 30 core-hours per million. Roughly 40% of kept
  QCD events have ≥ 6 jets, i.e. are in the signal region.

The example above is about 210 core-hours of signal and 30 of QCD, so about 4
hours on 64 cores. QCD is the expensive sample per useful event; produce more
of it if the high-mass tail is statistics-limited (the review hit that above
1.5 TeV).

## 3. Train on the GPU machine

```bash
python scripts/train_pairnet.py --gated --proxy --n-jets 7 \
    --sig "data/prod/sig_*_c[0-7].h5"  --bkg "data/prod/qcd_c[0-7].h5" \
    --val-sig "data/prod/sig_*_c8.h5"  --val-bkg "data/prod/qcd_c8.h5" \
    --test-sig "data/prod/sig_*_c9.h5" --test-bkg "data/prod/qcd_c9.h5" \
    --bkg-weight 3 --epochs 40 --bs 1024 --lr 2e-3 --amp \
    --cache-dir cache --out runs/gated_n7
```

- `--device auto` picks CUDA, then Apple MPS, then CPU. The whole dataset is
  kept on the device; 2M events × 7 jets is a few hundred MB.
- `--amp` turns on bfloat16 autocast (CUDA only).
- `--cache-dir` stores the parsed arrays, so later runs skip the h5 parsing.
- `--resume` continues from `<out>/last.pt`, which is written every epoch.
- `--select sig_at_bkg --select-bkg-pass 0.05` (the default) keeps the epoch
  with the most signal in the mass window at 5% weighted QCD pass of the
  asymmetry cut. That puts every run at the same operating point.
- `--bkg-weight` balances QCD against signal in the loss. Raise it when signal
  outnumbers QCD.
- With a larger batch, scale the learning rate roughly as √(batch / 256) from 1e-3.

Outputs: `model.pt`, `last.pt`, `metrics.json` (per-epoch validation and the
final test) and `eval.npz` (per-event test predictions). For figures:

```bash
python scripts/make_plots.py --runs "PairNet=runs/gated_n7/eval.npz" --out plots/gpu
```

## 4. Things worth trying with GPU budget

- A larger model (`--d-model 256 --layers 6`). On CPU at about 200k events it
  did not beat the default, but it was never trained to convergence.
- `--n-jets 8` with more epochs. The truth ceiling is 10 points higher than
  with 7 jets, but at the CPU budget it did not pay off.
- A continuous mass spectrum, via a finer `--masses` grid, to keep the network
  mass-agnostic.
