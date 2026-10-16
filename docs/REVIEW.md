# Combinatorial Solver review (October 2026)

Goal: from jet four-vectors only, choose the interpretation that looks most like
pair production of two equal-mass resonances decaying to three jets each, with
high signal efficiency, and send QCD to high-asymmetry and low-mass
interpretations.

Everything below is measured on a Pythia 8 + FastJet testbed
(`scripts/generate_pythia.py`). Signal is g̃g̃ with g̃ → uds at 500, 750, 1000,
1250, 1500 and 2000 GeV, 20k events each. QCD is HardQCD with a parton shower,
p̂T > 200 GeV, with weights. Jets are anti-kT R = 0.4, pT > 30 GeV and
|η| < 3.0, with calorimeter-like pT smearing. The signal region is ≥ 6 jets.
Training uses signal `*_s1` and QCD s1–s7; every number here comes from the
held-out signal `*_s2` and QCD s8–s10. The 0.1 asymmetry cut stands in for the
analysis preselection.

## Findings about the current network

1. **The labels are a heuristic, not truth.** With the shipped config
   (`use_mass_asymmetry_labels: true`) both training phases fit
   argmin |m1 − m2|. On showered jets that label agrees with generator truth
   for only 11–15% of the events whose true pairing is among the leading 7
   jets, so about 88% of the labels are wrong. On the parton-level
   `gogoj_offshell_10k.h5` it agrees 40% of the time.
2. **Truth-label training could not run with a QCD sample.** QCD files have no
   TARGETS, so `JetAssignmentDataset` crashed, or would have dropped every QCD
   event as invalid. This is fixed on this branch in `src/dataset.py`:
   background events get a placeholder label.
3. **One softmax is asked to do two things.** It has to find the true pairing
   in signal and the most asymmetric pairing in QCD. Retrained on the testbed,
   it still passes 65% of QCD through A < 0.1 (81% for argmin |Δm|).
4. **The ceiling is set by physics** (`scripts/matching_breakdown.py`):
   - 39–67% of signal has all six quarks matched to jets.
   - Each failing event's first failure reason is one of: merged (≈19% of
     events), soft/FSR (34% → 9% across the mass range), or forward (≈4%).
   - About half of the merging is between quarks from *different* gluinos
     (11–15% of events), which no pairing can undo.
   - Only 25–47% of events have the true pairing among the leading 7 jets.

## The new network (`src/pairnet.py`, gated mode)

- **Inputs:** jet four-vectors only, in a canonical frame and scaled by HT.
  A Transformer biased by pairwise ln ΔR, ln kT, ln z and ln m².
- **Signal scorer:** scores all 70 interpretations.
- **Anti-signal choice:** a fixed function of the four-vectors, asym + low m1+m2.
- **Gate:** a learned P(signal) mixes the two.
- **Training targets:**
  - Truth for reconstructable signal.
  - For other signal, the most truth-consistent interpretation, ties broken by
    mass closest to M.
  - For QCD, the anti-signal choice.

## Results (held-out, 7 jets, A < 0.1)

| method | true pairing (all signal) | signal in [0.85, 1.1] M | QCD passes | QCD passes, m_avg > 800 |
|---|---|---|---|---|
| argmin \|m1 − m2\| | 4.3% | 37% | 81% | 7.8% |
| current network | 3.7% | 27% | 65% | 2.6% |
| current network, truth labels | 3.9% | 36% | 60% | 3.3% |
| **gated PairNet** | **8.3%** | **39%** | **8.5%** | **0.17%** |
| gated PairNet, 8 jets | 7.6% | 35% | 7.4% | 0.15% |

The current network was retrained with its shipped loss on a 30k-signal
subsample: d = 128, 4 layers, single phase, 6 epochs. That was the CPU budget,
so treat its numbers as indicative.

## Mass variable (`scripts/mass_variables.py`)

The figure of merit is S/√B in the shortest window holding 68% of the
preselected signal, compared against m_avg of the chosen pairing:

- **Dressed masses**, adding leftover jets within ΔR < 0.8: +0–15%.
- **Thrust-hemisphere masses**: about half the sensitivity.
- **HT window**: equal at low mass, +30–40% at 1.25–1.5 TeV.
- **m_avg window plus a loose HT floor** keeping 90% of signal: +20–60%.

## Notes for the background estimate

The analysis builds fake 6-jet events by replacing the 2-jet hemisphere of a
5-jet event with a compatible 3-jet one. A stronger gate could learn the
cross-hemisphere correlations that mixing removes, so re-run the closure with
the new network before adopting it. `src/hemimix.py` (`mix_replace`) and
`scripts/closure.py` implement that test.

## Figures

`docs/figures/*.pdf`: one panel per file, Tufte style, EB Garamond.

## Data scaling

Training on 3× the signal statistics (180k events, with QCD weighted 3× to keep
the balance) does not improve signal-in-window at matched QCD pass rates:

| QCD pass | 1× signal | 3× signal |
|---|---|---|
| 2% | 21% | 23% |
| 5% | 35% | 32% |
| 8% | 39% | 33% |

The 3× run is lower at 5% and 8%, probably because its saved checkpoint sits at
a different operating point. Either way, more statistics alone does not help at
this model size.

## Model capacity

A model 4.5× larger (d = 256, 6 layers, 3.3M parameters, 2× signal) matches
the default at its last epoch on validation: 8.7% versus 8.3% exact pairing,
42% versus 41% signal in window. It is no better on test at matched QCD rate.
Neither more data nor more capacity helps, so on this testbed the 7-jet gated
network appears close to what the four-vectors allow. The remaining losses are
physical: merging, FSR, and decay jets outside the leading 7.

Methodological note: checkpoints were selected on (exact pairing + QCD
rejection), so each run settles at a slightly different operating point. For
production, select the checkpoint at a fixed QCD pass rate, or choose the
operating point afterwards with the gate threshold.

## How best to find the signal from four-vectors (follow-up study)

`scripts/search_study.py` reduces every strategy to an event score and a mass,
then takes S/√B with the best score cut and mass window at each mass. All are
evaluated on the same held-out events: 60k signal and 18k QCD (relative units).

| strategy | 500 | 750 | 1000 | 1250 | 1500 | 2000 GeV |
|---|---|---|---|---|---|---|
| argmin \|Δm\| + asym cut | 1.2 | 1.5 | 2.1 | 3.2 | 4.7 | 10 |
| current network + asym cut | 1.2 | 1.4 | 2.2 | 3.3 | 4.6 | 9 |
| gated PairNet + asym cut | 2.1 | 5.4 | 10 | 20 | 29 | 143 |
| expert-feature BDT + argmin pairing | 3.1 | 7.7 | 17 | 36 | 58 | 129 |
| **gated PairNet, cut on P(signal)** | **3.2** | **9.7** | **22** | **39** | **70** | **150** |
| same, true pairing (bound) | 3.3 | 10 | 26 | 49 | 86 | 176 |

- **Selection is the lever.** Cutting on the event-level P(signal) beats any
  mass-asymmetry cut by 3× at 500 GeV, 7× at 750 GeV and 10–16× from 1 TeV.
- **Pairing headroom is small.** The true pairing would add at most ~25%.
- **The peak is broadened by pairing, not by the jets.** With the true pairing
  the core half-width is 6.5–7.4%, against 17–25% with the chosen pairing. A
  learned M/HT estimator (`scripts/mass_estimators.py`) narrows it by 13–22%.
- **No sculpting.** The gate cut's QCD efficiency rises smoothly with mass
  (`docs/figures/sculpting_gate_argmax.pdf`).
- **Jet multiplicity.** The expert BDT's top feature is the jet count. The gate
  keeps an AUC of 0.952–0.957 within fixed multiplicity bins (0.974 overall), so
  most of its power is kinematic. Jet-multiplicity modelling should be checked
  in data.
- **Hemisphere mixing.** Mixed events have exactly 6 jets while the gate uses a
  7th, so use a 6-jet network or mix within jet-multiplicity bins.
- **Negative results:**
  - A parton-level prefilter for QCD generation gives no speed-up.
  - More signal statistics (3×) did not help at this scale.
  - A 4.5× larger model did not help at this scale.

The points at 1 TeV and above rest on few QCD MC events. More QCD, v2 (gate
sees the pairing summary and encoder sees 12 jets) and a 6-jet variant are
being added.

### Final comparison (30k held-out QCD) and the exported network

`results/search_study_final`. AUC is signal vs weighted QCD; S/√B uses one
P(signal) cut for all masses.

| P(signal) from | AUC: 6 / 7 / ≥8 jets / all | S/√B: 500 / 750 / 1000 / 1250 / 1500 / 2000 GeV |
|---|---|---|
| **gated PairNet, 7 jets** | 0.951 / 0.956 / 0.956 / 0.974 | 2.9 / 8.3 / 18 / 42 / 59 / 149 |
| v2: gate sees the pairings, 12-jet context | 0.948 / 0.954 / 0.968 / 0.977 | 3.4 / 12.5 / 21 / 39 / 55 / 161 |
| ensemble of 4 networks | 0.951 / 0.956 / 0.962 / 0.976 | 3.0 / 8.9 / 23 / 52 / 61 / 151 |
| stacked: expert features + PairNet | 0.952 / 0.958 / 0.967 / 0.978 | 3.4 / 9.7 / 21 / 44 / 66 / 167 |
| gated PairNet, 6 jets | 0.945 / 0.946 / 0.949 / 0.962 | 2.5 / 7.5 / 17 / 46 / 63 / 125 |
| current network, asymmetry cut | – / – / – / 0.697 | 0.6 / 0.9 / 1.2 / 2.1 / 3.1 / 7.1 |

- **The extra gain comes from ≥ 8-jet events.** Beyond the 7-jet gated network,
  v2, the ensemble and stacking gain only where there are 8 or more jets; within
  6 and 7 jets all options are equal. That gain rides on extra-radiation jets,
  which are QCD-modelling sensitive, so the 7-jet gated network is the
  recommended, robust choice.
- **ONNX export.** It is `models/pairnet_gated_7jet.onnx`, a drop-in for
  `ml_model.onnx`: same input and same 70-assignment order, with `p_signal`,
  `signal_logits` and `masses` added. See `models/README.md` for the interface
  and working points.
- **Pairing headroom.** With more QCD statistics, the true pairing adds only
  ~10% below 2 TeV.
- **Why exact pairing is low.** It is near the information limit
  (`scripts/pairing_ambiguity.py`). An oracle that knows the gluino mass finds
  the true pairing in only 27–34% of available events, the same as the network.

## Follow-up: truth definition, one-network mass, robustness

### Which quark defines the true jets

The files in `data/pythia` match jets to the decay quark traced to the end of
its own shower (Pythia `iBotCopyId`, like CMS `isLastCopy`). After a hard
emission that quark can point away from the jet that carries most of its
energy. The alternative is the quark before its own shower (Pythia `iBotCopy`,
like CMS `isLastCopyBeforeFSR`). Recoil is handled either way: the gluino's
daughters and their last carbon copy match identically.

`scripts/truth_quarks_generator.py`, 1500 events per mass:

| | 500 GeV | 1 TeV | 2 TeV |
|---|---|---|---|
| all six matched, before / after own FSR | 39% / 41% | 49% / 56% | 54% / 67% |
| matched-triplet mass, 16–84%: before own FSR | 0.84–1.00 M | 0.83–1.00 M | 0.83–1.00 M |
| matched-triplet mass, 16–84%: after own FSR | 0.80–1.00 M | 0.76–0.99 M | 0.75–0.99 M |
| events matched both ways whose true pairing differs | 14% | 20% | 23% |
| where it differs, triplets within 0.95 ± 0.1 M: before / after | 53% / 44% | 55% / 40% | 60% / 47% |

The before-FSR definition is the cleaner label. Scored against each definition
(`scripts/truth_definition.py`, 10k held-out events per mass, truth among the
leading 7 jets; network trained on after-FSR labels):

| | 500 GeV | 750 GeV | 1 TeV | 2 TeV |
|---|---|---|---|---|
| truth available, before / after | 25.0 / 24.7% | 30.5 / 31.6% | 34.1 / 35.5% | 41.8 / 46.5% |
| true-pairing m_avg core, before / after | ±6.5 / ±7.4% | ±6.1 / ±7.2% | ±5.6 / ±6.6% | ±5.1 / ±6.5% |
| gated PairNet exact pairing, before / after | 28.8 / 26.3% | 29.4 / 26.6% | 29.7 / 26.8% | 28.5 / 24.3% |
| mass oracle, before / after | 31.0 / 29.0% | 29.4 / 26.7% | 31.8 / 29.3% | 36.9 / 32.0% |

So the network's pairing is 2.5–4 points better than the after-FSR truth
said, and still close to the oracle except at 2 TeV, the top of the training
mass grid (an edge effect that a wider grid, as in `docs/PRODUCTION.md`,
should remove).

Retraining on before-FSR labels (same events, architecture and schedule; the
mass-head network below), scored on before-FSR truth:

| exact pairing, of available | 500 GeV | 750 GeV | 1 TeV | 2 TeV |
|---|---|---|---|---|
| trained on after-FSR labels | 26.4% | 28.1% | 29.9% | 29.9% |
| trained on before-FSR labels | 27.3% | 28.5% | 29.6% | 30.4% |

The pairing moves by under a point, i.e. within run-to-run noise: the network
had absorbed the label noise. The cleaner definition mainly changes what is
measured (+2.5–4 points, and a narrower true-pairing peak). It is still the
right label for production.

`generate_pythia.py` now stores both (`truth/quarks_before_fsr`,
`truth/quarks_after_fsr`); `--truth-quarks` picks the one used for `TARGETS`,
and `produce_samples.py` defaults to `before_fsr`. `train_pairnet.py
--truth-key` chooses the labels at training time.

### One network for the pairing, P(signal) and the mass

`--gate-interp --mass-head` lets the gate see the pairing summary and adds a
head that learns log(M/HT), so one forward pass gives the pairing, P(signal)
and a resonance-mass estimate (`m_est`). Same data and schedule as before,
held-out test:

| | AUC | QCD rejection at 50% signal | exact pairing | signal in window at 2 / 5 / 8% QCD pass |
|---|---|---|---|---|
| gated PairNet, 7 jets | 0.966 | 219 | 8.3% | – |
| + gate sees pairings + mass head | 0.965 | 198 | 8.1% | 21.7 / 35.3 / 38.8% |
| gated PairNet, 6 jets | 0.956 | 133 | 7.4% | 14.0 / 28.0 / 32.5% |
| + gate sees pairings + mass head | 0.956 | 146 | 7.5% | 13.9 / 27.9 / 32.6% |

Selection and pairing are unchanged within run-to-run noise. The learned mass
matches the separate BDT regressor of `scripts/mass_estimators.py` (core
half-width of the shortest 68% window, all signal):

| | 500 | 750 | 1000 | 1250 | 1500 | 2000 GeV |
|---|---|---|---|---|---|---|
| m_avg of the signal scorer's pairing | 25.3% | 21.8% | 20.7% | 19.1% | 17.9% | 17.0% |
| separate BDT regressor | 22.0% | 19.4% | 17.5% | 15.9% | 14.7% | 13.3% |
| **mass head, 7 jets** | 23.2% | 19.7% | 17.5% | 16.1% | 14.7% | 12.9% |
| mass head, 6 jets | 23.0% | 19.6% | 18.6% | 16.8% | 15.2% | 13.5% |

`export_pairnet_onnx.py` adds the `m_est` output for such checkpoints.

The recommended model was trained with (same events as `data/pythia`, before-FSR
labels from `scripts/regen_truth_before_fsr.sh`):

```bash
python scripts/train_pairnet.py --gated --proxy --gate-interp --mass-head --n-jets 7 \
    --sig data/pythia_bf/sig_m{500,750,1000,1250,1500,2000}_s1.h5 --bkg data/pythia/qcd_s{1..7}.h5 \
    --test-sig data/pythia/sig_m*_s2.h5 --test-bkg data/pythia/qcd_s8.h5 \
    --epochs 20 --threads 2 --select combined --out runs/g7_gi_mh_bf
python scripts/export_pairnet_onnx.py --model runs/g7_gi_mh_bf/model.pt --out models/pairnet_gated_7jet_mass.onnx
```

With the new 7-jet network trained on before-FSR labels (exported as
`models/pairnet_gated_7jet_mass.onnx`, now the recommended model), on the
search study's held-out events (`scripts/search_study.py`, one P(signal) cut
for all masses):

| | AUC 6 / 7 / ≥8 jets / all | S/√B: 500 / 750 / 1000 / 1250 / 1500 / 2000 GeV |
|---|---|---|
| gated PairNet, 7 jets, m_avg | 0.951 / 0.956 / 0.956 / 0.974 | 2.9 / 8.3 / 17.7 / 42.0 / 58.6 / 149 |
| gated PairNet + separate mass regressor | same | 2.8 / 8.9 / 22.1 / 43.1 / 64.9 / 190 |
| **new network, its own mass** | **0.955 / 0.958 / 0.958 / 0.975** | **3.0 / 9.8 / 25.7 / 39.9 / 68.0 / 199** |
| new network, m_avg | same | 2.8 / 9.0 / 20.2 / 57.5 / 60.6 / 152 |
| 6 jets, same additions, its own mass | 0.946 / 0.947 / 0.948 / 0.962 | 2.3 / 8.2 / 23.2 / 37.1 / 65.9 / 187 |

Selection is equal or slightly better in every jet bin, and the learned mass
lifts S/√B by 16–45% at 0.75, 1, 1.5 and 2 TeV (level at 1.25 TeV). Above
1 TeV the windows hold only 1–13 effective QCD MC events, so differences of
tens of percent there are not significant. At 1% QCD the new network keeps
33 / 56 / 71 / 80 / 86 / 91% of signal (500–2000 GeV), 1–2 points more than
before; `models/README.md` lists the working points.

### Modelling robustness

`scripts/robustness.py` fixes each method's cut on the nominal held-out QCD
and applies it to varied samples (`scripts/robustness_samples.sh`): 1 TeV
signal with g̃ → q̃q, q̃ → qq (m_q̃ = 500 GeV), and signal and QCD with both
shower α_s (ISR and FSR) at 0.150 and 0.123 instead of Monash 0.1365.
Detector-level variations are applied to the nominal events. Efficiency
relative to nominal at the 1% nominal-QCD working point:

| | decay chain | signal α_s +10% / −10% | QCD α_s +10% / −10% | signal: JES −3% / fwd −5% / +5% smear | QCD: JES −3% / fwd −5% / +5% smear |
|---|---|---|---|---|---|
| gated PairNet, 7 jets | 1.11 | 0.83 / 1.11 | 0.40 / 1.32 | 1.00 / 1.00 / 1.00 | 1.03 / 1.06 / 1.02 |
| new network (mass head) | 1.11 | 0.85 / 1.12 | 0.47 / 1.27 | 1.00 / 1.00 / 1.00 | 1.03 / 1.07 / 1.04 |
| v2 (12-jet context) | 1.08 | 0.94 / 1.05 | 0.57 / 1.28 | 0.99 / 1.00 / 0.99 | 1.01 / 1.01 / 0.97 |
| gated PairNet, 6 jets | 1.14 | 0.77 / 1.17 | 0.38 / 1.49 | 1.00 / 1.00 / 1.00 | 1.04 / 1.05 / 1.02 |
| expert BDT | 1.10 | 1.02 / 0.97 | 0.76 / 0.75 | 0.98 / 1.00 / 0.96 | 0.96 / 1.00 / 0.85 |

Statistical precision: ±0.01–0.02 for signal; ±0.07–0.17 for QCD (6k events
per varied QCD sample).

- **Decay chain:** selected 8–14% more often, peak unchanged. No loss for this
  signal model.
- **Detector:** the networks' signal efficiency moves by ≤ 1%, QCD within
  MC precision. P(signal) is exactly invariant under a global jet energy
  scale (the network divides by HT); only jets crossing 30 GeV break it. The
  peak follows the scale (0.961 → 0.932 M at −3%). The BDT loses 4% of
  signal with extra smearing at 1% QCD (8% at 0.1%).
- **Shower α_s: the networks' P(signal) depends on the radiation.** More
  radiation makes both signal and QCD look less signal-like (−15–17% signal,
  −53–60% QCD at 1% for the 7-jet networks); less radiation the opposite.
  The BDT's signal efficiency moves by ≤ 3%. The likely lever is the jet mass:
  QCD jets are heavier than signal jets (⟨m/pT⟩ of the leading six 0.164 vs
  0.118), and ±10% α_s moves it by about ±0.01 while the jet count barely
  changes. The four-vector carries the mass, so the network can use it as a
  quark-versus-gluon handle.
- **What it means:** hemisphere mixing takes real jets from data, so the
  background estimate inherits the data's response. The signal efficiency
  needs a shower systematic (here 11–17% for ±10% α_s at 1% QCD, larger at
  tighter cuts). To reduce it, train on a mix of shower settings or with
  massless four-vectors and measure what that costs; both are runs for the
  GPU machine.
- **Test: the same network with massless jets** (`train_pairnet.py --massless`,
  6 epochs). The shower sensitivity disappears: signal 1.02 / 1.02 at 1% QCD
  for α_s ±10% (against 0.85 / 1.12 with masses); QCD 0.89 ± 0.16 /
  0.68 ± 0.14 (against 0.47 / 1.27). So the jet mass is the lever. But the
  massless network keeps only 42% of 1 TeV signal at 1% QCD, against 71% with
  masses (17% against 37% at 0.1%), and its validation AUC levels off at 0.944
  against 0.964 at the same epoch.
- **Recommendation:** the jet mass is worth about ×1.7 in signal efficiency at
  a fixed QCD rate, far more than its 11–17% shower systematic. Keep it (it is
  part of the four-vector, though it acts as a quark/gluon handle), check the
  jet-mass modelling in data, for example in the 5-jet control region, and
  reduce the dependence by training on a mix of shower settings.
- Weighted QCD acceptances are not compared between generator runs: the
  bias-weight normalisation is not consistent across runs.

Figures: `docs/figures/robustness_physics.pdf`, `docs/figures/robustness_detector.pdf`,
`docs/figures/truth_definition.pdf`.

### Is the pairing near its limit? (follow-up)

Exact pairing is ~30% of the events whose true pairing is in the leading 7
jets. Four independent tests (before-FSR truth, held-out events) say this is
close to what the four-vectors allow:

| test (of available events) | 500 GeV | 1 TeV | 2 TeV |
|---|---|---|---|
| recommended network (gate + mass head) | 27.3% | 28.9% | 30.4% |
| pairing-only network, no background or gate | 28.1% | 28.3% | 30.1% |
| same, 3× signal statistics | plateaus at 30% on validation, same as 1× | | |
| gradient-boosted ranker on explicit per-pairing features | 27.0% | 27.9% | 28.7% |
| mass oracle that knows M, targets 0.94 M | 37.3% | 33.2% | 35.5% |

- Training-set accuracy equals test accuracy (29%), so it is not overfitting,
  and 3× data does not help: not a statistics limit.
- Only 44–58% of "available" truths have both triplets within ±8% of 0.94 M;
  the rest lost energy to FSR or a merged jet, and are found 12–16% of the
  time. On the mass-consistent ones the network gets 62–66% with exactly 6
  jets and 37–41% with a 7th jet.
- Half the misses choose the wrong jet to leave out (ISR); most others move a
  soft jet (~7% of HT) between triplets, which barely changes the masses.
- The only method above ~30% knows the true mass. A pairing conditioned on the
  mass hypothesis being tested (natural in a bump hunt) could gain up to
  4–9 points; nothing else tested does.
