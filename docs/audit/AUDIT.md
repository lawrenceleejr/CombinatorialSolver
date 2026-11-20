# Audit of the PairNet r14A search (Nov 2026)

Question: with only jet four-vectors, how well do we put the signal into a mass peak and push QCD away, and
how much of the projected sensitivity survives scrutiny?  Seven pieces (F0–F6, reports in the sub-directories)
plus a new-network attempt (F7, r16J).  Factors > 1 mean the published number is optimistic.

## 1. Bugs and corrections to the published projection

| # | Finding | Effect |
|---|---|---|
| 1 | Falling-QCD-spectrum (PEAK_MAX) requirement checked on the tuning half only (`lowmass_study4.py`) | ×1.13 (400), ×1.08 (700), ×1.03 (1000) |
| 2 | 10 % on B treated as uncorrelated between P categories; neither uncorrelated nor fully correlated is conservative (worst ρ ≈ 0.5–0.7) | categories headline ×1.14–1.34; with worst ρ the single category is as good at 700 and 1500 GeV |
| 3 | Best-of configuration choice made on evaluated limits | ×1.00–1.05 |
| 4 | QCD MC statistics after the final cuts: 11–48 effective events (8–15 at 1.9–2.3 TeV), single events carry up to 1/3 of B | ±12–20 % per limit (not a bias) |
| 5 | `window10.py` docstring says "fully correlated", headline uses uncorrelated; `fisher_high.py` KeyError for other tails; suite configs point to deleted /tmp paths | documentation / reproducibility |
| 6 | ONNX contract: <6 jets → NaN or P=1; pT=0,pz≠0 jet → NaN; input must be pT-sorted (unchecked; 15 % pairing change if not) | not hit by the current pipeline |

Corrected expected limits (fb, 300 fb⁻¹, f = 10 %), r14A + current recipe:

| M (GeV) | published single | corrected single | published cats | corrected cats |
|---|---|---|---|---|
| 400 | 3103 | 4030 | 2486 | 4170 |
| 700 | 151.5 | 168 | 117.5 | 168 |
| 1000 | 30.3 | 32.0 | 21.4 | 29.1 |
| 1500 | 7.69 | 8.09 | 6.41 | 8.09 |
| 2000 | 1.18 | 1.20 | 1.02 | 1.17 |

Reach: single 2123 → ~2113 GeV (MC spread 2101–2215); categories 2197 → 2129 GeV.

Checked and fine: limit formula (on/off Asimov, within 2–7 % of toys), no train/test leakage (held-out vs
training tail efficiency 1.0 ± 0.06), no duplicate events (209 families), QCD sample merge, signal
normalisation, generator filters (H_T filter exact, symmetry filter < 0.6 % loss, pThat slices agree),
ONNX = PyTorch, exact scale/η symmetries.

## 2. Context the projection needs

* **CMS Run-2 expected limits are better than this 300 fb⁻¹ projection below ~1.7 TeV** (700: 93 vs 152 fb;
  1000: 12.5 vs 30.3 fb).  PairNet wins only at ≥ 1.7–2 TeV (2000: 1.18 vs 3.38).  Our selections end at
  S/B ≈ 0.2, so the flat 10 % on B dominates; the background-estimate uncertainty matters as much as the network.
* QCD MC vs data (emulated CMS-EXO-21-004 selection with a q/g discriminator; full F2 sample with low-pThat slices):
  MC/data 1.5–1.8 (region 1), 0.58–0.66 (region 2), 0.56–0.89 (region 3), errors 10–40 %; pThat < 120 GeV slices
  contribute ≤ 1 % of each region (pThat < 80 GeV: nothing).
* Signal-model dependence: FSR α_s ±0.005 → ±35 % efficiency at M ≤ 1 TeV (±15 % at 1.5, ±5 % at 2 TeV); cascade
  decays ×0.43–2.05; width 10 % ×0.72.  The tight P cut uses the radiation pattern; the limit holds for the flat
  3-body benchmark only.

**Shower / tune dependence of the QCD background (F2 OSPool samples, full statistics, Nov 14).**  QCD with
A < 0.1 and m_avg in [0.85, 1.1] M, ratio to the production (Pythia default) sample with the same generator filter:

| sample | P > 0.9 | P > 0.99 | final P cut (0.9966 @700, 0.998 @1500) |
|---|---|---|---|
| default regenerated (closure), 700 GeV | 1.01 ± 0.04 | 0.79 ± 0.12 | 1.00 ± 0.55 |
| CP5-like tune, 700 GeV | 1.14 ± 0.03 | 1.51 ± 0.23 | 0.74 ± 0.34 |
| CP5-like tune, 1500 GeV | 1.05 ± 0.04 | 0.99 ± 0.08 | 1.10 ± 0.23 |
| **Vincia shower, 700 GeV** | 0.41 ± 0.01 | 0.27 ± 0.04 | **0.17 ± 0.08** |
| **Vincia shower, 1500 GeV** | 0.42 ± 0.02 | 0.43 ± 0.04 | **0.19 ± 0.07** |

The shower model moves the final-cut QCD by a factor ~5 (the tune by ≲ 1.5): the MC prediction of B at the working
point is uncertain at that level, and the tight P cut is where the two showers differ most.  This is the
largest systematic found in the audit; the signal efficiency under Vincia was not generated, so the net effect on
the limit is unknown.  It reinforces that B must come from data (the 10 % assumption is about that estimate, not
the MC).  Signal decay-model dependence at the final cut (full statistics): 700 GeV width 10 % 0.80 ± 0.06,
cascade via squark at 0.5 M 2.09 ± 0.13, at 0.85 M 0.44 ± 0.04, udb 1.03 ± 0.07; 1500 GeV 0.79, 1.66, 0.62, 1.00.

## 3. Robustness of r14A (limit ratio after re-tuning the P cut)

| Effect | Ratio |
|---|---|
| One extra soft (0.001–10 GeV) jet | QCD pass at P ≥ 0.997: 0.2 % → 44–57 % (not IR safe) |
| One extra 30 GeV jet | QCD ×3.8, signal ×4.1 pass rate |
| 20 % jet-mass smear + 10 % JER | ×1.50 (700), ×1.34–1.51 (1–2 TeV) |
| All jet masses zero | ×1.51 (700), ×1.7 (400), ×4.4 (1.5 TeV), ×7.5 (2 TeV) |
| JER +10 % / threshold 50 GeV / mass scale 0.9 | ×1.47 / ×1.43 / ×1.41 (700) |
| +2 pile-up jets | ×1.14–1.31 (up to ×1.7 at fixed cuts) |
| η/φ resolution, |η| < 2.4, 5 % jet loss | ≤ ×1.15 |

The P cut is in effect calibrated on the number of jets above threshold and on particle-level jet masses; it
must be re-derived on data, and a resolution-aware training is needed before the jet-mass gain can be claimed.

## 4. Where the performance is lost (peak)

* Only 39 % of P-selected signal is in [0.85, 1.1] M (23 % in [0.92, 1.04] M); r14A's peak 68 % half-width at
  1 TeV is 0.205 of M, of which 9.0 % of events are correct groupings and 25.7 % wrong-but-right-mass.
* With ancestry truth (each jet labelled parent 1 / parent 2 / other) and free group sizes over the leading
  12 jets, the narrow-window ceiling is 34–73 % of produced events (vs r14A 9–36 %), half-width 0.04 M.
  Losses: jet pT > 30 GeV at low mass (34 % of 400 GeV events have a decay quark below 30 GeV), forcing 3+3
  (15–22 points).  Free grouping also lets ×1.6–2.3 more QCD offer a symmetric pair.
* r14A is under-trained (validation loss still falling at the last epoch, no train–validation gap) and its
  checkpoint selection never looks at the gate.  The r1–r15 sweep was never seed-repeated; only the window term
  and "tail QCD hurts" are significant.

## 5. Improvements realised

1. **Pooled, mass-parametrised stage-2 MLP on r14A outputs** (16 features in units of m_avg: logit P, H_T/m,
   boost/Dalitz terms, A, unused-jet activity, n_jets, triplet pTs) — strictly nested.  Single category vs single
   category: **limits ×0.66 (geometric mean over 25 masses)**, 0.36–0.92 per mass; with 3 categories 0.67 (so the
   gain does not rely on the category treatment).  Examples (single category, f = 10 %): 700 GeV 178 → 131 fb,
   1 TeV 30.9 → 20.6 fb, 1.5 TeV 5.97 → 3.27 fb, 2 TeV 1.13 → 0.77 fb.  Caveat: relies on QCD-MC modelling of
   H_T/m and event shapes.
2. Not useful: hypothesis-conditioned pairing (sculpts QCD into the window), m_est as search variable,
   pairing-distribution variables; window-edge re-tune only 5–8 %.

## 6. New network attempt (F7, r16J): free-size groups from per-jet ancestry

Ancestry-labelled signal generated at 29 masses (1.3 M events, `F7_jetassign/samples`, code on branch
`audit/f7-jetassign`).  Two decoders: per-jet parent labels, and a pairwise same-parent head with an exact best
2-colouring.  Head-to-head with r14A on identical test events (93 k signal at 0.4–2 TeV, 1.07 M held-out QCD),
signal kept in the window at equal weighted QCD (1 % of r14A's window QCD):

| M | r16Jp (pair head, d128, epoch 10/20) / r14A | QCD in window with no P cut, r16Jp / r14A |
|---|---|---|
| 400 | 0.65 | ×5.5 |
| 700 | 0.53 | ×8.3 |
| 1000 | 0.58 | ×9.2 |
| 1500 | 0.48 | ×7.5 |
| 2000 | 0.69 | ×4.8 |

Final r16Jp (20 epochs), same comparison: 0.70 / 0.64 / 0.62 / 0.87 / 0.74 at 400 / 700 / 1000 / 1500 / 2000 GeV.
Per-jet labels are worse still (≤16 % of signal in the window at any size).  The free-size grouping puts as much
signal in the window as r14A (no P cut: 0.29 vs 0.31 at 700 GeV) but lets 5–9× more QCD form a symmetric
in-window pair, and the gate does not win it back.  The ancestry ceiling assumes a perfect assignment; with a
learned one, extra flexibility mostly manufactures QCD fakes.  **Not pursued further**; r14A's fixed 3+3 choice
plus the anti-signal mixture is the better trade at this level of assignment accuracy.

## 7. Why the r14A loss is flat after one epoch (F8)

Decomposition on 80 k held-out events (`F8_saturation/decompose.py`):
* One epoch = 17 k optimizer steps; validation loss keeps falling slowly (still falling at the last epoch).
* QCD mixture term: 0.335, of which 0.22 is an unlearnable floor (fixed anti-signal softmax; 26 % of QCD have no
  clear anti-signal choice).
* The window reward (weight 5) pays for probability on *any* symmetric in-window pairing: 85.5 % of signal events
  have at least one (median 5).  The network spreads its probability: on exact events P(true pairing) is 0.21
  on average (median 0.075) while 0.56 sits on in-window pairings; proxy targets cost 4.77 > log 70.
* No train–validation gap → not data-limited for this objective.  Capacity (width/depth/structure) was tested once
  (r9A d128 L4 vs r9B d256 L6 at 10 jets: identical).  Now queued on the r14A recipe at 12 jets: interpretation
  attention (`--interp-attn 2`), d256 L6/L8, d256 L6 ff×4 + attention, d384 L8 ff×4 (F5 worktree options).

## 8. Capacity, length and seed tests on the r14A recipe (F5, harvested Nov 14)

Mean signal efficiency in window over 400/700/1000/1500/2000 GeV (×100) at fixed QCD efficiency; difference to
r14A with paired bootstrap error (full table F5_train/TABLE.md, plots overtrain_*/eff_vs_epoch_*).

| run | 0.5 % QCD | 0.1 % QCD | 0.02 % QCD | val-loss gap (train−val) |
|---|---|---|---|---|
| r14A | 30.28 | 22.41 | 15.85 | 0.004 |
| seed 1 | +0.19 ± 0.07 | +0.35 ± 0.14 | 0.00 ± 0.17 | 0.004 |
| seed 2 | +0.07 ± 0.07 | +0.30 ± 0.12 | −0.03 ± 0.21 | 0.006 |
| seed 3 | −0.01 ± 0.05 | +0.08 ± 0.13 | — | 0.005 |
| jet-mass augmentation (±10 % scale, 10–20 % smear) | +0.08 ± 0.07 | +0.39 ± 0.15 | +0.36 ± 0.24 | 0.002 |
| physical QCD weights (clipped 10×) | −1.13 ± 0.09 | −1.01 ± 0.15 | −0.94 ± 0.29 | −0.004 |
| working-point gate weighting | −0.19 ± 0.09 | −0.09 ± 0.13 | −0.06 ± 0.22 | 0.002 |
| ensemble r14A + seeds 1–3 | +0.17 ± 0.04 | +0.32 ± 0.09 | +0.09 ± 0.16 | |
| ensemble r14A + seeds 1, 3 (gate + pairing) | +0.20 ± 0.06 | +0.34 ± 0.13 | — | |
| 30 epochs | +0.23 ± 0.08 | +0.36 ± 0.12 | −0.10 ± 0.21 | 0.018 |
| d256, 8 layers (≈8 M params) | +0.17 ± 0.09 | +0.14 ± 0.15 | −0.05 ± 0.28 | 0.048 (min val loss at epoch 8) |
| d384, 8 layers, ff×4 (≈20 M params) | −0.01 ± 0.10 | +0.19 ± 0.20 | −0.29 ± 0.25 | 0.075 (min val loss at epoch 7) |
| **interpretation attention (2 layers across the 70 pairings)** | **+0.69 ± 0.10** | **+0.48 ± 0.16** | +0.31 ± 0.21 | 0.002 |
| d256, 6 layers, ff×4 + interpretation attention | +0.89 ± 0.11 | +0.61 ± 0.15 | +0.28 ± 0.22 | 0.023 |
| dropout 0 | +0.43 ± 0.08 | +0.51 ± 0.15 | +0.32 ± 0.19 | 0.020 |
| focal-style QCD gate weight | −2.24 ± 0.13 | −0.31 ± 0.13 | — | |
| d256 6 layers + tail QCD | −1.46 ± 0.09 | −1.41 ± 0.14 | −1.95 ± 0.21 | |
| ensemble r14A+seed 1 | +0.15 ± 0.04 | +0.31 ± 0.09 | +0.11 ± 0.13 | |
| **massless (jet masses set to 0, retrained)** | −4.29 ± 0.14 | −3.83 ± 0.19 | −2.77 ± 0.34 | −0.009 |

Wider/deeper networks and longer training do **not** improve the working point (≤ +0.5 points, i.e. ≤ 2 %
relative, at the edge of seed noise) and start to overtrain (train–validation gap grows ×5–12).  f5_big6 (d256, 6 layers) was lost to a worker-side
transfer failure and not resubmitted (f5_big8 answers the question).  r14A is
not capacity-limited; with the current objective and inputs it is saturated.  The one structural change tried —
letting the 70 pairings attend to each other before scoring (`--interp-attn 2`, +0.26 M params) — gives the
largest gain of any single change: +0.7 points (+2.3 % relative) at 0.5 % QCD, +0.5 (+2 %) at 0.1 %, positive at
every working point and mass, with no overtraining, and a slightly better pairing (signal in window at 1 TeV
0.381 vs 0.372, QCD passing A < 0.1 0.090 vs 0.094).  Real but small: structure helps where width does not.  Adding width on top of the attention (d256, 6 layers, ff×4)
adds ≈ +0.1–0.2 more (+0.89 at 0.5 %, +0.61 at 0.1 %), within errors of the attention-only model, and starts to
overtrain (gap 0.023).

Jet masses are worth 14–17 % in signal efficiency at fixed QCD (×1.17 on the limit) when the network is
retrained without them — much less than zeroing the inputs of r14A suggests (×1.5–1.7, F3), and comparable to
what a realistic jet-mass resolution takes away (20 % smear + 10 % JER: ×1.5 for r14A).  A massless or
resolution-trained network is therefore close in performance and much more robust; it should be the default
until the jet-mass resolution of the real detector is in the training.  Mild jet-mass augmentation costs
nothing at nominal and reduces the QCD shift at fixed cuts only modestly (15 % smear: QCD at the nominal 0.1 %
threshold 0.122 % vs 0.140 % without augmentation; mass scale 0.9: 0.185 % vs 0.207 %); stronger augmentation
matched to the expected detector resolution is the recommended next training.  Loss re-weightings (physical QCD
weights, working-point weighting, focal) all lose.

## 10. Does more training data pay? (F9, data-scaling test)

Same recipe on a random 25 % / 50 % of the r14A training set (epochs scaled up to keep the number of steps);
100 % points are r14A, f8_ia2 and f5_big8.  Difference to r14A in mean signal efficiency (×100), paired bootstrap:

| model | fraction | 0.5 % QCD | 0.1 % QCD | 0.02 % QCD | gap (min val loss epoch) |
|---|---|---|---|---|---|
| r14A recipe | 25 % | +0.23 ± 0.11 | +0.67 ± 0.17 | +0.81 ± 0.24 | 0.123 (23 of 48) |
| r14A recipe | 50 % | +0.16 ± 0.07 | +0.54 ± 0.14 | +0.28 ± 0.22 | 0.024 (15 of 24) |
| r14A recipe | 100 % | 0 | 0 | 0 | 0.004 (12 of 12) |
| attention model | 25 % | +0.67 ± 0.10 | +0.78 ± 0.15 | +0.38 ± 0.30 | 0.077 (24 of 48) |
| attention model | 50 % | +0.64 ± 0.10 | +0.49 ± 0.13 | +0.15 ± 0.21 | 0.014 (20 of 24) |
| attention model | 100 % (f8_ia2) | +0.69 ± 0.10 | +0.48 ± 0.16 | +0.31 ± 0.21 | 0.002 (10 of 12) |
| big8 (d256, 8 layers) | 50 % | −0.82 ± 0.12 | −0.83 ± 0.15 | −1.03 ± 0.29 | 0.207 (14 of 40) |
| big8 (d256, 8 layers) | 100 % (f5_big8) | +0.17 ± 0.09 | +0.14 ± 0.15 | −0.05 ± 0.28 | 0.048 (8 of 20) |
| attention + longitudinal boost (η shift ≤ 0.5) | 100 % | +0.67 ± 0.11 | +0.73 ± 0.15 | +0.35 ± 0.21 | 0.001 (12 of 12) |

The mean over masses hides a shift: with 25 % of the data the r14A recipe is worse at low mass and better at high
mass (at 0.1 % QCD: 400 GeV 3.28 vs 3.61, 1000 GeV 17.9 vs 18.7, but 1500 GeV 32.9 vs 30.8, 2000 GeV 49.5 vs 46.8)
and overtrains strongly (gap 0.12).  At 50 % the low masses are back to the 100 % values.  The r14A curve is flat
from 50 % on.

**Verdict.**  Only the big model is data-hungry: doubling its data (50 → 100 %) gains ≈ +1.0 point (3–4 % relative)
and its overtraining drops from 0.21 to 0.05 — but at 100 % it only *matches* r14A.  The r14A-size and attention
models are flat from 50 % to 100 %.  Even if a further doubling gave the big model another +1 point, that is ≈ +3 %
over r14A — the same size as what the attention model plus the longitudinal boost already give with today's data,
and far smaller than the stage-2 gain (limits ×0.66) or the systematics (shower ×5 on the final-cut QCD).
Large filtered-QCD generation is therefore not recommended for the network; MC effort is better spent on the
QCD statistics of the *evaluation* (limits sit on 11–48 effective events) and on the shower systematic.
The attention model is flat from 25 % to 100 % in the mean (with the same low-mass/high-mass shift as r14A at 25 %:
400 GeV 3.42 vs 3.60 at 0.1 % QCD), confirming the verdict.  All six F9 runs harvested (Nov 18).

## 11. Optimising signal and QCD in the peak (F10)

Common test set: 93 k ancestry-labelled signal (400–2000 GeV) + 1.07 M held-out QCD (r14A's test files).  All
trainings: attention model + η boost, ancestry pairing targets (best 3+3 by true parent for every signal event),
checkpoint chosen on P(signal) at 0.5 % QCD; they differ in window-reward weight (5 / 1 / 0) and DisCo mass
decorrelation (0 / 1 / 5).  Script: F10_peak/peak_eval.py.  Ratios are signal kept at a fixed weighted QCD equal
to 10 % of r14A's QCD in the window, paired bootstrap errors.

**r14A sculpts smoothly but strongly.**  At global P cuts keeping 1 % / 0.1 % of QCD with A < 0.1, QCD efficiency
vs m_avg has log-log slope +2.0 / +2.8 (0.28× at 200–300 GeV to 30× at 2.1–2.5 TeV relative to average at 1 %);
no local bump beyond a smooth quadratic (≤ 3.7 σ).

**Pairing read-out.**  The analysis takes m_avg from the gated mixture, so low-P events get the anti-signal
(asymmetric) pairing.  At the tight working points (P ≈ 1) mixture and scorer pairings coincide, so the working
point numbers are unchanged; at no P cut the scorer pairing puts more signal in the window (400 GeV 25.4 % vs
21.3 %) and more QCD (×0.8–1.6).  Sculpting with the scorer pairing: slope +1.7 / +2.5.

Signal kept at a weighted QCD equal to 10 % of r14A's QCD in the wide window (ratio to r14A, bootstrap error),
with sculpting slope (log-log, QCD efficiency vs m_avg at global cuts keeping 1 % / 0.1 % of QCD):

| model | 400 | 700 | 1000 | 1500 | 2000 | sculpting slope | gap (min-val epoch) |
|---|---|---|---|---|---|---|---|
| r14A | 1 | 1 | 1 | 1 | 1 | +2.0 / +2.8 | 0.004 (12) |
| attention + η (quark targets, window reward 5) | 1.05 ± 0.02 | 1.03 ± 0.02 | 1.02 ± 0.04 | 1.03 ± 0.04 | 1.07 ± 0.04 | +2.0 / +2.9 | 0.001 (12) |
| f10a ancestry, window 5 | 0.74 ± 0.02 | 0.65 ± 0.02 | 0.82 ± 0.03 | 1.12 ± 0.04 | 1.02 ± 0.05 | +2.1 / +2.7 | 0.000 (7) |
| f10b ancestry, window 1 | 0.47 ± 0.02 | 0.72 ± 0.03 | 0.80 ± 0.02 | 1.02 ± 0.04 | 0.99 ± 0.07 | +2.4 / +3.1 | 0.011 (10) |
| f10c ancestry, window 0 | 0.26 ± 0.01 | 0.28 ± 0.01 | 0.55 ± 0.03 | 0.94 ± 0.05 | 0.84 ± 0.06 | +2.8 / +3.9 | 0.012 (9) |
| f10d ancestry, window 1, DisCo 1 | 0.43 ± 0.01 | 0.70 ± 0.03 | 0.81 ± 0.05 | 1.01 ± 0.05 | 0.99 ± 0.04 | +2.7 / +2.9 | 0.012 (3) |
| f10e ancestry, window 1, DisCo 5 | 0.52 ± 0.02 | 0.77 ± 0.03 | 0.86 ± 0.03 | 1.07 ± 0.04 | 1.06 ± 0.03 | +3.0 / +2.4 | 0.084 (4) |

Signal in the wide window with no P cut (r14A 21.3 / 30.7 / 35.9 / 45.8 / 61.0 %): f10a 15.3 / 30.5 / 41.7 / 56.6 /
62.4 % with QCD in the window ×0.9 / 1.5 / 2.0 / 1.6 / 1.0.  The ancestry target with the full window reward puts
more signal in the peak above 1 TeV (+16–24 % at 1–1.5 TeV) but also more QCD; at fixed QCD it wins only at 1.5
TeV and loses at ≤ 1 TeV.  Lowering the window reward loses signal from the window (the gated mixture read-out
then assigns the anti-signal pairing to low-P signal).  The training-level DisCo term did not reduce sculpting
at the tight cuts that matter (slopes unchanged or larger) and its strong version overtrains.  Caveat: the F10
runs were trained on the 0.85 M-event ancestry signal sample, without r14A's dedicated low-mass (H_T-filtered)
signal samples, which handicaps them at 400–700 GeV; the ancestry-target idea is therefore not cleanly tested at
low mass.  Plots: F10_peak/overtrain_*.pdf, eff_vs_epoch_*.pdf; full table F10_peak/peak_all.txt.

**Analysis-level decorrelation removes the sculpting at almost no cost** (F10_peak/ddt.py).  The gate logit is
replaced by its QCD tail quantile within the event's m_avg bin (14 log-spaced bins, 150–3000 GeV), fitted on half
of the test QCD and evaluated on the other half.  Any cut on it keeps the same QCD fraction at every m_avg:

| model | sculpting slope (1 % / 0.1 %) | S at 10 % of r14A window QCD, wide: 400 / 700 / 1000 / 1500 / 2000 |
|---|---|---|
| r14A | +2.0 / +2.8 | 1 / 1 / 1 / 1 / 1 |
| r14A, decorrelated cut | **−0.3 / −0.1** | 0.99 ± 0.01 / 0.96 ± 0.01 / 0.99 ± 0.01 / 0.98 ± 0.02 / 1.15 ± 0.09 |
| attention + η | +2.0 / +2.5 | 1.05 ± 0.02 / 1.02 ± 0.02 / 1.00 ± 0.05 / 1.06 ± 0.07 / 1.14 ± 0.04 |
| attention + η, decorrelated cut | **−0.6 / −0.5** | 1.06 ± 0.02 / 1.00 ± 0.03 / 1.00 ± 0.06 / 1.05 ± 0.06 / 1.07 ± 0.13 |

(evaluation on the held-out QCD half, so the reference numbers differ slightly from the table above.)  The
remaining residual at the 1 % cut (≈ 6 σ) comes from the steps between the 14 map bins; a smooth (interpolated)
map removes it.  Within a search window the per-mass P cut already acts locally, but the decorrelated score keeps
the QCD m_avg spectrum shape for the sideband/fit background estimate everywhere.

**Verdict (F10).**  Most signal and least QCD in the peak without sculpting: the attention + η network
(+2–8 % signal at fixed QCD) with the m_avg-binned quantile (decorrelated) cut, which flattens the QCD efficiency
vs m_avg (slope +2 → ≈ 0) at ≤ 4 % signal cost.  Ancestry pairing targets, lower window reward and the
training-level DisCo term do not help.  The stage-2 MLP must then be built on the decorrelated score and checked
for re-sculpting (it uses H_T/m and event shapes).

## 9. Status of running work

(updated as results arrive)
