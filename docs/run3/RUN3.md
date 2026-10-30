# Run 3 gluino → 3 jets search with PairNet: the argument (Oct 20–21, 2026)

Slides: https://claude.ai/artifact/HwLsPFYWtKfJKeHghta1Ep (section "The Run 3 search, in one slide" onward).
Results: `results/run3/`, scripts: `scripts/run3/`.

## Recipe

1. Six-jet trigger and preselection (H_T-seeded, as CMS scouting).
2. PairNet (ONNX `deliverables/pairnet_gated_12jet_win5_f1d279f.onnx`, run r8B) picks the pairing among the leading
   7 jets with 12-jet context.
3. A < 0.1 on that pairing; bump hunt in m_avg = (m1 + m2) / 2.
4. P(signal) threshold as a function of m_avg at fixed QCD efficiency ("DDT"-style), so the QCD m_avg spectrum keeps
   its smooth falling shape; efficiency chosen per mass region.
5. Background from a fit to the m_avg spectrum in data.

## Evidence (300 fb⁻¹ at 13.6 TeV, median expected 95% CL, Asimov counting in [0.85, 1.1] M)

All thresholds are tuned per mass on one half of the MC and evaluated on the other half (both ways).
QCD MC: classical pairings use all 4.05M 6+-jet events; PairNet uses its 0.81M validation+test events plus a new
3.0M-event sample with H_T > 1.2 TeV (never used in training). The two samples agree on σ(H_T > 1.5/2/2.5 TeV)
within 0.5%.

| | no systematics | 5% background uncertainty |
|---|---|---|
| argmin \|m1−m2\|, A < 0.1 only | 1.29 TeV | – |
| max Δ among A < 0.1 + tuned cuts | 1.93 TeV | 1.86 TeV |
| argmin + tuned H_T, p_T, Δ, Dalitz cuts | 2.02 TeV | 2.02 TeV |
| CMS-EXO-21-004 expected × √(128/300) (13 TeV) | 1.95 TeV | – |
| **PairNet + P(signal) cut** | **2.31 TeV** | **2.24 TeV** |

(reach against NNLO_approx+NNLL σ, NNLL-fast, PDF4LHC21; −1 s.d. of σ: PairNet 2.14 / 2.08 TeV. Final tuning: wide Δ (to 1.0–1.3 M)
and H_T (to 4 M) grids, ≥ 10 effective QCD MC events and ≥ 30 signal MC events in the tuning half.)

- Best tuned classical ÷ PairNet σ limit, no systematics: 1.24 (0.5 TeV), 1.25 (1 TeV), 1.50 (1.5), 1.70 (2.0), 2.68 (2.2), 2.97 (2.4).
- With 5% on B: 0.6–0.95 up to 1.2 TeV (region excluded by Run 2), 1.4–3 at 1.5–2.4 TeV.
- Luminosity equivalence: PairNet with 100–140 fb⁻¹ reaches as far as the tuned cuts with 300 fb⁻¹; +150–400 GeV
  from 60 to 1000 fb⁻¹; the tuned cuts do not reach PairNet's 300 fb⁻¹ reach even with 1000 fb⁻¹.
- Boosted low-mass tail (Δ, H_T): after the P cut Δ_max > 0.3 M raises S/B 2.6–3× at 0.5–1 TeV but keeps only ~3% of
  the signal; with 5% on B, H_T > 2.3 M improves PairNet's 500 GeV limit 3×. Elsewhere tight boost cuts do not
  generalise with current MC (needs ME-matched hard-ISR signal and a trigger threshold to study properly).
- H_T is the cut the argmin baseline was missing (×4–9 on its own); the CMS Dalitz/Δ/p_T variables add another ×1.3–2
  to a classical pairing but nothing to PairNet (its P(signal) already uses that information).
- Max-Δ pairing (the interpretation with the largest min-triplet Σp_T − m) is poor alone (picks light boosted
  triplets); restricted to A < 0.1 and with tuned cuts it equals argmin + cuts.
- Fixed P cuts sculpt the low end of the QCD m_avg spectrum into a turn-on (no localised bump). The m_avg-dependent
  threshold keeps it smooth (dijet-function χ²/ndf 31/44 at 1% QCD efficiency) and its limits match the per-mass tuned
  P cut within 15%.

## Caveats

- Jets are smeared Pythia jets; no detector simulation, no quark–gluon tagging (CMS uses QGD).
- Counting in a window, not a spectrum fit; flat 5% background uncertainty as a stand-in.
- Training signal stops at 2.4 TeV; test-only signal at 2.6–3.0 TeV shows PairNet still works there (see below).
- The QCD efficiency of a fixed P cut moves by ×1.65 / ×0.72 under shower α_s variations: take the background from data.

## Retrained model (r13A, done Oct 21 02:28)

r8B configuration + 2.1M high-H_T QCD in training (`deliverables/pairnet_gated_12jet_win5_htqcd_r13A.onnx`, validated).
On identical held-out events: σ-limit ratio r13A/r8B = 0.95 (0.5–1.4 TeV), 0.88 (1.5–2.4 TeV) without systematics;
0.96 and 0.81 with 5% on B. Reach on those events 2.27 → 2.34 TeV (no syst), 2.20 → 2.34 TeV (5%). No overtraining.
Use r13A for the Run 3 study; re-derive its per-mass P(signal) thresholds.

## Heavier signal

Test-only signal at 2.6, 2.8, 3.0 TeV (prod_sighi): PairNet limits 0.23, 0.17, 0.13 fb at 300 fb⁻¹ vs σ = 0.11, 0.046,
0.020 fb, so the 2.29 TeV crossing is real. Reach vs luminosity (r8B): 2.29 (300), 2.43 (450), 2.59 TeV (1000 fb⁻¹);
tuned cuts 1.96, 2.01, 2.32 TeV.

## Bump hunt on a smoothly falling spectrum (Oct 22–23)

Question: keep the QCD m_avg turnover (Sudakov peak) low, with the signal peak on a steeply falling background, and
maximise σ sensitivity. Results in `results/bumphunt/`, scripts in `scripts/lowmass/` (lowmass_study4.py, fisher_high.py,
exclusion_fit.py, bump_plot.py, lever_arm.py, peak_anatomy.py, peak_info.py, peak_fom.py). Deck section "bump hunt" (slides fit_*).

- Figure of merit: binned m_avg fit, signal + 4-parameter dijet background with all parameters free, fitted from the QCD
  peak to 2M, stat-only Fisher information at 300 fb⁻¹ (limit = 1.96σ_μ). Window counting with a known background is
  optimistic by 2–3× because the free function absorbs the broad part of the signal.
- Optimising cuts for this figure of merit puts the QCD turnover at 0.47–0.62 M from 350 GeV with no constraint imposed;
  optimising for window counting puts it at 0.82–0.87 M (on the window). Requiring peak ≤ 0.75 M costs nothing (fit) or
  ≤5% (counting) from 300 GeV, ×1.5–2 at 250 GeV. 200 GeV: infeasible at offline H_T ≥ 400 (QCD peak 1.2 M; joint
  H_T/M × Δ × P scan finds nothing).
- Lever arm: same selection, fit starting at 0.5 M instead of 0.75 M → 1.5–1.8× better σ_μ.
- Shape-fit limits (H_T>450, best of 450/500/600 offline): 2.1 pb (250), 1.0 pb (300), 0.39 pb (400), 0.20 pb (500),
  60 fb (700), 22 fb (1 TeV), 7.5 fb (1.5 TeV), 1.3 fb (2 TeV). P(signal) categories (2–3, own fit each): ×0.77–0.91.
  Reach 2.08 TeV (2.13 with categories); counting with B known 2.23 TeV; Run-2-like counting 1.87 TeV.
- PairNet vs argmin + tuned H_T/Δ/Dalitz/pT cuts under the same shape fit: 1.3× (300 GeV, 1 TeV), 2–3× at 350–900 GeV.
- Versus the naive CMS-EXO-21-004 300 fb⁻¹ projection: CMS 2–2.8× better at 0.9–1.6 TeV, comparable at 0.5–0.8 TeV,
  PairNet 1.4–1.9× better at 400 GeV and 1.8–2 TeV. The earlier "1.3–4.4× better than CMS" compared counting with a known
  background to CMS's fit and does not hold.
- Signal anatomy: only 17–21% of the selected window signal is correctly paired (r14A exact-split accuracy 20–22% of
  reconstructable events; argmin 15–17%); correct pairings carry 19–30% of the fit information. Accuracy is flat at
  25–30% across all 30 training configs (r1–r14, with/without window and symmetry terms): the pedestal is mostly physics
  (FSR/ISR, quarks without a leading jet). +10 points of accuracy ≈ ×0.88 in σ; needs new per-jet information.
- DDT (m_avg-decorrelated P quantile) gives the same limits as the per-mass P cut within 1–6% (noisy ±20–40% at 250–300).
- Fit function (spurious signal on the QCD MC after each tuned selection, `results/bumphunt/spurious_compare.png`):
  4-parameter dijet fails (χ² for zero spurious signal 35/16; coherent −4…−14× the limit at 450–800 GeV), 5-parameter
  (adds log³x) passes (21/16). Test precision is only 0.7–10× the limit (QCD MC statistics). 5th parameter costs ×1.0–1.3.
- Headline (5-par, 2–3 P categories, `lowmass14fit5_pk0.75` + `fisher_high_npar5.json`): 2.1 pb (250), 1.3 pb (300),
  0.42 pb (400), 0.17 pb (500), 60 fb (700), 17 fb (1 TeV), 6.4 fb (1.5 TeV), ≈2 fb (2 TeV); reach ≈2.0 TeV (1.97).
  vs classical (same fit): 1.5× (300), 2.2–2.9× (400–900); at 1 TeV classical better with 5-par (×0.57), worse with
  4-par (×1.3): fit-based numbers move ±30–40% per point with QCD MC statistics.
  vs naive CMS projection: CMS 2–3× lower at 0.9–1.6 TeV, comparable at 0.5–0.8 and 1.8–2 TeV, 1.45× higher at 400 GeV.
- QCD peak ≤ 0.6 M (instead of 0.75): free from 300 GeV; 250 GeV infeasible.
- Shape fit (4-par, stat.) plus a spurious-signal term equal to the CMS Run-2 fit uncertainty × B in the window
  (`lowmass14fisherrun2_*`): ×2–5 at 250–450 GeV (where S/B ≈ 10⁻³), ×1.2–1.5 at 500 GeV–1 TeV; e.g. 4.0 pb (300),
  1.3 pb (400), 0.27 pb (500), 79 fb (700), 27 fb (1 TeV). Below 500 GeV the search is limited by how well the fit
  function is known, not by statistics.

## Mass-window search with a 10% background uncertainty (Oct 23)

Window [0.85, 1.1] M, A < 0.1, cuts tuned per mass for 10% fully correlated on B (Asimov counting), held-out MC,
300 fb⁻¹, offline H_T > 450 (best of 450/500/600), QCD peak ≤ 0.75 M, ≥ 20 effective QCD MC events when tuning.
Full per-mass table (cuts, efficiency, B, S/B, MC statistics) and H_T-threshold table: `results/window10/window10_tables.md`.

- Limits: 57 pb (250), 18.5 pb (300), 3.8 pb (400), 0.72 pb (500), 185 fb (700), 36 fb (1 TeV), 39 fb (1.2), 14 fb
  (1.5), 3.3 fb (1.8), 2.1 fb (2.0), 1.7 fb (2.4). Reach 1.95 TeV (B known: 2.23; best tuned classical: 1.93).
- 10% costs ×10–60 at 250–500 GeV, ×4 at 1 TeV, ×3 at 2 TeV relative to B known; all selections end at S/B ≈ 0.2.
  10% → 2% improves the limit 4–5× (300–400), 3.4× (500), 2.4× (1 TeV) (`fsys_scan`).
- PairNet vs best tuned classical (same floors): 1.8–7× better below 1 TeV; 1.5–2× worse at 1.1–1.6 TeV (P cut only;
  extra cuts don't help PairNet there); 1–1.9× better at 1.8–2.2 TeV.
- Falling-spectrum requirement: ×1.1–1.7 at 250–450 GeV, free above. 200 GeV only without it (124 pb at H_T>450).
- MC-limited: tuned selections sit at the QCD MC floor (9–31 effective events on the evaluation half).
- Network committed from ~/CombinatorialSolver (claude/pairnet-review): models/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx
  + .json, models/r14A/ (PyTorch checkpoint, metrics, training args, ONNX validation).
