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
