# Experiment queue

Worked top to bottom by the design loop in `docs/DESIGN_LOOP.md`. Each item
has one change, one primary metric, one success criterion and a budget cap.
Statuses: `todo`, `running`, `kept`, `reverted`, `inconclusive`, `failed`,
`done` (for items that produce infrastructure rather than a model result).

Metric names are defined in `docs/DESIGN_LOOP.md` section 4.

---

## Phase A: make the Tier 0 problem well posed

### E0 Correct labels and mask the padded slot
- status: todo
- depends on: none
- hypothesis: the network is capped by 8 to 10% label noise and by 60
  meaningless hypotheses per 6-jet event, not by capacity
- change: (a) read TARGETS from the corrected grouping, validate m(g1) and
  m(g2) per event, fall back to argmin |m1−m2| over valid hypotheses only;
  (b) key-padding mask in the encoder, exclude padded jets from all physics
  features and the pT bias, set pad-in-triplet logits to −∞;
  (c) fix the column-based fallback reader to the 7-column layout
- primary metric: `acc` on the 1 TeV file, 20% flat smearing
- success: `acc` above `acc_classical` (31% masked) by more than 5 points;
  label agreement with truth on unsmeared data above 97%
- budget: 3 GPU-hours

### E1 Remove signal-side QCD proxies
- status: todo
- depends on: E0
- hypothesis: `lambda_qcd`, `lambda_entropy_asym`, `lambda_entropy_mass`
  reduce signal accuracy when no QCD sample is present
- change: set all three to 0 in the experiment config; nothing else
- primary metric: `acc`
- success: `acc` improves or is unchanged within noise; then the zeros become
  the default when `qcd_data_path` is null
- budget: 3 GPU-hours

### E2 Realistic detector proxy
- status: todo
- depends on: E1
- hypothesis: the flat 20% smearing hides the true difficulty ordering of
  changes; the pT-dependent proxy in section 3.6 is the right default
- change: implement the section 3.6 smearing with seeded validation and test
  realisations; run 5%, the section 3.6 proxy, and 20% as three arms
- primary metric: `gain` per arm
- success: infrastructure lands (`done`); the section 3.6 proxy becomes the
  default for all later items
- budget: 4 GPU-hours

### E3 Pairwise interaction bias and log inputs
- status: todo
- depends on: E2
- hypothesis: attention that sees ln m²_ij, ln ΔR_ij, ln kT_ij, ln z_ij in
  every layer beats the single log pT ratio bias
- change: per-pair features through a 2-layer MLP to a per-head additive
  bias in every encoder layer; per-jet inputs log pT/HT, η, φ relative to
  the leading jet, log E/HT; the old pT bias removed
- primary metric: `gain`
- success: `gain` improves by more than 3 points over E2 at the section 3.6
  proxy
- budget: 6 GPU-hours

### E4 Model size against data size
- status: todo
- depends on: E3
- hypothesis: 7M parameters on 20k events overfits; the README's 1M model
  matches it, and both improve with 10x data
- change: four arms, {1M, 7M} x {20k, 200k Tier 0 events at 1 TeV}; requires
  Tier 0 generation with the producer TARGETS fix (section 3.2 item 1)
- primary metric: `acc`
- success: a clear ordering with non-overlapping uncertainties; the winning
  size becomes the default
- budget: 8 GPU-hours plus generation

### E5 Loss ablation
- status: todo
- depends on: E4
- hypothesis: the ten-term loss can be reduced to CE over valid hypotheses
  plus label smoothing and a small `lambda_sym` without losing accuracy
- change: arms {full current loss, minus Phase 2 distillation, minus
  teacher forcing and the two extra ISR paths, minimal loss}; two-phase
  training kept as an arm only if the minimal loss fails to converge
- primary metric: `acc`
- success: minimal loss within one uncertainty of the best arm; it becomes
  the default
- budget: 8 GPU-hours

## Phase B: showered data

### E6 Producer fixes and the Tier 1 baseline
- status: todo
- depends on: E5
- hypothesis: none; this establishes the physical baseline
- change: implement section 3.2 items 1 to 4 in a producer branch; generate
  the 1000 GeV signal at Tier 1 and a first QCD slice; validate; train the
  E5 default; report `acc_by_nj`, `acc_classical`, and the fraction of
  signal events that are reconstructable
- primary metric: `gain` at Tier 1
- success: infrastructure lands (`done`); numbers recorded as the Phase B
  baseline
- budget: 6 GPU-hours plus generation, generation capped at 20% of the
  loop's wall-clock

### E7 Variable multiplicity
- status: todo
- depends on: E6
- hypothesis: using 8 and 9 jets as extra-jet candidates recovers signal
  events the 7-slot model drops
- change: `num_jets` 9 with masked enumeration of disjoint triplet pairs;
  padded slots masked as in E0
- primary metric: `eff_S(1000)` at fixed SR, and `acc_by_nj`
- success: `eff_S` improves with `acc` at 6 and 7 jets unchanged within noise
- budget: 6 GPU-hours

## Phase C: topology head and physics metrics

### E8 Topology head with QCD and decorrelation
- status: todo
- depends on: E7 and the full QCD sample
- hypothesis: an explicit "two reconstructable triplets" score trained on
  QCD, decorrelated from m_avg with distance correlation, gives a larger `Z`
  than any push of QCD toward anti-signal interpretations, with `sculpt`
  within guardrail
- change: scalar head on the pooled event embedding plus per-hypothesis
  classical discriminants; BCE against reconstructable-signal versus QCD
  and unreconstructable signal; DisCo penalty on QCD in the batch; the
  existing `lambda_bg` push disabled
- primary metric: `Z(1000)` relative to classical; guardrails `sculpt`, `disco`
- success: `Z` above classical by more than 20% with `sculpt` χ²/ndf below 2
- budget: 10 GPU-hours

### E9 Multi-mass training and mass agnosticity
- status: todo
- depends on: E8 and the Tier 1 signal grid
- hypothesis: training on the mass grid removes the 1 TeV memorisation and
  the leave-one-mass-out efficiency loss is under 10%
- change: train on all masses; six leave-one-out trainings
- primary metric: `mass_loo(M)`; also `Z(M)` for every M
- success: `mass_loo(M)` above 0.9 for every M
- budget: 12 GPU-hours

### E10 SPANet baseline
- status: todo
- depends on: E6
- hypothesis: none; external reference point
- change: train SPANet on the Tier 1 files, which are already in its format
  once section 3.2 item 2 lands; same splits, same metrics
- primary metric: `acc`, `eff_S(1000)`
- success: recorded as reference; if SPANet leads by more than 5 points,
  add an item to port its symmetric triplet output into this model
- budget: 6 GPU-hours

### E11 Hyperedge scoring prototype (stretch)
- status: todo
- depends on: E8
- hypothesis: scoring 3-jet hyperedges directly handles 3+3, 3+X and
  "no triplet" in one output and matches E8's `Z`
- change: hyperedge head over all C(n,3) triplets with a disjointness
  decoder; the topology decision read from hyperedge confidences
- primary metric: `Z(1000)`, `sculpt`
- success: within noise of E8 with fewer parameters, or better
- budget: 10 GPU-hours
