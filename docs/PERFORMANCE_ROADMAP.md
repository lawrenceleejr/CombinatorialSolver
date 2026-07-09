# Performance roadmap for a real LHC analysis

Goal: the best possible identification of the combinatorial solution (which
jets came from which parent) and the best rejection of QCD multijet
background, without sculpting the m_avg spectrum.

## Recently implemented

1. **Pairwise-interaction attention bias** (`PairwiseInteractionBias` in
   `src/model.py`).  Every encoder layer's attention logits receive a learned
   per-head bias computed from (ln ΔR, ln kT, ln z, ln m²_ij, ln pT_i/pT_j)
   for each jet pair.  This is the interaction-matrix idea from the Particle
   Transformer (arXiv:2202.03772) and the single best-established
   architectural gain for jet transformers: the pairwise invariant mass and
   Lund-plane variables are exactly what distinguishes "these three jets came
   from one resonance" from "this pair is a QCD splitting".
2. **Richer jet tokens**: per-jet inputs extended from the raw four-vector to
   (E, px, py, pz, log pT, log E, η, log m).
3. **Event-level QCD discriminant with DisCo decorrelation**
   (`lambda_event`, `lambda_disco`).  Gives the analysis an explicit
   P(QCD|event) handle on top of the combinatorial self-selection, kept
   independent of m_avg by a distance-correlation penalty
   (arXiv:2001.05310) so a score cut cannot fake a bump.
4. **z-boost augmentation** (`data.z_boost_aug`): exact longitudinal-boost
   symmetry added to the existing φ-rotation / η-flip augmentation.
5. **Test-time augmentation** (`--tta N` in `src/evaluate.py`): average
   probabilities over exact-symmetry copies of each event.
6. **Confidence-based quality cut**: per-event max softmax probability saved
   with the results plus a purity-vs-efficiency table, so the analysis can
   trade efficiency for a sharper, purer mass peak.
7. **ONNX export fixed on modern PyTorch** (legacy exporter + `Asinh`
   symbolic; verified to reproduce eager logits to ~1e-7 with a dynamic
   batch axis).
8. **Rest-frame decay features per assignment** (`n_group_physics` 24 → 29):
   |cos θ*| = |tanh(Δy/2)| of the parent-candidate pair (pair production is
   central; QCD is forward-peaked), and the rest-frame Dalitz energy
   fractions x_i = 2(P·p_i)/m² per triplet, computed Lorentz-invariantly
   (a real 3-body decay shares energy democratically; a fake triplet
   collapses onto the Dalitz boundary).
9. **Event-shape features for the event head**: transverse sphericity
   (closed-form 2×2 eigenvalues), leading-pT fraction, pT hierarchy,
   rapidity span, min/mean ΔR — kinematics-only global-topology handles,
   LayerNormed and concatenated with the pooled embeddings.

## A note on the label definition

Training labels are argmin |m1−m2| computed on the *unsmeared* four-momenta
at load time, while Phase 1's pseudolabels are recomputed per batch on the
smeared, augmented inputs.  So the two phases target different things by
construction: Phase 1 teaches the network to mimic the classical heuristic
as applied to realistic (smeared) inputs; Phase 2 then trains toward the
clean-kinematics answer, i.e. the network learns to *undo* detector
smearing — which is exactly where it can outperform the classical solver
run on realistic data.  (`use_mass_asymmetry_labels: false` with
parton-matched TARGETS remains available as a cross-check of the label
definition on samples where it is trustworthy.)

## Highest-impact next steps

(All kinematics-only — no jet-substructure or quark/gluon-tagging inputs.)

### 1. Variable jet multiplicity

Real events do not have exactly 7 jets.  Support 6–10 jets with padding
masks in the encoder (masked attention), enumerate assignments over the
actual multiplicity, and allow ≥2 ISR/extra jets (choose 6 of N, then
partition).  Restricting to exactly-7 events costs signal efficiency and
biases toward cleaner topologies; the QCD control regions also need the
higher multiplicities.

### 2. Symmetry-aware target and partial credit

The 70-way cross-entropy treats "swapped one jet between groups" the same as
"completely wrong".  Options: (a) supervise a per-jet-pair "same parent"
adjacency matrix (SPANet-style symmetric tensor attention, arXiv:2106.03898)
and derive the assignment from it; (b) soft labels that give partial credit
proportional to the number of correctly grouped jets.  Both densify the
gradient signal and are known to help at high combinatorial multiplicity.

### 3. Systematics-aware training

Before unblinding, the JES/JER story matters: augment with correlated
jet-energy-scale shifts (not just uncorrelated smearing), and consider a
small domain-adversarial term between nominal and shifted copies so the
assignment is JES-stable.  Cheap insurance against the dominant experimental
systematic of an all-hadronic analysis.

### 4. Ensembling and calibration

- Snapshot ensembles fall out of the existing warm-restart schedule for
  free: keep the best checkpoint per restart cycle and average
  probabilities (complementary to `--tta`).
- Temperature-calibrate the confidence on a validation set so the
  purity-efficiency cut is stable across mass points and data-taking
  conditions.

### 5. ABCD with two decorrelated discriminants

The event QCD score + the mass asymmetry of the chosen assignment form a
natural pair for a data-driven ABCD background estimate; adding a second
DisCo term between the two scores themselves (the ABCDisCo construction,
arXiv:2007.14400) would let the analysis estimate the QCD background from
data instead of relying on multijet MC.

### 6. Significance-aware working point

Once shapes are stable, choose the confidence / event-score working points
by maximising expected bump-hunt significance (e.g. asymptotic AMS on the
m_avg spectrum with the multi-mass signal grid), rather than fixed
percentiles.
