# CombinatorialSolver

Transformer-based neural network for combinatorial jet assignment in pair-produced resonance searches at the LHC.

Given 7 leading jets per event, the model identifies which jet is ISR and assigns the remaining 6 jets into two groups of 3, each corresponding to a parent particle. The predicted grouping yields a reconstructed mass variable for bump-hunt analysis.

## Architecture

- **Input**: 7 jets as (E, px, py, pz) four-vectors, normalized by event HT, extended internally to 8 token features per jet (+ log pT, log E, η, log m) so the encoder sees both the linear four-vector and the log/angular parametrisation the physics lives in
- **Encoder**: Transformer with self-attention over jet tokens (4 layers, 8 heads, d=128) plus a **pairwise-interaction attention bias** (Particle-Transformer-style): a small MLP maps (ln ΔR, ln kT, ln z, ln m²_ij, ln pT_i/pT_j) for every jet pair to a per-head bias added to the attention logits of every layer — handing attention the QCD splitting variables directly instead of hoping it rediscovers them
- **GroupTransformer**: Shared mini-Transformer (1 layer, 4 heads) replaces sum-pooling for group embeddings, preserving intra-group angular ordering and multi-particle correlations
- **Scorer**: Enumerates all 70 possible (ISR, group1, group2) assignments, pools jet embeddings per group with the GroupTransformer, scores with an MLP. Group symmetry is handled via sum and Hadamard product of the two group embeddings.
- **Extended physics features** (`n_group_physics=29`) per assignment:
  - *7 inter-group*: mass sum, mass asymmetry |m1-m2|/(m1+m2), mass ratio, m1, m2, ΔR between group CoM, |cos θ*| production angle (tanh(Δy/2) — pair production is central, QCD is forward-peaked)
  - *11 intra-group × 2 groups = 22*: max pT ratio, pT coefficient of variation, minimum Lund splitting fraction z, maximum Lund kT, ECF₂(β=1), ECF₃(β=1), D₂ = ECF₃/ECF₂², max and min Dalitz pairwise mass ratio, max and min rest-frame Dalitz energy fraction x_i = 2(P·p_i)/m² (a real 3-body decay shares energy democratically; a fake triplet collapses onto the Dalitz boundary)
- **Adversarial head**: Gradient-reversed MLP predicts parent mass from jet embeddings — penalizes the encoder if mass information leaks, preventing sculpting of the m_avg distribution.
- **Event-level QCD discriminant**: A dedicated head on the pooled jet embeddings plus six kinematics-only event shapes (transverse sphericity, leading-pT fraction, pT hierarchy, rapidity span, min/mean ΔR) outputs P(QCD | event), trained with class-balanced BCE when a QCD sample is provided and decorrelated from the reconstructed average mass with a **DisCo** (distance-correlation) penalty so the analysis can cut on the score without sculpting the bump-hunt variable.

## Training Losses

| Loss term | Purpose |
|-----------|---------|
| `CrossEntropyLoss` (assignment) | Main supervised combinatorial loss |
| `lambda_adv × MSE` (adversary) | Decorrelate latent space from parent mass |
| `lambda_sym × E[mass_asym]` | Prefer balanced-mass assignments on average |
| `lambda_qcd × (-E[H · mass_asym])` | QCD penalty: push high-pT-hierarchy events to prefer high-asymmetry interpretations, making QCD background self-select non-signal-like regions of mass space |
| `lambda_bg × (CE + soft push)` | Background-rejection loss (QCD sample only): commit background events to the most anti-signal interpretation (high asymmetry OR low mass) |
| `lambda_event × BCE` | Event-level signal-vs-QCD discriminant (QCD sample only), class-balanced per batch |
| `lambda_disco × dCorr(score, m_avg)` | DisCo decorrelation of the event score from the expected average mass on background events — cutting on the score cannot sculpt a bump into the QCD m_avg spectrum |

The **QCD penalty** (`lambda_qcd`) uses H = log(pT_max / pT_min) as a per-event hierarchy score. Events with large H (QCD-like, one dominant jet) are pushed to assign to interpretations with large mass asymmetry and low average mass, disfavouring the signal-like bump-hunt region. Signal events (more balanced pT) are governed by the cross-entropy loss and resist this push.

## Setup

### Native (Mac with MPS, or CUDA)

```bash
pip install -r requirements.txt
```

### Docker

```bash
docker compose build
```

## Usage

### 1. Generate mock data (for testing)

```bash
python scripts/generate_mock_data.py --output data/mock_data.h5 --n-events 10000
# Or via Docker:
docker compose run generate-mock-data
```

### 2. Train

```bash
python -m src.train --config configs/default.yaml --data "data/*.h5"
# Or via Docker:
docker compose run train
```

The model auto-selects the best device (MPS -> CUDA -> CPU). Checkpoints are saved to `checkpoints/`, logs to `logs/`.

### 3. Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pt --data "data/test*.h5" --output results
# Or via Docker:
docker compose run evaluate
```

Outputs `results/mass_reconstruction.csv` (per-event) and `results/mass_arrays.npz` (numpy arrays for plotting the m_avg distribution).

Useful evaluation options:

- `--tta N` — test-time augmentation: average assignment probabilities over N azimuthal rotations × 2 η flips (exact symmetries of the physics; a free variance reduction at inference time, 4–8 recommended).
- Per-event **assignment confidence** (max softmax probability) and the **event QCD score** are written to the CSV/npz, and a purity-vs-efficiency table is printed so the analysis can trade signal efficiency for combinatorial purity (sharper mass peak) with a simple confidence cut.

## Data Format

Expects HDF5 files in the [MadGraphMLProducer](https://github.com/lawrenceleejr/MadGraphMLProducer) format:

- `jet_features`: (N, 20, 6) — [pt, eta, phi, mass, parent_pdg, is_signal]
- `jet_mask`: (N, 20)
- `event_features`: (N, 7) — [n_jets, met_x, met_y, met_pt, ht, n_signal, weight]

## Mass Agnosticity

The network is designed to not learn a specific mass value:

1. **Multi-mass-point training**: Mix signal samples from many parent masses
2. **Adversarial decorrelation**: Gradient-reversed head penalizes mass information in the latent space
3. **HT normalization**: Removes absolute energy scale from inputs
4. **No mass-based loss**: Only combinatorial assignment cross-entropy
5. **DisCo decorrelation** of the event-level QCD score from the reconstructed average mass

Training also applies exact-symmetry augmentation on every batch: random φ rotations, η reflections, and longitudinal (z) boosts (`data.z_boost_aug`, rapidity uniform in ±1 by default) plus fresh per-epoch pT smearing — none of which change the truth labels, so they are free statistics.

This ensures the m_avg distribution is not sculpted when applied to background events.

## QCD Background Handling

Several design choices make QCD multijet backgrounds self-select non-signal-like combinatorial interpretations:

1. **pT hierarchy features** in each candidate group (max pT ratio, pT CV, Lund splitting z, Lund kT): QCD splittings are collinear/soft-enhanced, so these features directly identify QCD-like internal topology
2. **Energy Correlation Functions** (ECF₂, ECF₃, D₂ with β=1): IRC-safe multi-particle angular observables that distinguish isotropic signal from collimated QCD topologies
3. **Dalitz pairwise mass ratios**: Probe internal 3-body resonance structure; signal decays populate specific Dalitz regions while QCD fills it according to DGLAP evolution
4. **QCD penalty loss** (`lambda_qcd`): Soft-weights the assignment distribution by the event's pT hierarchy, pushing QCD-like events to prefer high-mass-asymmetry interpretations
5. **GroupTransformer pooling**: Attention over the 3 jets in each group preserves angular ordering and relative momentum flow that sum-pooling destroys
6. **Event-level discriminant** (`lambda_event`): an explicit P(QCD|event) score for the analysis to cut on, DisCo-decorrelated (`lambda_disco`) from m_avg so the cut does not sculpt the background spectrum
7. **Pairwise-interaction attention bias**: the encoder's attention sees ln ΔR, ln kT, ln z, ln m²_ij for every jet pair — the variables in which QCD splittings are collinear/soft-enhanced

See `docs/PERFORMANCE_ROADMAP.md` for the prioritised list of further improvements toward a real LHC analysis.
