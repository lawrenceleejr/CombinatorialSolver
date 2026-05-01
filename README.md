# CombinatorialSolver

Transformer-based neural network for combinatorial jet assignment in pair-produced resonance searches at the LHC.

## TL;DR — Quick-start commands

```bash
# 0. Generate mock data (for testing without real data)
python scripts/generate_mock_data.py --output data/mock_data.h5 --n-events 10000

# 1. Train
python -m src.train --config configs/default.yaml --data "data/*.h5"

# 2. Evaluate
python -m src.evaluate --checkpoint checkpoints/best_model.pt --data "data/test*.h5" --output results/

# 3. Export to ONNX (ML model)
python -m src.export_onnx --checkpoint checkpoints/best_model.pt --output-dir onnx_models/

# 3a. Export classical solver to ONNX (no checkpoint needed)
python -m src.export_onnx --classical-only --output-dir onnx_models/

# 4. Diagnose failures (ML model, HT-normalised inputs)
python scripts/diagnose_onnx.py \
    --model onnx_models/ml_model.onnx \
    --data "data/test*.h5" \
    --num-jets 7 --n-examples 30 --output results/

# 4a. Diagnose failures (classical solver, raw inputs)
python scripts/diagnose_onnx.py \
    --model onnx_models/classical_mass_asymmetry.onnx \
    --data "data/test*.h5" \
    --no-normalize
```

Given 7 leading jets per event, the model identifies which jet is ISR and assigns the remaining 6 jets into two groups of 3, each corresponding to a parent particle. The predicted grouping yields a reconstructed mass variable for bump-hunt analysis.

## Architecture

- **Input**: 7 jets as (E, px, py, pz) four-vectors, normalized by event HT
- **Encoder**: Transformer with self-attention over jet tokens (6 layers, 8 heads, d=256) plus per-head learnable pT-hierarchy attention bias (log pT_i/pT_j)
- **Physics-Conditioned GroupTransformer** *(inspired by [arXiv:2202.03772](https://arxiv.org/abs/2202.03772))*: Shared mini-Transformer (2 layers, 4 heads) for intra-group attention pooling. Before each GroupTransformer call the 9 per-group intra-group physics observables (pT hierarchy, Lund kT, ECF₂/ECF₃/D₂, Dalitz ratios) are computed directly from raw kinematics, projected to `d_model` via a shared `group_physics_proj` (Linear→GELU), and appended as a 4th conditioning token to the 3 jet-embedding tokens. The intra-group self-attention therefore sees **[jet₁, jet₂, jet₃, phys_cond]** and can condition on physics-derived context while those features remain anchored to the raw four-momenta rather than the latent space.
- **Scorer**: Enumerates all 70 possible (ISR, group1, group2) assignments, pools jet embeddings per group with the GroupTransformer, scores with an MLP. Group symmetry is handled via sum, Hadamard product, and absolute difference of the two group embeddings.
- **Extended physics features** (`n_group_physics=24`) per assignment (also fed to scorer MLP after LayerNorm):
  - *6 inter-group*: mass sum, mass asymmetry |m1-m2|/(m1+m2), mass ratio, m1, m2, ΔR between group CoM
  - *9 intra-group × 2 groups = 18*: max pT ratio, pT coefficient of variation, minimum Lund splitting fraction z, maximum Lund kT, ECF₂(β=1), ECF₃(β=1), D₂ = ECF₃/ECF₂², max and min Dalitz pairwise mass ratio
- **Adversarial head**: Gradient-reversed MLP predicts parent mass from jet embeddings — penalizes the encoder if mass information leaks, preventing sculpting of the m_avg distribution.

See [`architecture_diagram.pdf`](architecture_diagram.pdf) for a full data-flow diagram.

## Training Losses

| Loss term | Purpose |
|-----------|---------|
| `CrossEntropyLoss` (assignment) | Main supervised combinatorial loss |
| `lambda_adv × MSE` (adversary) | Decorrelate latent space from parent mass |
| `lambda_sym × E[mass_asym]` | Prefer balanced-mass assignments on average |
| `lambda_qcd × (-E[H · mass_asym])` | QCD penalty: push high-pT-hierarchy events to prefer high-asymmetry interpretations, making QCD background self-select non-signal-like regions of mass space |

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

This ensures the m_avg distribution is not sculpted when applied to background events.

## QCD Background Handling

Several design choices make QCD multijet backgrounds self-select non-signal-like combinatorial interpretations:

1. **pT hierarchy features** in each candidate group (max pT ratio, pT CV, Lund splitting z, Lund kT): QCD splittings are collinear/soft-enhanced, so these features directly identify QCD-like internal topology
2. **Energy Correlation Functions** (ECF₂, ECF₃, D₂ with β=1): IRC-safe multi-particle angular observables that distinguish isotropic signal from collimated QCD topologies
3. **Dalitz pairwise mass ratios**: Probe internal 3-body resonance structure; signal decays populate specific Dalitz regions while QCD fills it according to DGLAP evolution
4. **QCD penalty loss** (`lambda_qcd`): Soft-weights the assignment distribution by the event's pT hierarchy, pushing QCD-like events to prefer high-mass-asymmetry interpretations
5. **GroupTransformer pooling**: Attention over the 3 jets in each group preserves angular ordering and relative momentum flow that sum-pooling destroys
