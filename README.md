# CombinatorialSolver

Transformer-based neural network for combinatorial jet assignment in pair-produced resonance searches at the LHC.

Given 7 leading jets per event, the model identifies which jet is ISR and assigns the remaining 6 jets into two groups of 3, each corresponding to a parent particle. The predicted grouping yields a reconstructed mass variable for bump-hunt analysis.

## Architecture

- **Input**: 7 jets as (E, px, py, pz) four-vectors, normalized by event HT
- **Encoder**: Transformer with self-attention over jet tokens (4 layers, 8 heads, d=128)
- **Scorer**: Enumerates all 70 possible (ISR, group1, group2) assignments, aggregates jet embeddings per group with sum-pooling, scores with an MLP. Symmetry between the two parent groups is handled by sorting group embeddings by norm before concatenation.
- **Adversarial head**: Gradient-reversed MLP predicts parent mass from jet embeddings — penalizes the encoder if mass information leaks, preventing sculpting of the m_avg distribution.

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
