# PairNet ONNX models

| file | jets in | recommended | notes |
|---|---|---|---|
| `pairnet_gated_7jet.onnx` | leading 7 | **yes** | drop-in for the current `ml_model.onnx` |
| `pairnet_v2_12jet.onnx` | leading 12 (pairs the leading 7) | for comparison | best MC numbers, but its gain comes only from events with ≥ 8 jets: it leans on extra-radiation jets, which depend on QCD modelling, and it does not help 6-jet hemisphere-mixed events |

Both models were trained on the Pythia 8 + FastJet testbed (gluino → uds at
500–2000 GeV, QCD p̂T > 200 GeV, anti-kT R = 0.4, pT > 30 GeV, |η| < 3). Retrain
on your samples with `scripts/train_pairnet.py`, then re-export with
`scripts/export_pairnet_onnx.py` and check the result with `scripts/validate_onnx.py`.

## Interface

**Input:** `four_momenta`, float32 of shape (batch, J, 4).
- Each jet is (E, px, py, pz), leading jet first, zero-padded. J = 7 (or 12 for v2).
- Units are free. Raw GeV and HT-normalised inputs give the same pairings and
  `p_signal`; `masses` come out in the input's units.

**Outputs:** the 70 assignments are in the order of
`src.combinatorics.enumerate_assignments(7)`, so `argmax` gives the same index
your code already decodes into (ISR, group1, group2).

| name | shape | meaning |
|---|---|---|
| `logits` | (batch, 70) | Gated-mixture log-probabilities. Signal-like events get their most signal-like pairing; QCD-like events get their high-asymmetry, low-mass pairing. A drop-in for the current `logits`. |
| `p_signal` | (batch,) | Event-level P(signal): the recommended selection variable. |
| `signal_logits` | (batch, 70) | The signal scorer alone. Use this pairing together with a `p_signal` cut. |
| `masses` | (batch, 70, 2) | m1 and m2 of the two triplets for every assignment. |

```python
import numpy as np, onnxruntime as ort
from src.combinatorics import enumerate_assignments

sess = ort.InferenceSession("models/pairnet_gated_7jet.onnx")
logits, p_signal, signal_logits, masses = sess.run(None, {"four_momenta": x})  # x: (N, 7, 4) float32

# as today: pairing from `logits`, then your asymmetry preselection
idx = logits.argmax(1)
isr, g1, g2 = zip(*(enumerate_assignments(7)[i] for i in idx))

# recommended: select on p_signal; mass from the signal scorer's pairing
best = signal_logits.argmax(1)
m1, m2 = masses[np.arange(len(best)), best].T
m_avg, asym = 0.5 * (m1 + m2), np.abs(m1 - m2) / (m1 + m2)
selected = p_signal > 0.98
```

## Working points on the testbed (7-jet model)

These must be re-derived on your own samples.

| cut | QCD kept (weighted) | signal kept: 500 / 750 / 1000 / 1250 / 1500 / 2000 GeV |
|---|---|---|
| `p_signal > 0.895` | 1% | 0.31 / 0.54 / 0.69 / 0.78 / 0.84 / 0.90 |
| `p_signal > 0.960` | 0.3% | 0.16 / 0.35 / 0.52 / 0.62 / 0.70 / 0.80 |
| `p_signal > 0.979` | 0.1% | 0.09 / 0.24 / 0.39 / 0.50 / 0.59 / 0.70 |

A single cut at about 0.1% QCD efficiency gave the best S/√B across masses
(`results/search_study_final`, `scripts/search_study.py`).

## Validation (`scripts/validate_onnx.py`)

- ONNX Runtime and PyTorch pick the same assignment in 100% of events.
  `p_signal` agrees within 5×10⁻⁴ and masses within 2×10⁻⁴ (relative).
- The legacy-order index decodes to exactly the jets PairNet chose.
- Raw GeV and HT-normalised inputs give identical pairings.
- Exact pairing on 1 TeV signal, computed from the ONNX outputs alone: 26% of
  events where the true pairing is available, the same as in PyTorch.
