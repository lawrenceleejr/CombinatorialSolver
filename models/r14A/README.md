# PairNet r14A (current model, Oct 22 2026)

Gated PairNet, 12 context jets, pairings among the leading 7 (70 interpretations), trained with the window term
(w_win = 5, w_qcd_pass = 5), on the 500–2400 GeV signal plus low-mass (200–1000 GeV) inclusive and boosted signal and
QCD with p̂T > 200 GeV, H_T > 1.2 TeV and p̂T > 80 GeV (H_T > 400 GeV) samples. 12 epochs, selection `win_a01`.

| file | what |
|---|---|
| `../pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx` | ONNX export (sha256 `cec5389e83cfe3ebf1f79fb40c3bbc790a9cd07937e25c40f8b8cebd17ed8aba`) |
| `../pairnet_gated_12jet_win5_lowmass_boost_r14A.json` | spec: inputs, outputs, assignment order |
| `pairnet_r14A_model.pt` | PyTorch checkpoint (`state`, `args`), load with `src.pairnet.load_checkpoint` (sha256 `7e5292be6bf575d1bf09f2c240ad51e5b3a40e1766ad4dc940d70fc916bb16d8`) |
| `metrics.json` | per-epoch training/validation metrics |
| `train_args.txt` | the exact `train_pairnet.py` arguments (HTCondor submit line) |

Interface as for the other models in `models/README.md`: input `four_momenta` (batch, 12, 4) of the leading 12 jets,
pT-ordered, zero-padded; `argmax(logits)` indexes `src.combinatorics.enumerate_assignments(7)`; `p_signal` is the
event-level selection variable. Recommended use: m_avg of the chosen pairing, A = |m1 − m2|/(m1 + m2) < 0.1, a per-mass
`p_signal` cut and the mass window [0.85, 1.1] M. Per-mass thresholds and expected limits: `docs` of the study in
`~/pairnet_study` (RUN3.md) and `scripts/run3_bumphunt`.
