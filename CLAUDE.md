# CombinatorialSolver

Transformer that assigns the leading jets of an event to two 3-jet gluino
candidates plus extra jets, for an RPV multijet bump hunt in the average
triplet mass. Physics context, architecture and losses are in `README.md`.

## If you were handed compute to improve the network

Read `docs/DESIGN_LOOP.md` first and follow it. Then `experiments/QUEUE.md`
for what to do next and `experiments/ledger.csv` for what has been done.
Section 9 of the runbook lists facts already established; do not re-derive
them.

## Rules that always apply here

- Commit and push before any run longer than ten minutes. Push to the
  session's branch, never to `main`. No pull request unless asked.
- Every metric is reported next to the classical min |m1−m2| baseline on
  the same events, with an uncertainty.
- The test split is read only by `src.evaluate` at the end of an experiment.
- Plots follow Tufte: no gridlines, no chartjunk, direct labels, PDF, the
  serif style helpers already in `src/train.py`. Uncertainty is always shown.
- Failed or killed runs are reported as failed. Nothing is inferred from a
  run that did not finish.

## Layout

| path | what |
|---|---|
| `src/model.py` | encoder, factored ISR plus grouping scorer, physics features, classical solver |
| `src/train.py` | two-phase training loop, losses, per-epoch plots |
| `src/dataset.py` | HDF5 loader, pT sort, truth labels, smearing flags |
| `src/evaluate.py` | test-split evaluation and mass reconstruction |
| `configs/default.yaml` | current defaults |
| `data/MANIFEST.md` | every sample and its validation |
| `experiments/` | queue, ledger, per-experiment reports |

## Running

The reference machine is a Mac Studio; training uses the MPS backend
automatically. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the shell. For
unattended loops the user creates `.claude/settings.json` from the block in
`docs/DESIGN_LOOP.md` section 2.5 before starting.

```bash
pip install -r requirements.txt
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train --config configs/default.yaml --data "data/*.h5"
python -m src.evaluate --checkpoint checkpoints/best_model.pt --data "data/test*.h5" --output results
```

Sample generation uses `lawrenceleejr/MadGraphMLProducer`; the runbook's
section 3 lists the producer fixes required before its output can be used
for training.
