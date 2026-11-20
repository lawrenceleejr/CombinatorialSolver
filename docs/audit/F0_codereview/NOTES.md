# F0 — code review of src/pairnet.py, src/pairdata.py, scripts/train_pairnet.py (r14A)

No outright bug found in model, loss, loader or training loop. Findings that matter:

1. Pairing accuracy is ~25% on quark-matched signal at every mass and does not rise at P>=0.99
   (eval.npz of r14A, test split).  Of the signal that ends in the window at P>=0.99, only ~20% has the
   exact pairing, ~10% the proxy pairing, ~25-35% is matched-but-wrong and ~45-60% is not reconstructable.
   When the pairing IS exact, 92-98% of events fall in [0.85,1.1]M; matched signal at P>=0.99 gets only 60-80%.
   => the "peak" is mostly wrong-pairing-right-mass; the window reward (w_win=5) traded accuracy for it.
2. Evaluation/selection use argmax of the gated mixture: for events with P below ~0.75 the anti-signal
   pairing wins, so quoted accuracy / window fraction without a P cut are not the signal scorer's.
3. Checkpoint selection (win_a01 = window fraction / sqrt(weighted QCD passing A<0.1), no P cut) is taken at
   a working point ~1000x looser than the search (QCD eff 1e-4) and is flat after epoch 1
   (1.05 -> 1.15); weighted QCD pass is dominated by a few high-weight low-pThat events.
4. Val loss 2.217 (epoch 0) -> 2.172 (epoch 11), AUC 0.930 -> 0.932, no train/val gap: everything is learned
   in the first epoch; the model is saturated or capacity/objective-limited, not overtrained.
5. Gate BCE and mixture NLL are unweighted in QCD: P is the likelihood ratio against the *generated* QCD
   mixture (pThat-slice and filter composition), not physical QCD, and pooled over all signal masses.
6. pooled = [mean over real jets, max over ALL tokens incl. zeroed padding] -> max-pool is floor-clamped at 0
   for events with <12 jets (padding dependence; quantified in F3).
7. AMP was off for r14A (no bf16 mass precision issue).
