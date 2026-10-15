"""
AUC of each strategy's event score, overall and within jet-multiplicity bins
(6, 7, >= 8 jets), from search_study's per_event.npz. QCD is weighted; signal
masses are pooled with equal weight per event.

  python scripts/auc_by_njets.py results/search_study/per_event.npz --match "gate cut"
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_event")
    ap.add_argument("--match", default="gate cut", help="only scores whose name contains this")
    a = ap.parse_args()
    ev = np.load(a.per_event)
    s, w, nj = ev["is_signal"].astype(bool), ev["w"], ev["njall"]
    wt = np.where(s, 1.0, w)
    bins = {"6": nj == 6, "7": nj == 7, ">=8": nj >= 8, "all": np.ones_like(s)}
    print(f"{'score':64s} " + " ".join(f"{b:>6s}" for b in bins))
    for k in ev.files:
        if not k.startswith("score::") or a.match not in k or "TRUTH" in k:
            continue
        sc = ev[k]
        row = []
        for b, m in bins.items():
            row.append(roc_auc_score(s[m], sc[m], sample_weight=np.where(s[m], 1.0 / s[m].sum(),
                                                                          wt[m] / wt[m][~s[m]].sum())))
        print(f"{k[7:]:64s} " + " ".join(f"{v:6.3f}" for v in row))


if __name__ == "__main__":
    main()
