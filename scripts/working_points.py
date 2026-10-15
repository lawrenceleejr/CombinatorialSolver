"""
Working points of an event score from search_study's per_event.npz: the cut
that keeps 1% / 0.3% / 0.1% of (weighted) QCD and the signal fraction kept at
each mass, as tabulated in models/README.md.

  python scripts/working_points.py results/search_study/per_event.npz "gated PairNet, gate cut"
"""

import sys

import numpy as np


def main():
    ev = np.load(sys.argv[1])
    sc = ev["score::" + sys.argv[2]]
    s, w, pm = ev["is_signal"].astype(bool), ev["w"], ev["pmass"]
    q, wq = sc[~s], w[~s]
    o = np.argsort(-q)
    cw = np.cumsum(wq[o]) / wq.sum()
    masses = sorted(np.unique(pm[s]).astype(int))
    print("| cut | QCD kept (weighted) | signal kept: " + " / ".join(f"{m}" for m in masses) + " GeV |")
    print("|---|---|---|")
    for e in (0.01, 0.003, 0.001):
        thr = q[o][min(np.searchsorted(cw, e), len(cw) - 1)]
        eff = [np.mean(sc[s & (pm == m)] >= thr) for m in masses]
        print(f"| `p_signal > {thr:.4f}` | {e:.1%} | " + " / ".join(f"{x:.2f}" for x in eff) + " |")


if __name__ == "__main__":
    main()
