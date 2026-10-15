"""Figures of merit for the mass-variable study from mass_variables.py's vars.npz.

Per mass point and variable, after the asymmetry preselection on the chosen
interpretation:
  core68 : S/sqrt(B) in the SHORTEST window holding 68% of the preselected signal
           (a two-sided bump window set by the variable's own resolution)
  width  : that window's half-width / its centre (relative resolution)
  centre : window centre / M
  S      : fraction of ALL signal events in the window
  B      : weighted fraction of ALL QCD (>=6 jets) in the window

  python scripts/mass_variables_fom.py results/mass_variables_n7/vars.npz
"""
import json
import sys

import numpy as np

VARS = ["m3", "dressed0.8", "dressed1.2", "dressedinf", "thrust", "HT"]


def shortest(v, frac=0.68):
    v = np.sort(v)
    n = len(v)
    k = max(int(np.ceil(frac * n)), 2)
    w = v[k - 1:] - v[: n - k + 1]
    i = int(np.argmin(w))
    return v[i], v[i + k - 1]


def main(path, asym_cut=0.1):
    d = np.load(path)
    s = d["is_signal"].astype(bool)
    pre = d["asym"] < asym_cut
    w = d["w"]
    wq = w[~s].sum()
    floor = 3 * w[~s].mean() / wq
    out = {}
    for m in np.unique(d["pmass"][s]):
        sm = s & (d["pmass"] == m)
        r = {}
        for v in VARS:
            x = d[v]
            lo, hi = shortest(x[sm & pre])
            S = (sm & pre & (x >= lo) & (x <= hi)).sum() / sm.sum()
            B = (w * (~s & pre & (x >= lo) & (x <= hi))).sum() / wq
            r[v] = {"core68": float(S / np.sqrt(max(B, floor))), "S": float(S), "B": float(B),
                    "width": float((hi - lo) / (hi + lo)), "centre_over_M": float((hi + lo) / 2 / m),
                    "window": [float(lo), float(hi)]}
        out[int(m)] = r
        print(int(m), " ".join(f"{v}:{r[v]['core68']:.1f}(w{r[v]['width']:.2f},c{r[v]['centre_over_M']:.2f})" for v in VARS))
    json.dump(out, open(path.replace("vars.npz", "fom_core68.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
