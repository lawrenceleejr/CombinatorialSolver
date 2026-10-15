"""
Does a cut on the event score sculpt the QCD mass spectrum?

For a bump hunt the QCD selection efficiency must vary smoothly with the mass
variable.  Plots, for several score thresholds (fixed overall QCD efficiency),
the weighted QCD efficiency per mass bin; one PDF per mass variable.

  python scripts/sculpting.py results/mass_estimators/per_event.npz \
      --score gate --masses argmax reg --out plots/search
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle  # noqa: E402

plt = plotstyle.use()
SHADES = ["#1f1f1f", "#4c72b0", "#c44e52", "#8c8c8c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--score", default="gate")
    ap.add_argument("--masses", nargs="+", default=["argmax"])
    ap.add_argument("--effs", nargs="+", type=float, default=[0.1, 0.01, 0.002])
    ap.add_argument("--out", default="plots/search")
    a = ap.parse_args()
    d = np.load(a.npz)
    s = d["is_signal"].astype(bool)
    w = d["w"][~s]
    score = d[a.score][~s]
    bins = np.array([0, 150, 250, 350, 450, 550, 700, 850, 1000, 1200, 1450, 1750, 2100, 2500])
    ctr = 0.5 * (bins[1:] + bins[:-1])
    summary = {}
    os.makedirs(a.out, exist_ok=True)
    for mv in a.masses:
        m = d[mv][~s]
        tot, _ = np.histogram(m, bins, weights=w)
        totn, _ = np.histogram(m, bins)
        fig, ax = plt.subplots(figsize=(4.4, 3.1))
        summary[mv] = {}
        ymin, ymax = 1, 0
        for eff, c in zip(a.effs, SHADES[1:]):
            # threshold giving this weighted QCD efficiency overall
            order = np.argsort(-score)
            cw = np.cumsum(w[order]) / w.sum()
            thr = score[order][np.searchsorted(cw, eff)]
            sel = score >= thr
            num, _ = np.histogram(m[sel], bins, weights=w[sel])
            nn, _ = np.histogram(m[sel], bins)
            e = np.where(tot > 0, num / np.maximum(tot, 1e-30), np.nan)
            ok = (nn >= 3) & (totn >= 20)
            err = e / np.sqrt(np.maximum(nn, 1))
            ax.errorbar(ctr[ok], e[ok], yerr=err[ok], fmt="o-", color=c, ms=3, lw=1, capsize=0)
            if ok.any():
                k = np.where(ok)[0][-1]
                ax.text(ctr[k] * 1.04, e[k], f"{eff:g} of QCD kept", color=c, va="center", fontsize=8)
                ymin, ymax = min(ymin, np.nanmin(e[ok])), max(ymax, np.nanmax(e[ok]))
            summary[mv][str(eff)] = {"bins": ctr[ok].tolist(), "eff": e[ok].tolist()}
        ax.set_yscale("log")
        ax.set_xlabel({"argmax": r"$m_{avg}$ of chosen pairing [GeV]", "reg": "learned mass estimate [GeV]",
                       "mode": "posterior-mode mass [GeV]"}.get(mv, mv))
        ax.set_ylabel(f"QCD efficiency of the {a.score} cut")
        ax.set_xlim(0, 2700)
        ax.spines["bottom"].set_bounds(0, 2500)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.out, f"sculpting_{a.score}_{mv}.{ext}"), bbox_inches="tight")
        plt.close(fig)
    json.dump(summary, open(os.path.join(a.out, f"sculpting_{a.score}.json"), "w"), indent=1)
    print("wrote", [f"sculpting_{a.score}_{mv}" for mv in a.masses])


if __name__ == "__main__":
    main()
