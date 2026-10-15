"""Signal decay-quark |eta|: justification for the |eta|<3 jet acceptance.

  python scripts/eta_plots.py "data/pythia/sig_*.h5" --out plots/report
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, "..", "assets", "tufte.mplstyle"))
SHADES = ["#a9bfdf", "#8aa6d2", "#6a8cc2", "#4c72b0", "#34558e", "#1f3a6b"]


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


ap = argparse.ArgumentParser()
ap.add_argument("pattern")
ap.add_argument("--out", default="plots/report")
a = ap.parse_args()
by = defaultdict(list)
for fp in sorted(glob.glob(a.pattern)):
    with h5py.File(fp) as f:
        by[int(f.attrs["mass"])].append(f["truth/quarks"][:, :, 1])
eta = {m: np.abs(np.concatenate(v)) for m, v in sorted(by.items())}
ms = sorted(eta)

# (1) per-quark |eta| distribution (log scale shows the forward tail)
fig, ax = plt.subplots(figsize=(4.2, 3))
bins = np.linspace(0, 5, 51)
c = 0.5 * (bins[1:] + bins[:-1])
for i, m in enumerate(ms):
    h, _ = np.histogram(eta[m].ravel(), bins, density=True)
    ax.step(c, h, where="mid", color=SHADES[i], lw=1)
    if i == 0:
        ax.text(3.75, 1.2e-2, f"{m} GeV", color=SHADES[i], fontsize=8)
    if i == len(ms) - 1:
        ax.text(2.2, 6e-3, f"{m} GeV", color=SHADES[i], fontsize=8)
ax.set_yscale("log")
for x in (2.5, 3.0):
    ax.axvline(x, color="#8c8c8c", lw=0.6, ls=(0, (2, 2)))
    ax.text(x, 1.6, "%.1f" % x, fontsize=7, color="#8c8c8c", ha="center")
ax.set_xlabel(r"decay-quark $|\eta|$")
ax.set_ylabel("probability density")
ax.set_ylim(1e-3, 1.5)
ax.spines["bottom"].set_bounds(0, 5)
save(fig, a.out, "quark_eta")

# (2) fraction of events with all six quarks inside |eta| < X
fig, ax = plt.subplots(figsize=(4.2, 3))
xs = np.linspace(1.5, 5, 71)
res = {}
for i, m in enumerate(ms):
    mx = eta[m].max(1)
    y = [(mx < x).mean() for x in xs]
    ax.plot(xs, y, color=SHADES[i], lw=1)
    res[m] = {"all6_in_2.5": float((mx < 2.5).mean()), "all6_in_3.0": float((mx < 3.0).mean())}
    if i in (0, len(ms) - 1):
        ax.text(1.45, y[0], f"{m} GeV", color=SHADES[i], fontsize=8, ha="right", va="center")
for x in (2.5, 3.0):
    ax.axvline(x, color="#8c8c8c", lw=0.6, ls=(0, (2, 2)))
    lo_ = min(r[f"all6_in_{x}"] for r in res.values())
    hi_ = max(r[f"all6_in_{x}"] for r in res.values())
    ax.text(x + 0.05, 0.08 if x == 2.5 else 0.2,
            r"$|\eta|<%.1f$" % x + "\n%.0f to %.0f%%" % (100 * lo_, 100 * hi_),
            fontsize=7, color="#555555")
ax.set_xlabel(r"jet acceptance $|\eta|_{max}$")
ax.set_ylabel(r"events with all six quarks inside")
ax.set_ylim(0, 1.02)
ax.spines["bottom"].set_bounds(1.5, 5)
ax.set_xlim(0.9, 5.1)
ax.set_xticks([1.5, 2, 3, 4, 5])
ax.spines["left"].set_bounds(0, 1)
save(fig, a.out, "quark_eta_acceptance")
json.dump(res, open(os.path.join(a.out, "eta_acceptance.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
