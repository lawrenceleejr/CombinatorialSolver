"""Expected limit vs fractional background uncertainty f, for the mass-window search (low mass, offline H_T > 450).
For each mass and f, the best of the selections tuned for f = 0, the Run-2-like uncertainty and f = 0.10 (all with the
QCD peak <= 0.75 M), each re-evaluated at f from its held-out-half signal efficiency and background (envelope).
usage: fsys_scan.py style out sel_json [sel_json ...]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
style, out = sys.argv[1:3]; files = sys.argv[3:]
sys.path.insert(0, style); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
from limits import s_up  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
L = 300.0; FS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
sels = {}
for f in files:
    for m, x in json.load(open(f))["results"].items():
        for k in ("P", "P+boost", "P+all"):
            if k in x and np.isfinite(x[k]["limit_fb"]):
                sels.setdefault(int(m), []).append([(h["effS"], h["xsB_fb"] * L) for h in x[k]["halves"]])
res = {}
for m, ss in sorted(sels.items()):
    res[m] = {}
    for f in FS:
        res[m][f] = min(float(np.exp(np.mean([np.log(s_up(B, f) / (L * e)) for e, B in hs]))) for hs in ss)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
cols = {300: "#5b8c5a", 400: "#b07d2b", 500: "#9b2c2c", 700: "#2b5d8a", 1000: "0.3"}
for m, c in cols.items():
    if m in res:
        y = np.array([res[m][f] / res[m][0.0] for f in FS])
        ax.plot([100 * f for f in FS], y, "o-", ms=2.5, lw=1.2, color=c)
        ax.annotate(f"{m} GeV", (100 * FS[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=c, va="center")
ax.set_yscale("log"); ax.set_xlabel("fractional background uncertainty [%]"); ax.set_ylabel("limit / limit with B known")
ax.set_title(r"mass window, offline $H_T>450$ GeV: price of the background uncertainty (best selection per value)", fontsize=6.2, loc="left")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/fsys_scan.pdf"); fig.savefig(f"{out}/fsys_scan.png", dpi=170, bbox_inches="tight")
json.dump({str(m): {str(f): v for f, v in r.items()} for m, r in res.items()}, open(f"{out}/fsys_scan.json", "w"), indent=1)
print("M  " + "  ".join(f"f={100*f:g}%" for f in FS))
for m, r in res.items():
    print(m, "  ".join("%.3g" % r[f] for f in FS))
