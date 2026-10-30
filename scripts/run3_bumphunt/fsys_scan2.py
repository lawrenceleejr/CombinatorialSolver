"""Expected limit vs fractional background uncertainty f for the mass-window search (enriched QCD), 300-2400 GeV.
For each mass and f: the best of the selections tuned for f = 0, 5% and 10% (low mass: lowmass_study4 halves with effS,
xsB_fb; high mass: fisher_high COUNT_F halves with S_per_fb, B), each re-evaluated at f on its held-out half.
usage: fsys_scan2.py style out low1.json,low2.json,... high1.json,high2.json,..."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out, lows, highs = sys.argv[1:5]
sys.path.insert(0, style); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
from limits import s_up  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
L = 300.0; FS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1]
sels = {}
for f in lows.split(","):
    for m, x in json.load(open(f))["results"].items():
        for k in ("P", "P+boost", "P+all"):
            if k in x and np.isfinite(x[k]["limit_fb"]):
                sels.setdefault(int(m), []).append([(h["effS"] * L, h["xsB_fb"] * L) for h in x[k]["halves"]])
for f in highs.split(","):
    for m, x in json.load(open(f))["results"].items():
        if int(m) > 1000 and np.isfinite(x["limit_fb"]) and all(h.get("feasible") and "S_per_fb" in h for h in x["halves"]):
            sels.setdefault(int(m), []).append([(h["S_per_fb"], h["B"]) for h in x["halves"]])
res = {m: {f: min(float(np.exp(np.mean([np.log(s_up(B, f) / S) for S, B in hs]))) for hs in ss) for f in FS} for m, ss in sorted(sels.items())}
fig, ax = plt.subplots(figsize=(4.6, 3.0))
cols = {300: "#5b8c5a", 500: "#9b2c2c", 700: "#b07d2b", 1000: "0.3", 1500: "#2b5d8a", 2000: "#7a5195"}
for m, c in cols.items():
    if m in res:
        y = np.array([res[m][f] / res[m][0.0] for f in FS])
        ax.plot([100 * f for f in FS], y, "o-", ms=2.5, lw=1.2, color=c)
        ax.annotate(f"{m} GeV", (100 * FS[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=c, va="center")
ax.set_yscale("log"); ax.set_xlabel("fractional background uncertainty [%]"); ax.set_ylabel("limit / limit with B known")
ax.set_title("mass window, enriched QCD: what the background uncertainty costs (best selection per value)", fontsize=6.2, loc="left")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/fsys_scan2.pdf"); fig.savefig(f"{out}/fsys_scan2.png", dpi=170, bbox_inches="tight")
json.dump({str(m): {str(f): v for f, v in r.items()} for m, r in res.items()}, open(f"{out}/fsys_scan2.json", "w"), indent=1)
print("M  " + "  ".join(f"{100 * f:g}%" for f in FS))
for m, r in res.items():
    print(m, "  ".join("%.3g" % r[f] for f in FS))
