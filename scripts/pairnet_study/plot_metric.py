"""Plot one per-epoch metric from train_pairnet.py JSON log lines (one PDF per metric)."""
import argparse, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--style-dir", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--metric", required=True)
ap.add_argument("--ylabel", required=True)
ap.add_argument("--band", type=float, nargs=2, default=None)
ap.add_argument("--band-label", default="testbed reference")
ap.add_argument("--fmt", default="{:.3f}")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.style_dir)
from tufte_helpers import spines_to_data_range, save_pdf
plt.style.use(f"{a.style_dir}/tufte.mplstyle")

rows = {}
for line in open(a.log):
    if '"epoch"' in line and "{" in line:
        try:
            r = json.loads(line[line.index("{"):])
            rows[r["epoch"]] = r
        except json.JSONDecodeError:
            pass
ep = sorted(rows)
y = [rows[e][a.metric] for e in ep]

fig, ax = plt.subplots(figsize=(4.5, 3))
if a.band:
    ax.axhspan(*a.band, color="0.92", lw=0, zorder=0)
    ax.text(ep[-1], a.band[0], a.band_label, ha="right", va="bottom", fontsize=8, color="0.45")
ax.plot(ep, y, color="0.15", lw=1.2, marker="o", ms=2.5)
ax.annotate(a.fmt.format(y[-1]), (ep[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=8)
ax.set_xlabel("epoch")
ax.set_ylabel(a.ylabel)
spines_to_data_range(ax)
save_pdf(fig, a.out)
fig.savefig(a.out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print(a.metric, len(ep), "epochs", ep[0], "-", ep[-1], "last", y[-1])
