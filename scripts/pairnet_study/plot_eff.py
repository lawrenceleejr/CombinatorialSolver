import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sk, tail, out = sys.argv[1:4]
sys.path.insert(0, sk)
from tufte_helpers import spines_to_data_range, save_pdf
plt.style.use(f"{sk}/tufte.mplstyle")

rows = []
for line in open(tail):
    if '"epoch"' in line and "{" in line:
        try: rows.append(json.loads(line[line.index("{"):]))
        except json.JSONDecodeError: pass
rows.sort(key=lambda r: r["epoch"])
ep = [r["epoch"] for r in rows]
eff = [r["eff_on_reconstructable"] for r in rows]

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.axhspan(0.22, 0.30, color="0.92", lw=0, zorder=0)
ax.text(ep[-1], 0.225, "testbed reference", ha="right", va="bottom", fontsize=8, color="0.45")
ax.plot(ep, eff, color="0.15", lw=1.2, marker="o", ms=2.5)
ax.annotate(f"{eff[-1]:.3f}", (ep[-1], eff[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
ax.set_xlabel("epoch")
ax.set_ylabel("exact-pairing efficiency\n(validation, reconstructable events)")
ax.set_ylim(0.20, 0.35)
spines_to_data_range(ax)
save_pdf(fig, out)
fig.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print(len(rows), "epochs", ep[0], "-", ep[-1])
