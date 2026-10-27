"""r15 training status (both variants, r14A reference): efficiency vs epoch incl. the low-working-point figure of merit,
and train-vs-validation loss (overtraining check). One panel per file.
usage: r15_status.py style out name=train.log ..."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out = sys.argv[1:3]; runs = [a.split("=", 1) for a in sys.argv[3:]]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
COL = {"r14A": "0.55", "r15A": "#9b2c2c", "r15B": "#2b5d8a"}


def rows_of(path):
    rows = []
    for l in open(path, errors="ignore"):
        l = l.strip()
        if l.startswith("{") and '"epoch"' in l:
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    best = {}
    for r in rows:  # resumed slices can repeat an epoch: keep the last
        best[r["epoch"]] = r
    return [best[k] for k in sorted(best)]


data = {n: rows_of(p) for n, p in runs}
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for n, rows in data.items():
    if not rows:
        continue
    ep = [r["epoch"] for r in rows]; c = COL.get(n, "0.3")
    ax.plot(ep, [r.get("sig_at_psig_bkg", {}).get("0.005", np.nan) for r in rows], color=c, lw=1.6, marker="o", ms=3, label=f"{n}: signal in window at 0.5% QCD (P cut)")
    ax.plot(ep, [r["sig_in_window_at_bkg"]["0.05"] for r in rows], color=c, lw=0.9, ls="--", marker="o", ms=2, label=f"{n}: signal in window at 5% QCD (A cut)")
    ax.plot(ep, [r["eff_on_reconstructable"] for r in rows], color=c, lw=0.9, ls=":", label=f"{n}: exact pairing (reconstructable)")
ax.set_xlabel("epoch"); ax.set_ylabel("efficiency (validation)")
ax.set_title("r15 (r14A + enriched hard-negative QCD, epoch chosen at 0.5% QCD) vs r14A", fontsize=6.5, loc="left")
ax.legend(fontsize=5.6, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
fig.savefig(f"{out}/r15_eff_vs_epoch.png", dpi=170, bbox_inches="tight"); plt.close(fig)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for n, rows in data.items():
    if not rows or n == "r14A":
        continue
    ep = [r["epoch"] for r in rows]; c = COL.get(n, "0.3")
    ax.plot(ep, [r["val_loss"] for r in rows], color=c, lw=1.4, marker="o", ms=3, label=f"{n}: validation loss")
    ax.plot(ep, [r["train_eval_loss"] for r in rows], color=c, lw=1.0, ls="--", marker="o", ms=2, label=f"{n}: training-subset loss (eval mode)")
ax.set_xlabel("epoch"); ax.set_ylabel("loss")
ax.set_title("r15 overtraining check", fontsize=6.5, loc="left")
ax.legend(fontsize=5.8, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
fig.savefig(f"{out}/r15_loss_vs_epoch.png", dpi=170, bbox_inches="tight"); plt.close(fig)
for n, rows in data.items():
    if rows:
        r = rows[-1]
        print(f"{n} epoch {r['epoch']}: s@0.5%QCD(P) {r.get('sig_at_psig_bkg', {}).get('0.005', float('nan')):.4f}  s@5%(A) {r['sig_in_window_at_bkg']['0.05']:.3f}  "
              f"pairing {r['eff_on_reconstructable']:.3f}  val {r['val_loss']:.4f} train {r['train_eval_loss']:.4f} gap {r['val_loss'] - r['train_eval_loss']:+.4f}  {r['time']:.0f} s")
    else:
        print(f"{n}: no epochs yet")
