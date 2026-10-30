"""r13A training status: efficiency vs epoch and train-vs-validation loss vs epoch (one panel per file).

Validation for r13A includes the high-HT QCD files c1400-1699, so its QCD metrics are not comparable to r8B's."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, jsonl, out = sys.argv[1:4]
NAME = sys.argv[4] if len(sys.argv) > 4 else "r13A"
TITLE = sys.argv[5] if len(sys.argv) > 5 else "r8B setup + 2.1M high-$H_T$ QCD in training"
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
rows = [json.loads(l) for l in open(jsonl)]
ep = np.array([r["epoch"] for r in rows])

fig, ax = plt.subplots(figsize=(4.4, 2.8))
for k, lab, c in (("eff_on_reconstructable", "true pairing (reconstructable signal)", "#4a87a8"),
                  ("sig_pass_asym_in_window", r"signal with $A<0.1$ in the $m_{\rm avg}$ window", "#9b2c2c"),
                  ("bkg_pass_asym", r"QCD passing $A<0.1$", "0.4")):
    ax.plot(ep, [r[k] for r in rows], color=c, lw=1.3, marker="o", ms=3, label=lab)
ax.plot(ep, [r["sig_in_window_at_bkg"]["0.05"] for r in rows], color="#5f8a5f", lw=1.3, marker="o", ms=3, label="signal in window at 5% QCD")
ax.set_xlabel("epoch"); ax.set_ylabel("efficiency (validation)")
ax.set_title(f"{NAME}: {TITLE}", fontsize=7, loc="left")
ax.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
fig.savefig(f"{out}/{NAME}_eff_vs_epoch.png", dpi=170, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(4.4, 2.8))
ax.plot(ep, [r["val_loss"] for r in rows], color="#9b2c2c", lw=1.3, marker="o", ms=3, label="validation loss")
ax.plot(ep, [r["train_eval_loss"] for r in rows], color="0.3", lw=1.3, marker="o", ms=3, label="training-subset loss (eval mode)")
ax.set_xlabel("epoch"); ax.set_ylabel("loss")
ax.set_title(f"{NAME} overtraining check: matched class counts", fontsize=7, loc="left")
ax.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
fig.savefig(f"{out}/{NAME}_loss_vs_epoch.png", dpi=170, bbox_inches="tight")
r = rows[-1]
print(f"epoch {r['epoch']}: pairing {r['eff_on_reconstructable']:.3f}  sig A<0.1 in window {r['sig_pass_asym_in_window']:.3f}  "
      f"QCD A<0.1 {r['bkg_pass_asym']:.3f}  s@5% {r['sig_in_window_at_bkg']['0.05']:.3f}  rej50 {r['bkg_rej_at_50']:.0f}  "
      f"val {r['val_loss']:.3f} train {r['train_eval_loss']:.3f}  {r['time']:.0f} s/epoch")
