import json, glob, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sty = glob.glob("/home/llee/.claude/skills/synced/*/tufte-plots/assets/tufte.mplstyle")
if sty: plt.style.use(sty[0])
for n in sys.argv[1:]:
    h = json.load(open(f"runs/out_{n}/{n}/metrics.json"))["history"]
    ep = [r["epoch"] + 1 for r in h]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(ep, [r["val_loss"] for r in h], color="k", lw=1); ax.plot(ep, [r["train_eval_loss"] for r in h], color="0.55", lw=1)
    ax.text(ep[-1] + 0.2, h[-1]["val_loss"], "validation", va="center", fontsize=8)
    ax.text(ep[-1] + 0.2, h[-1]["train_eval_loss"], "training subset", va="center", fontsize=8, color="0.45")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title(n, fontsize=9)
    fig.savefig(f"plots/overtrain_{n}.pdf", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 3))
    for key, lab, c in (("0.005", "0.5 % QCD", "k"), ("0.002", "0.2 % QCD", "0.5")):
        y = [r["sig_at_psig_bkg"][key] for r in h]
        ax.plot(ep, y, color=c, lw=1); ax.text(ep[-1] + 0.2, y[-1], lab, va="center", fontsize=8, color=c)
    ax.set_xlabel("epoch"); ax.set_ylabel("signal in window (validation)"); ax.set_title(n, fontsize=9)
    fig.savefig(f"plots/eff_vs_epoch_{n}.pdf", bbox_inches="tight"); plt.close(fig)
print("ok")
