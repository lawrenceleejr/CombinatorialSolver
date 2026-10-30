"""Plots from lowmass_study.py outputs: limits per selection, gains of the boost cuts, trigger-threshold scenarios."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, d, tag = sys.argv[1:4]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
load = lambda trig, f: json.load(open(f"{d}/lowmass_{tag}_trig{trig:g}_f{f:g}.json"))["results"] if os.path.exists(f"{d}/lowmass_{tag}_trig{trig:g}_f{f:g}.json") else None
SEL = (("P", "PairNet + P cut", "#9b2c2c", "--"), ("P+boost", "PairNet + P + $H_T$, $\\Delta$ cuts", "#9b2c2c", "-"),
       ("P+all", "PairNet + P + all cuts", "#c46a5f", ":"), ("am+all", "argmin + all tuned cuts", "0.35", "-"))


def save(fig, name):
    save_pdf(fig, f"{d}/{name}.pdf"); fig.savefig(f"{d}/{name}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


for f in (0.0, 0.05):
    r = load(0, f)
    if r is None:
        continue
    M = np.array(sorted(int(k) for k in r))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for k, lab, c, ls in SEL:
        ax.plot(M, [r[str(m)][k]["limit_fb"] for m in M], color=c, ls=ls, lw=1.4, marker="o", ms=2.5, label=lab)
    ax.set_yscale("log"); ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"expected 95% CL limit on $\sigma$ [fb], 300 fb$^{-1}$")
    ax.set_title("low masses, dedicated MC; " + ("no systematics" if f == 0 else "5% background uncertainty") + "; no trigger threshold", fontsize=6.5, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    spines_to_data_range(ax)
    save(fig, f"lm_limits_f{f:g}")

fig, ax = plt.subplots(figsize=(4.6, 3.0))
for f, ls in ((0.0, "-"), (0.05, "--")):
    r = load(0, f)
    if r is None:
        continue
    M = np.array(sorted(int(k) for k in r))
    g = lambda a, b: [r[str(m)][a]["limit_fb"] / r[str(m)][b]["limit_fb"] for m in M]
    sy = "no syst." if f == 0 else "5% on B"
    ax.plot(M, g("P", "P+boost"), color="#9b2c2c", ls=ls, lw=1.4, marker="o", ms=2.5, label=f"P cut alone / P + $H_T$, $\\Delta$ ({sy})")
    ax.plot(M, g("am+all", "P+all"), color="0.35", ls=ls, lw=1.4, marker="o", ms=2.5, label=f"argmin + cuts / PairNet + cuts ({sy})")
ax.axhline(1, color="0.6", lw=0.6)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"limit ratio ($>1$: right-hand selection better)")
ax.set_title("gains at low mass, tuned on one half, tested on the other", fontsize=6.5, loc="left")
ax.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save(fig, "lm_gains")

for f in (0.0, 0.05):
    fig, ax = plt.subplots(figsize=(4.6, 3.0)); any_ = False
    for trig, c in ((0, "#9b2c2c"), (600, "#4a87a8"), (1050, "#5f8a5f")):
        r = load(trig, f)
        if r is None:
            continue
        any_ = True
        M = np.array(sorted(int(k) for k in r))
        ax.plot(M, [r[str(m)]["P+boost"]["limit_fb"] for m in M], color=c, lw=1.4, marker="o", ms=2.5, label=f"PairNet + P + boost, trigger $H_T>{trig}$ GeV" if trig else "PairNet + P + boost, no trigger threshold")
        ax.plot(M, [r[str(m)]["am+all"]["limit_fb"] for m in M], color=c, lw=1.0, ls=":", label=f"argmin + cuts, $H_T>{trig}$ GeV" if trig else "argmin + cuts, no threshold")
    if not any_:
        plt.close(fig); continue
    ax.set_yscale("log"); ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"expected 95% CL limit on $\sigma$ [fb], 300 fb$^{-1}$")
    ax.set_title("trigger $H_T$ threshold; " + ("no systematics" if f == 0 else "5% background uncertainty"), fontsize=6.5, loc="left")
    ax.legend(fontsize=6.2, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    spines_to_data_range(ax)
    save(fig, f"lm_trigger_f{f:g}")

for trig in (0, 600, 1050):
    for f in (0.0, 0.05):
        r = load(trig, f)
        if r is None:
            continue
        print(f"trig {trig} f {f}:", {m: {k: round(x[k]["limit_fb"], 3 if x[k]["limit_fb"] < 10 else 1) for k in ("P", "P+boost", "P+all", "am+all")} | {"ISR": round(x.get("power_over_default_boost_eff") or 0, 2)} for m, x in r.items()})
