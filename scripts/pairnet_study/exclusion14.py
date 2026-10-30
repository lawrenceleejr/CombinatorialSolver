"""Expected exclusion, 200-2400 GeV, 300 fb^-1 at 13.6 TeV, for an H_T-triggered (scouting) search.

M <= 1000 GeV: lowmass_study3 results (held-out MC, offline H_T > trig), best of the PairNet selections per mass
(P cut; P + H_T/Delta boost cuts; P + all cuts); argmin + all tuned cuts as the classical reference.
M > 1000 GeV: compare_models.py results (PairNet + tuned P cut, held-out QCD incl. the high-H_T sample).
Theory: NNLOapprox+NNLL (NNLL-fast, 13.6 TeV) from 500 GeV; below, Pythia LO x K(M) with log K linear in M fitted to
NNLO/LO at 500-1000 GeV (approximate, flagged in the plot).
usage: exclusion14.py style lowmass_dir cmp_json nnll_json out trig
"""
import glob, json, os, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, lmdir, cmp_json, nnll, out = sys.argv[1:6]
TRIG = float(sys.argv[6]) if len(sys.argv) > 6 else 450
NEW, OLDM = (os.environ.get("MODELS", "r14A,r13A")).split(",")
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

th = json.load(open(nnll))["data"]
tm = np.array(sorted(int(float(k)) for k in th)); tx = np.array([th[str(m)]["xsec_pb"] * 1e3 for m in tm]); tu = np.array([th[str(m)]["unc_pb"] * 1e3 for m in tm])
LM = "/scratch/llee/pairnet/prod_lowmass"; P3 = "/scratch/llee/pairnet/data_r3/prod"


def lo_sigma(M):
    fs = sorted(glob.glob(f"{LM}/siglm_m{M}_incl_c*.h5"))[:5] or sorted(glob.glob(f"{P3}/sig_m{M}_c*.h5"))[:5]
    return np.mean([h5py.File(f).attrs["sigma_mb"] for f in fs]) * 1e12 if fs else np.nan


LOM = np.array([200, 250, 300, 350, 400, 450, 500]); LOX = np.array([lo_sigma(m) for m in LOM])
KM = np.array([500, 600, 700, 800, 900, 1000])
K = np.array([np.exp(np.interp(m, tm, np.log(tx))) / lo_sigma(m) for m in KM])
kfit = np.polyfit(KM, np.log(K), 1)


def sigma_th(M):
    M = np.atleast_1d(M).astype(float)
    nn = np.exp(np.interp(M, tm, np.log(tx)))
    lo = np.exp(np.interp(M, LOM, np.log(LOX)))
    return np.where(M >= 500, nn, lo * np.exp(np.polyval(kfit, M)))


def low(model, fk):
    p = f"{lmdir}/lowmass_eval_{model}_trig{TRIG:g}_{fk}.json"
    return json.load(open(p))["results"] if os.path.exists(p) else None


cm = json.load(open(cmp_json))["limits"]
res = {}
for fk in ("f0", "f0.05", "frun2"):
    for model in (NEW, OLDM):
        lr = low(model, fk)
        rows = {}
        if lr:
            for m, x in lr.items():
                best = min([k for k in ("P", "P+boost", "P+all", "DDT", "DDT+boost") if k in x], key=lambda k: x[k]["limit_fb"])
                rows[int(m)] = {"limit_fb": x[best]["limit_fb"], "sel": best, "argmin_fb": x["am+all"]["limit_fb"]}
        for m, x in cm.get(model, {}).items():
            if int(m) > 1000 and fk in x:
                rows[int(m)] = {"limit_fb": x[fk], "sel": "P (high mass)"}
        res[f"{model}_{fk}"] = dict(sorted(rows.items()))

summary = {"K_fit": kfit.tolist(), "K_points": dict(zip(KM.tolist(), K.tolist())), "trig": TRIG, "rows": {}}
fig, ax = plt.subplots(figsize=(5.0, 3.4))
mm = np.arange(200, 2401, 10)
st = sigma_th(mm)
ax.plot(mm[mm >= 500], st[mm >= 500], color="0.1", lw=1.4, label=r"$\sigma(\tilde g\tilde g)$ NNLO$_{\rm approx}$+NNLL, 13.6 TeV")
ax.plot(mm[mm <= 500], st[mm <= 500], color="0.1", lw=1.2, ls="--", label=r"below 500 GeV: Pythia LO $\times$ extrapolated $K$")
for key, lab, c, ls in ((f"{NEW}_f0", f"PairNet {NEW}, no syst.", "#9b2c2c", "-"), (f"{NEW}_f0.05", f"PairNet {NEW}, 5% on B", "#9b2c2c", "--"),
                        (f"{NEW}_frun2", f"PairNet {NEW}, Run-2-like bkg. unc.", "#9b2c2c", ":"),
                        (f"{OLDM}_f0", f"PairNet {OLDM}, no syst.", "#c9a227", "-"), (f"{OLDM}_f0.05", f"PairNet {OLDM}, 5% on B", "#c9a227", "--")):
    r = res.get(key) or {}
    if not r:
        continue
    M = np.array(list(r)); y = np.array([r[m]["limit_fb"] for m in M])
    ax.plot(M, y, color=c, ls=ls, lw=1.4, marker="o", ms=2.2, label=lab)
    s = sigma_th(M)
    summary["rows"][key] = {int(m): {"limit_fb": float(l), "sigma_th_fb": float(t), "mu95": float(l / t), "sel": r[m]["sel"]} for m, l, t in zip(M, y, s)}
    ex = np.where(y < s)[0]
    if len(ex):
        i = ex[-1]
        if i + 1 < len(M):
            d0, d1 = np.log(y[i] / s[i]), np.log(y[i + 1] / s[i + 1])
            summary[f"reach_{key}"] = float(M[i] + (M[i + 1] - M[i]) * (-d0) / (d1 - d0))
        else:
            summary[f"reach_{key}"] = float(M[i])
ax.set_yscale("log"); ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"expected 95% CL limit on $\sigma$ [fb], 300 fb$^{-1}$")
ax.set_title(f"scouting-like offline $H_T>{TRIG:g}$ GeV; held-out MC", fontsize=6.5, loc="left")
ax.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/exclusion_trig{TRIG:g}.pdf"); fig.savefig(f"{out}/exclusion_trig{TRIG:g}.png", dpi=170, bbox_inches="tight")
json.dump(summary, open(f"{out}/exclusion_trig{TRIG:g}.json", "w"), indent=1)
for k, v in summary["rows"].items():
    print(k, {m: (round(x["limit_fb"], 2), f"mu={x['mu95']:.2g}", x["sel"]) for m, x in list(v.items())[:6]}, "reach", summary.get(f"reach_{k}"))
print("K(M) at 500-1000:", {m: round(k, 2) for m, k in zip(KM.tolist(), K.tolist())})
