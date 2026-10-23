"""Expected exclusion for the falling-spectrum (bump-hunt) version of the search, 300 fb^-1, 13.6 TeV, offline H_T > trig.

Curves (PairNet r14A, held-out MC, QCD m_avg spectrum required to peak at <= 0.75 M so the window sits on its falling side):
  shape fit   binned m_avg fit, signal + free 4-parameter dijet background, stat-only (headline, realistic)
  counting    window [0.85, 1.1] M with the background known exactly (optimistic reference)
  Run-2-like  window counting with the CMS-EXO-21-004 fit uncertainty on B
CMS-EXO-21-004 expected limit scaled to 300 fb^-1 for reference. Theory NNLOapprox+NNLL (LO x K below 500 GeV).
usage: exclusion_fit.py style out trig fisher_low.json fisher_high.json shape_dir cmp_json cms_fig4c.yaml nnll.json
"""
import glob, json, os, sys
import h5py
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out, trig, f_low, f_high, shape_dir, cmp_json, cms_yaml, nnll = sys.argv[1:10]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
CONFS = ("P", "P+boost", "P+all")
TRIGS = os.environ.get("TRIGS", sys.argv[3]).split(",")

th = json.load(open(nnll))["data"]
tm = np.array(sorted(int(float(k)) for k in th)); tx = np.array([th[str(m)]["xsec_pb"] * 1e3 for m in tm])
LM = "/scratch/llee/pairnet/prod_lowmass"


def lo_sigma(M):
    fs = sorted(glob.glob(f"{LM}/siglm_m{M}_incl_c*.h5"))[:5]
    return np.mean([h5py.File(f).attrs["sigma_mb"] for f in fs]) * 1e12


LOM = np.array([200, 250, 300, 350, 400, 450, 500]); LOX = np.array([lo_sigma(m) for m in LOM])
KM = np.array([500, 600, 700, 800, 900, 1000])
kfit = np.polyfit(KM, np.log([np.exp(np.interp(m, tm, np.log(tx))) / lo_sigma(m) for m in KM]), 1)


def sigma_th(M):
    M = np.atleast_1d(M).astype(float)
    return np.where(M >= 500, np.exp(np.interp(M, tm, np.log(tx))), np.exp(np.interp(M, LOM, np.log(LOX))) * np.exp(np.polyval(kfit, M)))


def best(x):
    c = [k for k in CONFS if k in x and np.isfinite(x[k]["limit_fb"])]
    return (min(x[k]["limit_fb"] for k in c), min(c, key=lambda k: x[k]["limit_fb"])) if c else (np.inf, None)


curves = {}
# several comma-separated low-mass files (offline H_T thresholds >= trig): take the best per mass, since a tighter
# offline H_T cut is an allowed selection for an H_T > trig trigger (this also averages out tuning noise)
fls = [json.load(open(f))["results"] for f in f_low.split(",")]; fh = json.load(open(f_high))["results"]
fl = {m: min((best(f[m])[0] for f in fls if m in f)) for m in fls[0]}
curves["shape fit"] = {**{int(m): v for m, v in fl.items() if int(m) <= 1000}, **{int(m): x["limit_fb"] for m, x in fh.items() if int(m) > 1000}}
if len(sys.argv) > 10:  # P(signal) categories, each with its own free background fit
    fc = json.load(open(sys.argv[10]))["results"]
    bc = lambda x: min([x[k]["limit_fb"] for k in x if k.endswith("+cats") and isinstance(x[k], dict)] + [np.inf])
    curves["shape fit, P categories"] = {**{int(m): bc(x) for m, x in fc.items() if int(m) <= 1000}, **{int(m): x["limit_cats_fb"] for m, x in fh.items() if int(m) > 1000}}
if os.environ.get("LOW5"):  # 5-parameter dijet function (passes the spurious-signal check), P categories
    f5 = [json.load(open(f))["results"] for f in os.environ["LOW5"].split(",")]; h5 = json.load(open(os.environ["HIGH5"]))["results"]
    bc5 = lambda x: min([x[k]["limit_fb"] for k in x if k.endswith("+cats") and isinstance(x[k], dict)] + [best(x)[0]])
    curves["5-par fit, P categories"] = {**{int(m): min(bc5(f[m]) for f in f5 if m in f) for m in f5[0] if int(m) <= 1000},
                                         **{int(m): x["limit_cats_fb"] for m, x in h5.items() if int(m) > 1000}}
    curves.pop("shape fit, P categories", None)
cm = json.load(open(cmp_json))["limits"]["r14A"]
for key, fk in (("counting", "f0"), ("Run-2-like", "frun2")):
    sls = [json.load(open(f"{shape_dir}/lowmass_eval_r14A_trig{t}_{fk}.json"))["results"] for t in TRIGS
           if os.path.exists(f"{shape_dir}/lowmass_eval_r14A_trig{t}_{fk}.json")]
    curves[key] = {**{int(m): min(best(sl[m])[0] for sl in sls if m in sl) for m in sls[0]}, **{int(m): x[fk] for m, x in cm.items() if int(m) > 1000}}

d = yaml.safe_load(open(cms_yaml))
cmM = np.array([x["value"] for x in d["independent_variables"][0]["values"]], float)
col = {v["header"]["name"]: np.array([x["value"] for x in v["values"]], float) for v in d["dependent_variables"]}
cms = col["Expected XSCN"] / col["Acceptance"] * 1e3 * np.sqrt(128 / 300)
segs = np.split(np.arange(len(cmM)), [i for i in range(1, len(cmM)) if cmM[i] == cmM[i - 1]])

summary = {"trig": float(trig), "curves": {}, "reach_GeV": {}}
fig, ax = plt.subplots(figsize=(5.0, 3.4))
mm = np.arange(250, 2401, 10); st = sigma_th(mm)
ax.plot(mm[mm >= 500], st[mm >= 500], color="0.1", lw=1.4, label=r"$\sigma(\tilde g\tilde g)$ NNLO$_{\rm approx}$+NNLL, 13.6 TeV")
ax.plot(mm[mm <= 500], st[mm <= 500], color="0.1", lw=1.2, ls="--", label=r"below 500 GeV: LO $\times$ 2.07")
for n, s in enumerate(segs):
    ax.plot(cmM[s], cms[s], color="#2b5d8a", lw=1.2, label=r"CMS-EXO-21-004 expected $\times\sqrt{128/300}$" if n == 0 else None)
STY = {"5-par fit, P categories": ("#9b2c2c", "-", 2.2, "PairNet, 5-par. fit in 2-3 P(signal) categories"),
       "shape fit, P categories": ("#9b2c2c", "-", 2.2, "PairNet, shape fit in 2-3 P(signal) categories"),
       "shape fit": ("#c26a5a", "-", 1.0, "PairNet, 4-par. fit, one category (stat. only)"),
       "counting": ("#9b2c2c", "--", 1.0, "PairNet, window counting, B known exactly"),
       "Run-2-like": ("#9b2c2c", ":", 1.2, "PairNet, window counting, Run-2-like bkg. unc.")}
for key, r in curves.items():
    M = np.array(sorted(m for m in r if np.isfinite(r[m]) and m >= 250)); y = np.array([r[m] for m in M]); s = sigma_th(M)
    c, ls, lw, lab = STY[key]
    ax.plot(M, y, color=c, ls=ls, lw=lw, marker="o" if ("fit" in key) else None, ms=2.2, label=lab)
    summary["curves"][key] = {int(m): {"limit_fb": float(l), "mu95": float(l / t)} for m, l, t in zip(M, y, s)}
    ex = np.where(y < s)[0]
    if len(ex) and ex[-1] + 1 < len(M):
        i = ex[-1]; d0, d1 = np.log(y[i] / s[i]), np.log(y[i + 1] / s[i + 1])
        summary["reach_GeV"][key] = float(M[i] + (M[i + 1] - M[i]) * (-d0) / (d1 - d0))
ax.set_yscale("log"); ax.set_xlim(230, 2450)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"expected 95% CL limit on $\sigma$ [fb], 300 fb$^{-1}$")
ax.set_title(f"bump hunt on a falling spectrum: QCD $m_{{\\rm avg}}$ peak $\\leq$ 0.75 M, offline $H_T>{trig}$ GeV", fontsize=6.5, loc="left")
ax.legend(fontsize=6.0, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/exclusion_fit_trig{trig}.pdf"); fig.savefig(f"{out}/exclusion_fit_trig{trig}.png", dpi=170, bbox_inches="tight")
json.dump(summary, open(f"{out}/exclusion_fit_trig{trig}.json", "w"), indent=1)
for k, v in summary["curves"].items():
    print(k, {m: round(x["limit_fb"], 3) for m, x in v.items()}, "reach", summary["reach_GeV"].get(k))
for m in (300, 400, 500, 700, 1000, 1500, 2000):
    j = [i for i in range(len(cmM)) if cmM[i] == m]
    if j and m in summary["curves"]["shape fit"]:
        print(m, "CMS naive 300 / shape fit: %.2f" % (min(cms[j]) / summary["curves"]["shape fit"][m]["limit_fb"]))
