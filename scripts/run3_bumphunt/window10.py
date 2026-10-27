"""Mass-window search with a 10% background uncertainty: exclusion plot and per-mass tables.

Window [0.85, 1.1] M on m_avg of the PairNet pairing, A < 0.1, per-mass cuts tuned to minimise the Asimov counting limit
with a 10% (fully correlated) uncertainty on B (Bhattiprolu-Martin-Wells), held-out MC, 300 fb^-1, 13.6 TeV.
Low mass (<= 1 TeV): lowmass_study4.py (P, H_T/M, Delta, Dalitz, p_T6, A); high mass: fisher_high.py COUNT_F=0.1 (P cut).
Classical reference: argmin |m1 - m2| with the same tuned cuts (low mass) / cuts_study3 am+all (high mass).
usage: window10.py style out cfg.json      (cfg: paths, see main)
"""
import glob, json, sys
import h5py
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out, cfgp = sys.argv[1:4]
cfg = json.load(open(cfgp))
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
L = 300.0
CONFS = ("P", "P+boost", "P+all")

th = json.load(open(cfg["nnll"]))["data"]
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


def low_rows(path, confs=CONFS):
    d = json.load(open(path))["results"]; rows = {}
    for m, x in d.items():
        c = [k for k in confs if k in x and np.isfinite(x[k]["limit_fb"])]
        if not c:
            rows[int(m)] = {"limit_fb": np.inf}; continue
        k = min(c, key=lambda k: x[k]["limit_fb"]); h = x[k]["halves"]
        rows[int(m)] = {"limit_fb": x[k]["limit_fb"], "sel": k, "cuts": h[0]["cuts"], "eff": float(np.mean([a["effS"] for a in h])),
                        "B": float(np.mean([a["xsB_fb"] for a in h])) * L, "neffB": float(min(a["neffB"] for a in h)),
                        "qcd_peak": float(np.mean([a.get("qcd_peak_over_M", np.nan) for a in h]))}
    return rows


def high_rows(path):
    d = json.load(open(path))["results"]; rows = {}
    for m, x in d.items():
        h = [a for a in x["halves"] if a.get("feasible")]
        if not np.isfinite(x["limit_fb"]) or not h:
            rows[int(m)] = {"limit_fb": np.inf}; continue
        rows[int(m)] = {"limit_fb": x["limit_fb"], "sel": "P", "cuts": {"psig": h[0]["psig"]}, "eff": float(np.mean([a["S_per_fb"] for a in h])) / L,
                        "B": float(np.mean([a["B"] for a in h])), "neffB": float(min(a["neffB"] for a in h)), "qcd_peak": float(np.mean([a["qcd_peak_over_M"] for a in h]))}
    return rows


def best_of(*tabs):
    out = {}
    for t in tabs:
        for m, r in t.items():
            if m not in out or r["limit_fb"] < out[m]["limit_fb"]:
                out[m] = r
    return out


# ---- curves
cur = {}
# M >= 1 TeV: best of the low-mass and high-mass samples (below 1 TeV the high-mass QCD sample is incomplete at low H_T)
hi_pk = {m: r for m, r in high_rows(cfg["high_pk"]).items() if m >= 1000}; hi_free = {m: r for m, r in high_rows(cfg["high_free"]).items() if m >= 1000}
cur["PairNet, 10% on B"] = best_of(best_of(*[low_rows(p) for p in cfg["low_pk"]]), hi_pk)
cur["PairNet, 10% on B, no falling-spectrum req."] = best_of(best_of(*[low_rows(p) for p in cfg["low_free"]]), hi_free)
if cfg.get("cats"):  # P(signal) categories, 10% per category (uncorrelated)
    cl_low = {}
    for p in cfg["low_pk"]:
        for m, x in json.load(open(p))["results"].items():
            v = min([x[k]["limit_fb"] for k in x if k.endswith("+cats")] + [np.inf])
            if np.isfinite(v) and (int(m) not in cl_low or v < cl_low[int(m)]["limit_fb"]):
                cl_low[int(m)] = {"limit_fb": v}
    hc = json.load(open(cfg["high_pk"]))["results"]
    cur["PairNet, 10% on B, P categories"] = best_of(cl_low, {int(m): {"limit_fb": x["count_cats_fb"]} for m, x in hc.items() if int(m) >= 1000 and "count_cats_fb" in x})
am_low = best_of(*[low_rows(p, ("am+all",)) for p in cfg["low_pk"]])
am_high = json.load(open(cfg["classical_high"]))
# classical: argmin + tuned cuts (low mass); above 1 TeV the best of argmin / max-Delta / max-Delta-among-A<0.1 + tuned cuts
cl_high = {int(m): {"limit_fb": min(x[k]["limit_fb"] for k in ("am+all", "dm+all", "dma+all"))} for m, x in am_high.items() if int(m) >= 1000}
cur["best tuned classical, 10% on B"] = best_of(am_low, cl_high)
cur["PairNet, B known exactly"] = {**{m: {"limit_fb": v} for m, v in cfg["f0_curve"].items()}} if isinstance(cfg.get("f0_curve"), dict) else {}
cur = {k: {int(m): r for m, r in v.items()} for k, v in cur.items() if v}

d = yaml.safe_load(open(cfg["cms_yaml"]))
cmM = np.array([x["value"] for x in d["independent_variables"][0]["values"]], float)
col = {v["header"]["name"]: np.array([x["value"] for x in v["values"]], float) for v in d["dependent_variables"]}
cms = col["Expected XSCN"] / col["Acceptance"] * 1e3 * np.sqrt(128 / 300)
segs = np.split(np.arange(len(cmM)), [i for i in range(1, len(cmM)) if cmM[i] == cmM[i - 1]])

STY = {"PairNet, 10% on B, P categories": ("#9b2c2c", "-", 2.2), "PairNet, 10% on B": ("#c26a5a", "-", 1.3), "PairNet, 10% on B, no falling-spectrum req.": ("#d08770", "--", 1.1),
       "best tuned classical, 10% on B": ("0.35", "-", 1.2), "PairNet, B known exactly": ("#9b2c2c", ":", 1.0)}
summary = {"curves": {}, "reach_GeV": {}}
fig, ax = plt.subplots(figsize=(5.0, 3.4))
mm = np.arange(200, 2401, 10); st = sigma_th(mm)
ax.plot(mm[mm >= 500], st[mm >= 500], color="0.1", lw=1.4, label=r"$\sigma(\tilde g\tilde g)$ NNLO$_{\rm approx}$+NNLL, 13.6 TeV")
ax.plot(mm[mm <= 500], st[mm <= 500], color="0.1", lw=1.2, ls="--", label=r"below 500 GeV: LO $\times$ 2.07")
for n, sg in enumerate(segs):
    ax.plot(cmM[sg], cms[sg], color="#2b5d8a", lw=1.2, label=r"CMS-EXO-21-004 expected $\times\sqrt{128/300}$" if n == 0 else None)
for key, r in cur.items():
    M = np.array(sorted(m for m in r if np.isfinite(r[m]["limit_fb"]))); y = np.array([r[m]["limit_fb"] for m in M]); s = sigma_th(M)
    c, ls, lw = STY[key]
    ax.plot(M, y, color=c, ls=ls, lw=lw, marker="o" if key.startswith("PairNet, 10% on B") and "no falling" not in key else None, ms=2.2, label=key)
    summary["curves"][key] = {int(m): {**{k: v for k, v in r[m].items()}, "sigma_th_fb": float(t), "mu95": float(l / t)} for m, l, t in zip(M, y, s)}
    ex = np.where(y < s)[0]
    if len(ex) and ex[-1] + 1 < len(M):
        i = ex[-1]; d0, d1 = np.log(y[i] / s[i]), np.log(y[i + 1] / s[i + 1])
        summary["reach_GeV"][key] = float(M[i] + (M[i + 1] - M[i]) * (-d0) / (d1 - d0))
ax.set_yscale("log"); ax.set_xlim(180, 2450)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"expected 95% CL limit on $\sigma$ [fb], 300 fb$^{-1}$")
ax.set_title(r"mass window [0.85, 1.1] M, 10% background uncertainty, offline $H_T>450$ GeV", fontsize=6.5, loc="left")
ax.legend(fontsize=6.0, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/exclusion_window10.pdf"); fig.savefig(f"{out}/exclusion_window10.png", dpi=170, bbox_inches="tight")

# ---- tables
NM = {"psig": "P", "ht": "H_T/M", "delta_max": "Δmax/M", "delta_min": "Δmin/M", "pt_low6": "pT6/M", "d32_max": "D²[3,2]", "d2_63_32": "D²[6,3]", "asym": "A"}


def cutstr(c):
    parts = []
    for k, v in (c or {}).items():
        if k == "psig":
            parts.append(f"P≥{v:.4g}" if v > -1 else "")
        elif k in ("ht", "delta_max", "delta_min", "pt_low6") and v > -1e8 and v != 0:
            parts.append(f"{NM[k]}>{v:g}")
        elif k in ("d32_max", "d2_63_32", "asym") and v < 1e8:
            parts.append(f"{NM[k]}<{v:g}")
    return ", ".join(p for p in parts if p) or "none"


main = summary["curves"]["PairNet, 10% on B"]; am = summary["curves"].get("best tuned classical, 10% on B", {})
lines = ["| M (GeV) | σ_th (fb) | limit (fb) | limit/σ | selection | cuts (half 0) | sig. eff. in window | B in window, 300 fb⁻¹ | S/B at limit | QCD peak / M | min. eff. QCD MC events | classical ÷ PairNet |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for m, r in sorted(main.items()):
    sb = r["limit_fb"] * r.get("eff", np.nan) * L / r["B"] if r.get("B") else np.nan
    ra = am[m]["limit_fb"] / r["limit_fb"] if m in am else np.nan
    lines.append(f"| {m} | {r['sigma_th_fb']:.3g} | {r['limit_fb']:.3g} | {r['mu95']:.2g} | {r.get('sel', '')} | {cutstr(r.get('cuts'))} | {r.get('eff', np.nan):.3g} | {r.get('B', np.nan):.3g} | {sb:.2g} | {r.get('qcd_peak', np.nan):.2f} | {r.get('neffB', np.nan):.0f} | {ra:.2f} |")
trig = ["| offline H_T > | " + " | ".join(str(m) for m in cfg["trig_masses"]) + " |", "|---" * (len(cfg["trig_masses"]) + 1) + "|"]
summary["trig_table"] = {}
for t, p in cfg["trig_files"].items():
    rr = low_rows(p); summary["trig_table"][t] = {m: rr[m]["limit_fb"] for m in cfg["trig_masses"] if m in rr}
    trig.append(f"| {t} | " + " | ".join(("%.3g" % rr[m]["limit_fb"]) if m in rr and np.isfinite(rr[m]["limit_fb"]) else "—" for m in cfg["trig_masses"]) + " |")
md = "\n".join(["## Mass window, 10% on B: per-mass numbers (offline H_T > 450 GeV, best of 450/500/600 offline)", ""] + lines +
               ["", "Expected 95% CL σ limits in fb by offline H_T threshold (— = no selection with a falling QCD spectrum under the window):", ""] + trig +
               ["", "Reach (limit = σ_th): " + ", ".join(f"{k}: {v:.0f} GeV" for k, v in summary["reach_GeV"].items())])
open(f"{out}/window10_tables.md", "w").write(md + "\n")
json.dump(summary, open(f"{out}/window10.json", "w"), indent=1, default=float)
print(md)
