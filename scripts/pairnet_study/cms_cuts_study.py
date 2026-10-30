"""Do CMS-style cuts (lowest jet pT of the six, Delta, D2_[3,2], D2_[(6,3)+(3,2)]) help on top of PairNet or argmin?

Per mass M, after A < 0.1 and the m_avg window [0.85, 1.1] M:
  pn_P       PairNet + P(signal) cut (the current analysis)
  pn_P+<v>   PairNet + P cut + one extra variable, thresholds scanned jointly
  pn_P+all   PairNet + P cut + all four variables (coordinate descent, 3 rounds)
  pn_cms     PairNet + the fixed CMS-EXO-21-004 region cuts + P cut
  am         argmin |m1-m2| (no further cuts)
  am+all     argmin + all four variables (coordinate descent)
  am_cms     argmin + the fixed CMS-EXO-21-004 region cuts, with A_m < CMS value instead of 0.1
Thresholds are tuned on one half of the test set (even / odd rows) and the expected limit is evaluated on the other half;
both directions are run and the two limits averaged geometrically. Tuning requires >= 10 QCD MC events kept in the
tuning half. Limit: median expected 95% CL, Asimov counting, no systematics (as xs_study.py).
"""
import glob, json, math, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

style, vars_npz, xs_json, cms_json, out = sys.argv[1:6]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

L = 300.0
MASSES = list(range(500, 2401, 100))
MIN_NB = 10
v = np.load(vars_npz); s = v["is_signal"]; q = ~s; w = v["w"].astype(np.float64); pm = v["pmass"]
half = np.arange(len(s)) % 2
qfiles = sorted(glob.glob("/scratch/llee/pairnet/data_r3/prod/qcd_c*9.h5"))
kq = np.mean([h5py.File(f).attrs["sigma_mb"] / h5py.File(f).attrs["weight_sum"] for f in qfiles]) * 1e12 / len(qfiles)
filt = {int(k): x for k, x in json.load(open(xs_json))["filter_eff"].items()}

VARS = {  # name: (key suffix, direction, thresholds in units of M where marked)
    "pt_low6": ("pt_low6", ">", [0, .03, .05, .07, .09, .11, .13, .15, .18, .22], True),
    "delta": ("delta_min", ">", [-1e9, -.5, -.4, -.3, -.2, -.1, 0, .05, .1, .15], True),
    "d2_32": ("d2_32_max", "<", [1e9, .5, .4, .3, .25, .2, .15, .1], False),
    "d2_63_32": ("d2_63_32", "<", [1e9, 2.5, 2, 1.5, 1.25, 1, .8, .6], False),
}
LABEL = {"pt_low6": "lowest jet $p_T$ of the six", "delta": r"$\Delta=\sum|p_T|-m_{jjj}$", "d2_32": r"$D^2_{[3,2]}$",
         "d2_63_32": r"$D^2_{[(6,3)+(3,2)]}$", "psig": "P(signal)"}


def cms_cuts(pre, M):
    """CMS-EXO-21-004 resolved-trijet region cuts (regions 2 and 3); 6th leading jet pT, Delta per triplet."""
    if M < 900:
        c = dict(pt6=50, d63=1.0, am=0.175, delta=180, d32=0.175)
    else:
        c = dict(pt6=100, d63=0.9, am=0.15, delta=150, d32=0.2)
    return ((v["pt6_lead"] > c["pt6"]) & (v[f"{pre}_d2_63_32"] < c["d63"]) & (v[f"{pre}_delta_min"] > c["delta"])
            & (v[f"{pre}_d2_32_max"] < c["d32"])), c["am"]


def s_up(B):
    if B <= 0:
        return 3.0
    f = lambda x: 2 * (x - B * math.log1p(x / B)) - 1.96 ** 2
    return max(brentq(f, 1e-9, 1e3 + 50 * math.sqrt(B) + 10 * B), 3.0)


class Mass:
    """Candidates for one mass hypothesis, split in halves."""

    def __init__(self, pre, M, extra=None, amax=0.1):
        win = (v[f"{pre}_mavg"] > 0.85 * M) & (v[f"{pre}_mavg"] < 1.1 * M) & (v[f"{pre}_asym"] < amax)
        if pre == "am":
            win &= v["am_valid"]
        if extra is not None:
            win &= extra
        self.sigM = s & (pm == M)
        self.nsig = [int((self.sigM & (half == h)).sum()) for h in (0, 1)]
        idx = np.where(win & (self.sigM | q))[0]
        self.idx, self.M, self.pre = idx, M, pre
        self.sg, self.h, self.w = s[idx], half[idx], w[idx]
        self.x = {n: v[f"{pre}_{k}"][idx] / (M if rel else 1) for n, (k, _, _, rel) in VARS.items()}
        self.x["psig"] = v["psig"][idx]

    def passes(self, cuts):
        ok = np.ones(len(self.idx), bool)
        for n, t in cuts.items():
            if n == "psig":
                ok &= self.x[n] >= t
            elif VARS[n][1] == ">":
                ok &= self.x[n] > t
            else:
                ok &= self.x[n] < t
        return ok

    def limit(self, ok, h):
        m = self.h == h
        eS = (ok & m & self.sg).sum() / max(self.nsig[h], 1) * filt[self.M]
        b = ok & m & ~self.sg
        xB = 2 * kq * self.w[b].sum()
        nB = int(b.sum())
        return (s_up(xB * L) / (L * eS) if eS > 0 else np.inf), nB, eS, xB

    def p_grid(self, h):
        ps = self.x["psig"][(self.h == h) & self.sg]
        return [-1.0] + list(np.quantile(ps, 1 - np.linspace(0.96, 0.2, 20))) if len(ps) else [-1.0]

    def tune(self, h, free, fixed=None, use_p=True, rounds=3):
        """Coordinate descent over the variables in `free` (+ P cut) on half h; returns the best cut dict."""
        cuts = dict(fixed or {})
        grids = {n: VARS[n][2] for n in free}
        if use_p:
            grids["psig"] = self.p_grid(h)
        for n in grids:
            cuts.setdefault(n, grids[n][0])
        best = self.limit(self.passes(cuts), h)[0]
        for _ in range(rounds):
            changed = False
            for n, g in grids.items():
                for t in g:
                    trial = {**cuts, n: t}
                    lim, nB, _, _ = self.limit(self.passes(trial), h)
                    if nB >= MIN_NB and lim < best * (1 - 1e-9):
                        best, cuts, changed = lim, trial, True
            if not changed:
                break
        return cuts

    def crossval(self, free, use_p=True, fixed=None):
        """Tune on one half, evaluate on the other, both ways; geometric-mean limit and the stats of the test halves."""
        lims, nBs, eSs, xBs, cs = [], [], [], [], []
        for h in (0, 1):
            c = self.tune(h, free, fixed, use_p)
            lim, nB, eS, xB = self.limit(self.passes(c), 1 - h)
            lims.append(lim); nBs.append(nB); eSs.append(eS); xBs.append(xB); cs.append(c)
        return {"limit_fb": float(np.exp(np.mean(np.log(lims)))), "limits_halves": lims, "nB_test": nBs, "effS": float(np.mean(eSs)),
                "xsB_fb": float(np.mean(xBs)), "cuts": [{k: float(x) for k, x in c.items()} for c in cs]}


res = {}
for M in MASSES:
    r = {}
    pn = Mass("pn", M)
    r["pn_P"] = pn.crossval([])
    for n in VARS:
        r[f"pn_P+{n}"] = pn.crossval([n])
    r["pn_P+all"] = pn.crossval(list(VARS))
    ex, amx = cms_cuts("pn", M)
    r["pn_cms"] = Mass("pn", M, ex).crossval([])
    am = Mass("am", M)
    r["am"] = am.crossval([], use_p=False)
    r["am+all"] = am.crossval(list(VARS), use_p=False)
    ex, amx = cms_cuts("am", M)
    r["am_cms"] = Mass("am", M, ex, amax=amx).crossval([], use_p=False)
    res[M] = r
    print(f"M={M:4d} " + "  ".join(f"{k} {x['limit_fb']:.3g}" for k, x in r.items()), flush=True)
json.dump(res, open(f"{out}/cuts_study.json", "w"), indent=1)

# ---------------- summary plots
Ms = np.array(MASSES)
get = lambda k: np.array([res[M][k]["limit_fb"] for M in MASSES])
fig, ax = plt.subplots(figsize=(5.0, 3.3))
for k, lab, c, ls in (("pn_P+pt_low6", LABEL["pt_low6"], "#c9a227", "-"), ("pn_P+delta", LABEL["delta"], "#5f8a5f", "-"),
                      ("pn_P+d2_32", LABEL["d2_32"], "#6a5f9b", "-"), ("pn_P+d2_63_32", LABEL["d2_63_32"], "#4a87a8", "-"),
                      ("pn_P+all", "all four", "#9b2c2c", "-"), ("pn_cms", "fixed CMS cuts", "0.35", "--")):
    y = get(k) / get("pn_P")
    ax.plot(Ms, y, color=c, lw=1.6 if k == "pn_P+all" else 1.1, ls=ls)
    ax.annotate(lab, (Ms[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=c)
ax.axhline(1, color="0.6", lw=0.6)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel("expected limit / PairNet + P cut")
ax.set_title("PairNet + P(signal) cut + extra cut(s); thresholds tuned and tested on separate halves", fontsize=7, loc="left")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/cuts_gain_pairnet.pdf"); fig.savefig(f"{out}/cuts_gain_pairnet.png", dpi=170, bbox_inches="tight")

cms = json.load(open(cms_json))
fig, ax = plt.subplots(figsize=(5.0, 3.3))
for k, lab, c, ls in (("am", r"argmin, $A<0.1$", "0.55", "--"), ("am_cms", "argmin + fixed CMS cuts", "0.3", "-"),
                      ("am+all", "argmin + tuned cuts", "0.3", ":"), ("pn_P", "PairNet + P cut", "#9b2c2c", "-"),
                      ("pn_P+all", "PairNet + P cut + tuned cuts", "#9b2c2c", ":")):
    y = get(k)
    ax.plot(Ms, y, color=c, lw=1.3, ls=ls)
    ax.annotate(lab, (Ms[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=c)
cm_m = np.array(sorted(int(x) for x in cms["ratio_cms_exp_over_ours"]))
ax.plot(cm_m, [cms["ratio_cms_exp_over_ours"][str(x)]["cms_exp_300_fb"] for x in cm_m], color="#2b5d8a", lw=1.3, marker="o", ms=2.5)
ax.annotate(r"CMS expected $\times\sqrt{128/300}$", (cm_m[-1], cms["ratio_cms_exp_over_ours"][str(cm_m[-1])]["cms_exp_300_fb"]),
            xytext=(4, 6), textcoords="offset points", fontsize=7, color="#2b5d8a")
ax.set_yscale("log"); ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"expected 95% CL limit on $\sigma$ [fb], 300 fb$^{-1}$")
ax.set_title("half-sample tuned, other half evaluated; no systematics", fontsize=7, loc="left")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/cuts_limits.pdf"); fig.savefig(f"{out}/cuts_limits.png", dpi=170, bbox_inches="tight")

# ---------------- variable distributions after PairNet A < 0.1 + window, before and after the tuned P cut
for M in (1000, 2000):
    pn = Mass("pn", M)
    pcut = res[M]["pn_P"]["cuts"][0]["psig"]
    for n, (k, _, _, rel) in VARS.items():
        x = pn.x[n] * (M if rel else 1)
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        lo, hi = np.quantile(x, [0.005, 0.995])
        bins = np.linspace(lo, hi, 40)
        pp = pn.x["psig"] >= pcut
        for sel, lab, c, ls, wt in ((pn.sg, f"signal {M} GeV", "#1f4e8c", "-", None), (pn.sg & pp, f"signal, P cut", "#1f4e8c", "--", None),
                                    (~pn.sg, "QCD", "0.25", "-", pn.w), (~pn.sg & pp, "QCD, P cut", "0.25", "--", pn.w)):
            ww = None if wt is None else wt[sel]
            hh, _ = np.histogram(x[sel], bins, weights=ww)
            if hh.sum() > 0:
                ax.stairs(hh / hh.sum(), bins, color=c, ls=ls, lw=1.2, label=lab)
        ax.set_xlabel(LABEL[n] + (" [GeV]" if rel else "")); ax.set_ylabel("fraction per bin (shape)")
        ax.set_title(f"PairNet, $A<0.1$, window, {M} GeV", fontsize=7, loc="left")
        ax.legend(fontsize=6.5, frameon=False)
        spines_to_data_range(ax)
        fig.savefig(f"{out}/dist_{n}_m{M}.png", dpi=170, bbox_inches="tight"); save_pdf(fig, f"{out}/dist_{n}_m{M}.pdf")
