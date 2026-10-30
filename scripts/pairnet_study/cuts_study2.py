"""CMS-style cuts on top of each pairing, with the larger QCD samples of big_vars.npz.

Pairings: am (argmin |m1-m2|), dm (max Delta_min), dma (max Delta_min among A < 0.1), pn (PairNet + P(signal) cut).
QCD: classical pairings use all 1000 QCD files (4.05M events, no training involved); PairNet uses its validation and test
files (c*8, c*9; 0.81M events; validation was only used to pick the epoch). Signal: test files (c*9).
Per mass M, base selection = A < 0.1 and m_avg in [0.85, 1.1] M. Configurations per pairing:
  base         no further cut (PairNet: P cut tuned)
  +<group>     one variable group tuned (pt_low6; Delta min/max; D2_32 max/min; D2_63_32), PairNet with the P cut
  +all         all groups tuned together (coordinate descent)
  cms          fixed CMS-EXO-21-004 region cuts: 6th-leading-jet pT, D2_63_32, A_m (replaces A < 0.1), and at least one
               triplet with Delta > cut and D2_32 < cut (PairNet: plus a tuned P cut)
Thresholds are tuned on one half (even/odd rows), evaluated on the other, both ways, limits averaged geometrically.
Tuning keeps >= 10 QCD MC events in the tuning half. Limit: median expected 95% CL, Asimov counting, no systematics.
"""
import glob, json, math, os, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

style, big, ht_npy, xs_json, cms_json, out = sys.argv[1:7]
FSYS = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0  # relative background uncertainty
sys.path.insert(0, style); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limits import s_up as s_up_f  # noqa: E402
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

L = 300.0
MASSES = list(range(500, 2401, 100))
MIN_NB = 10
v = dict(np.load(big))
v["ht"] = np.load(ht_npy)
assert len(v["ht"]) == len(v["w"])
s = v["is_signal"]; q = ~s; w = v["w"].astype(np.float64); pm = v["pmass"]
half = np.arange(len(s)) % 2
P = "/scratch/llee/pairnet/data_r3/prod"
fac = {}
for f in sorted(glob.glob(f"{P}/qcd_c*.h5")):
    with h5py.File(f) as h:
        fac[f] = h.attrs["sigma_mb"] / h.attrs["weight_sum"] * 1e12
k_all = np.mean(list(fac.values())) / len(fac)                       # fb per unit weight, all 1000 files
k_pn = np.mean([x for f, x in fac.items() if f[-4] in "89"]) / sum(f[-4] in "89" for f in fac)
filt = {int(k): x for k, x in json.load(open(xs_json))["filter_eff"].items()}
for p in ("am", "dm", "dma", "pn"):
    v[f"{p}_mavg"] = 0.5 * (v[f"{p}_m1"] + v[f"{p}_m2"])
    v[f"{p}_asym"] = np.abs(v[f"{p}_m1"] - v[f"{p}_m2"]) / np.maximum(v[f"{p}_m1"] + v[f"{p}_m2"], 1e-9)
    v[f"{p}_delta_min"] = np.minimum(v[f"{p}_delta1"], v[f"{p}_delta2"]); v[f"{p}_delta_max"] = np.maximum(v[f"{p}_delta1"], v[f"{p}_delta2"])
    v[f"{p}_d32_max"] = np.maximum(v[f"{p}_d32_1"], v[f"{p}_d32_2"]); v[f"{p}_d32_min"] = np.minimum(v[f"{p}_d32_1"], v[f"{p}_d32_2"])

VARS = {  # name: (key, direction, grid, grid in units of M)
    "pt_low6": ("pt_low6", ">", [0, .03, .05, .07, .09, .11, .13, .15, .18, .22], True),
    "ht": ("ht", ">", [0, .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3], True),
    "delta_min": ("delta_min", ">", [-1e9, -.5, -.4, -.3, -.2, -.1, 0, .05, .1, .15], True),
    "delta_max": ("delta_max", ">", [-1e9, -.3, -.2, -.1, 0, .05, .1, .15, .2, .3], True),
    "d32_max": ("d32_max", "<", [1e9, .5, .4, .3, .25, .2, .15, .1], False),
    "d32_min": ("d32_min", "<", [1e9, .3, .2, .15, .1, .07, .05], False),
    "d2_63_32": ("d2_63_32", "<", [1e9, 2.5, 2, 1.5, 1.25, 1, .8, .6], False),
}
GROUPS = {"ht": ["ht"], "pt_low6": ["pt_low6"], "delta": ["delta_min", "delta_max"], "d32": ["d32_max", "d32_min"], "d2_63_32": ["d2_63_32"]}


def s_up(B):
    return s_up_f(B, FSYS)


def cms_mask(p, M):
    c = dict(pt6=50, d63=1.0, am=0.175, delta=180, d32=0.175) if M < 900 else dict(pt6=100, d63=0.9, am=0.15, delta=150, d32=0.2)
    trip = ((v[f"{p}_delta1"] > c["delta"]) & (v[f"{p}_d32_1"] < c["d32"])) | ((v[f"{p}_delta2"] > c["delta"]) & (v[f"{p}_d32_2"] < c["d32"]))
    return (v["ht"] > 600) & (v["pt6_lead"] > c["pt6"]) & (v[f"{p}_d2_63_32"] < c["d63"]) & trip, c["am"]


class Mass:
    def __init__(self, p, M, extra=None, amax=0.1):
        use_q = q & (v["has_pn"] if p == "pn" else True)
        self.k = 2 * (k_pn if p == "pn" else k_all)  # x2: each half is half the sample
        sel = (v[f"{p}_mavg"] > 0.85 * M) & (v[f"{p}_mavg"] < 1.1 * M) & (v[f"{p}_asym"] < amax) & v[f"{p}_valid"]
        if extra is not None:
            sel &= extra
        sigM = s & (pm == M)
        self.nsig = [int((sigM & (half == h)).sum()) for h in (0, 1)]
        idx = np.where(sel & (sigM | use_q))[0]
        self.M, self.p = M, p
        self.sg, self.h, self.w = s[idx], half[idx], w[idx]
        self.x = {n: v[k if k == "ht" else f"{p}_{k}"][idx] / (M if rel else 1) for n, (k, _, _, rel) in VARS.items()}
        if p == "pn":
            self.x["psig"] = v["psig"][idx]

    def passes(self, cuts):
        ok = np.ones(len(self.sg), bool)
        for n, t in cuts.items():
            ok &= (self.x[n] >= t) if n == "psig" else (self.x[n] > t) if VARS[n][1] == ">" else (self.x[n] < t)
        return ok

    def limit(self, ok, h):
        m = self.h == h
        eS = (ok & m & self.sg).sum() / max(self.nsig[h], 1) * filt[self.M]
        b = ok & m & ~self.sg
        xB = self.k * self.w[b].sum()
        return (s_up(xB * L) / (L * eS) if eS > 0 else np.inf), int(b.sum()), eS, xB

    def tune(self, h, free, rounds=3):
        grids = {n: VARS[n][2] for n in free}
        if self.p == "pn":
            ps = self.x["psig"][(self.h == h) & self.sg]
            grids["psig"] = [-1.0] + list(np.quantile(ps, 1 - np.linspace(0.96, 0.2, 20)))
        cuts = {n: g[0] for n, g in grids.items()}
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

    def crossval(self, free):
        r = {"limits": [], "nB_test": [], "effS": [], "xsB_fb": [], "cuts": []}
        for h in (0, 1):
            c = self.tune(h, free)
            lim, nB, eS, xB = self.limit(self.passes(c), 1 - h)
            for k, x in zip(("limits", "nB_test", "effS", "xsB_fb", "cuts"), (lim, nB, eS, xB, {k: float(t) for k, t in c.items()})):
                r[k].append(x)
        r["limit_fb"] = float(np.exp(np.mean(np.log(r["limits"]))))
        r["min_nB_test"] = int(min(r["nB_test"]))
        return r


res = {}
for M in MASSES:
    r = {}
    for p in ("pn", "am", "dm", "dma"):
        mobj = Mass(p, M)
        r[f"{p}"] = mobj.crossval([])
        for g, names in GROUPS.items():
            r[f"{p}+{g}"] = mobj.crossval(names)
        r[f"{p}+all"] = mobj.crossval(list(VARS))
        ex, amx = cms_mask(p, M)
        r[f"{p}_cms"] = Mass(p, M, ex, amax=amx).crossval([])
    res[M] = r
    print(f"M={M:4d} " + "  ".join(f"{k} {x['limit_fb']:.3g}({x['min_nB_test']})" for k, x in r.items() if "+" not in k or k.endswith("all")), flush=True)
json.dump(res, open(f"{out}/cuts_study2_f{FSYS:g}.json", "w"), indent=1)
if FSYS > 0:
    sys.exit(0)

# ---------------- plots
Ms = np.array(MASSES)
get = lambda k: np.array([res[M][k]["limit_fb"] for M in MASSES])
low = lambda k: np.array([res[M][k]["min_nB_test"] < 5 for M in MASSES])


def line(ax, k, lab, c, ls, lw=1.2, ref=None):
    y = get(k) / (get(ref) if ref else 1)
    ax.plot(Ms, y, color=c, ls=ls, lw=lw)
    bad = low(k)
    if bad.any():
        ax.plot(Ms[bad], y[bad], ls="none", marker="o", mfc="white", mec=c, ms=3)
    ax.annotate(lab, (Ms[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.8, color=c)


for p, title in (("pn", "PairNet + P(signal) cut"), ("am", r"argmin $|m_1-m_2|$"), ("dma", r"max $\Delta_{\min}$ among $A<0.1$")):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for k, lab, c in ((f"{p}+ht", "$H_T$", "0.2"), (f"{p}+pt_low6", "lowest jet $p_T$ of the six", "#c9a227"), (f"{p}+delta", r"$\Delta$ (min, max triplet)", "#5f8a5f"),
                      (f"{p}+d32", r"$D^2_{[3,2]}$ (worst, best triplet)", "#6a5f9b"), (f"{p}+d2_63_32", r"$D^2_{[(6,3)+(3,2)]}$", "#4a87a8"),
                      (f"{p}+all", "all tuned together", "#9b2c2c"), (f"{p}_cms", "fixed CMS cuts", "0.35")):
        line(ax, k, lab, c, "--" if k.endswith("cms") else "-", 1.6 if k.endswith("all") else 1.1, ref=p)
    ax.axhline(1, color="0.6", lw=0.6)
    ax.set_yscale("log"); ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(f"expected limit / {title.split(' +')[0]} alone")
    ax.set_title(f"{title} + extra cuts; tuned and tested on separate halves; open circles: < 5 QCD MC events", fontsize=6.5, loc="left")
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/gain_{p}.pdf"); fig.savefig(f"{out}/gain_{p}.png", dpi=170, bbox_inches="tight")

cms = json.load(open(cms_json))
fig, ax = plt.subplots(figsize=(5.0, 3.4))
for k, lab, c, ls in (("am", r"argmin, $A<0.1$", "0.6", "--"), ("am+all", "argmin + tuned cuts", "0.6", "-"),
                      ("dma", r"max $\Delta$ ($A<0.1$)", "#5f8a5f", "--"), ("dma+all", r"max $\Delta$ ($A<0.1$) + tuned cuts", "#5f8a5f", "-"),
                      ("dm", r"max $\Delta$", "#c9a227", "--"), ("am_cms", "argmin + fixed CMS cuts", "0.2", ":"),
                      ("pn", "PairNet + P cut", "#9b2c2c", "-"), ("pn+all", "PairNet + P + tuned cuts", "#9b2c2c", ":")):
    line(ax, k, lab, c, ls, 1.3)
cm_m = np.array(sorted(int(x) for x in cms["ratio_cms_exp_over_ours"]))
yc = [cms["ratio_cms_exp_over_ours"][str(x)]["cms_exp_300_fb"] for x in cm_m]
ax.plot(cm_m, yc, color="#2b5d8a", lw=1.3, marker="o", ms=2.5)
ax.annotate(r"CMS exp. $\times\sqrt{128/300}$", (cm_m[-1], yc[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.8, color="#2b5d8a")
ax.set_yscale("log"); ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"expected 95% CL limit on $\sigma$ [fb], 300 fb$^{-1}$")
ax.set_title("half-sample tuned, other half evaluated; no systematics; open circles: < 5 QCD MC events", fontsize=6.5, loc="left")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/limits_all.pdf"); fig.savefig(f"{out}/limits_all.png", dpi=170, bbox_inches="tight")

# m_avg after A < 0.1 at 2 TeV for each pairing: signal and QCD
for M in (1000, 2000):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bins = np.linspace(0, 1.6 * M, 65)
    for p, lab, c in (("am", "argmin", "0.55"), ("dm", r"max $\Delta$", "#c9a227"), ("dma", r"max $\Delta$, $A<0.1$", "#5f8a5f"), ("pn", "PairNet", "#9b2c2c")):
        base = (v[f"{p}_asym"] < 0.1) & v[f"{p}_valid"]
        sg = s & (pm == M)
        qq = q & (v["has_pn"] if p == "pn" else True)
        hs, _ = np.histogram(v[f"{p}_mavg"][sg & base], bins); hq, _ = np.histogram(v[f"{p}_mavg"][qq & base], bins, weights=w[qq & base])
        ax.stairs(hs / sg.sum(), bins, color=c, lw=1.3, label=f"signal, {lab}")
        ax.stairs(hq / w[qq].sum(), bins, color=c, lw=0.9, ls="--", label=f"QCD, {lab}")
    ax.axvspan(0.85 * M, 1.1 * M, color="#9b2c2c", alpha=0.06, lw=0)
    ax.set_yscale("log"); ax.set_xlabel(r"$m_{\rm avg}$ of the chosen pairing [GeV]"); ax.set_ylabel("fraction of the sample per bin")
    ax.set_title(f"A < 0.1; signal {M} GeV (solid), QCD (dashed)", fontsize=7, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6.5, frameon=False)
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/mavg_pairings_m{M}.pdf"); fig.savefig(f"{out}/mavg_pairings_m{M}.png", dpi=170, bbox_inches="tight")
