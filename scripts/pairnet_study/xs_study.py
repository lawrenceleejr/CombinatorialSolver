"""Rough expected 95% CL cross-section limits on pp -> gluino pair (-> uds uds), observation = background.

Signal: Pythia LO sigma per mass (file attrs), x 6-jet filter efficiency (written / tried), x selection efficiency on
the test files. QCD: per-event sigma = normweight * sigma / weight_sum, averaged over the test files.
Selections, each in the m_avg window [0.85, 1.1] M: argmin|m1-m2| pairing + A < 0.1; PairNet + A < 0.1;
PairNet + A < 0.1 + P(signal) cut chosen per mass for the best expected limit (>= 10 QCD MC events kept).
Median expected CLs-style limit from the Asimov counting formula 2[s - B ln(1 + s/B)] = 1.96^2, floored at 3 events.
No systematic uncertainties."""
import glob, json, math, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

style, npz, out = sys.argv[1:4]
sys.path.insert(0, style); sys.path.insert(0, "/scratch/llee/CombinatorialSolver")
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
from src.pairdata import load_events  # noqa: E402
from src.pairnet import to_cartesian  # noqa: E402
from src.combinatorics import enumerate_assignments  # noqa: E402
import torch  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

P = "/scratch/llee/pairnet/data_r3/prod"
MB_TO_FB = 1e12
LUMIS = (300.0, 140.0)  # fb^-1
MASSES = list(range(500, 2401, 100))

# ---------------- normalisation
qfiles = sorted(glob.glob(f"{P}/qcd_c*9.h5"))
kq = np.mean([h5py.File(f).attrs["sigma_mb"] / h5py.File(f).attrs["weight_sum"] for f in qfiles]) * MB_TO_FB / len(qfiles)
sig_xs, sig_filt = {}, {}
for M in MASSES:
    fs = sorted(glob.glob(f"/scratch/llee/pairnet/prod/sig_m{M}_c*.h5"))
    sig_xs[M] = np.mean([h5py.File(f).attrs["sigma_mb"] for f in fs]) * MB_TO_FB
    sig_filt[M] = np.mean([h5py.File(f).attrs["n_rows"] / h5py.File(f).attrs["n_tried"] if "n_rows" in h5py.File(f).attrs
                           else 10000 / h5py.File(f).attrs["n_tried"] for f in fs])


def s_up(B):
    """Median expected 95% CL upper limit on signal events for background B (Asimov, no systematics)."""
    if B <= 0:
        return 3.0
    f = lambda s: 2 * (s - B * math.log1p(s / B)) - 1.96 ** 2
    return max(brentq(f, 1e-9, 1e3 + 50 * math.sqrt(B) + 10 * B), 3.0)


# ---------------- PairNet (r8B) test predictions
e = np.load(npz); s = e["is_signal"]; q = ~s; w = e["w"]
pn = {"A": e["asym"], "mavg": e["mavg"], "p": e["psig"]}

# ---------------- classical argmin|m1-m2| on the same test files
fs_sig = sorted(glob.glob(f"{P}/sig_*_c*9.h5"))
d = [load_events(fs_sig, 7, 3, True, None, min_jets=6), load_events(qfiles, 7, 3, False, None, min_jets=6)]
pt, eta, phi, m, mask = (np.concatenate([x[k] for x in d]) for k in ("pt", "eta", "phi", "m", "mask"))
c_is_sig = np.concatenate([x["is_signal"] for x in d]); c_pm = np.concatenate([x["pmass"] for x in d]); c_w = np.concatenate([x["w"] for x in d])
p4 = (to_cartesian(*(torch.from_numpy(v).float() for v in (pt, eta, phi, m))) * torch.from_numpy(mask).unsqueeze(-1)).numpy().astype(np.float64)
asg = enumerate_assignments(7); g1 = np.array([a[1] for a in asg]); g2 = np.array([a[2] for a in asg])
mm = lambda v: np.sqrt(np.maximum(v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1), 0))
m1 = mm(p4[:, g1].sum(2)); m2 = mm(p4[:, g2].sum(2))
ok = mask[:, g1].all(-1) & mask[:, g2].all(-1)
b = np.where(ok, np.abs(m1 - m2), np.inf).argmin(1); ar = np.arange(len(b))
cl = {"A": np.abs(m1[ar, b] - m2[ar, b]) / (m1[ar, b] + m2[ar, b]), "mavg": 0.5 * (m1[ar, b] + m2[ar, b])}
print("QCD 6+ jets cross section [fb]: PairNet set %.3e, classical set %.3e" % (kq * w[q].sum(), kq * c_w[~c_is_sig].sum()))

res = {"lumi": LUMIS, "sigma_LO_fb": sig_xs, "filter_eff": sig_filt, "rows": {}}
for M in MASSES:
    row = {}
    for name, S_sel, B_sel, nB in (
            ("argmin |m1-m2|, A<0.1", None, None, None), ("PairNet, A<0.1", None, None, None), ("PairNet, A<0.1 + P cut", None, None, None)):
        pass
    # classical
    cs = c_is_sig & (c_pm == M); win = (cl["mavg"] > 0.85 * M) & (cl["mavg"] < 1.1 * M); sel = (cl["A"] < 0.1) & win
    eS = sel[cs].mean() * sig_filt[M]; xB = kq * c_w[~c_is_sig & sel].sum(); nB = int((~c_is_sig & sel).sum())
    row["argmin"] = {"effS": float(eS), "xsB_fb": float(xB), "nB": nB}
    # PairNet
    ps = s & (e["pmass"] == M); win = (pn["mavg"] > 0.85 * M) & (pn["mavg"] < 1.1 * M); base = (pn["A"] < 0.1) & win
    eS = base[ps].mean() * sig_filt[M]; xB = kq * w[q & base].sum()
    row["pairnet_A"] = {"effS": float(eS), "xsB_fb": float(xB), "nB": int((q & base).sum())}
    best = None
    for eff in np.linspace(0.2, 1.0, 41):
        t = np.quantile(pn["p"][ps], 1 - eff) if eff < 1 else -1
        sel = base & (pn["p"] >= t); nB = int((q & sel).sum())
        if nB < 10:
            continue
        eS = sel[ps].mean() * sig_filt[M]; xB = kq * w[q & sel].sum()
        lim = s_up(xB * LUMIS[0]) / (LUMIS[0] * eS)
        if best is None or lim < best[0]:
            best = (lim, {"effS": float(eS), "xsB_fb": float(xB), "nB": nB, "p_cut_sig_eff": float(eff)})
    row["pairnet_AP"] = best[1]
    for k in ("argmin", "pairnet_A", "pairnet_AP"):
        r = row[k]
        for L in LUMIS:
            r[f"limit_fb_{int(L)}"] = s_up(r["xsB_fb"] * L) / (L * r["effS"]) if r["effS"] > 0 else float("inf")
    res["rows"][M] = row
    print(f"M={M:4d}  sigma_LO={sig_xs[M]:9.2f} fb | limit@300/fb [fb]: argmin {row['argmin']['limit_fb_300']:9.1f}  PairNet A {row['pairnet_A']['limit_fb_300']:8.2f}  PairNet A+P {row['pairnet_AP']['limit_fb_300']:8.2f} | effS {row['pairnet_AP']['effS']:.3f} B {row['pairnet_AP']['xsB_fb']*300:.0f} ev (nMC {row['pairnet_AP']['nB']})")


def reach(key, L):
    """Largest mass where the expected limit is below the LO cross section (log-linear interpolation)."""
    last = None
    for M in MASSES:
        r = res["rows"][M][key][f"limit_fb_{int(L)}"]
        if r < sig_xs[M]:
            last = M
    if last is None or last == MASSES[-1]:
        return last
    M2 = last + 100
    f = lambda Mx: math.log(np.interp(Mx, [last, M2], [res["rows"][last][key][f"limit_fb_{int(L)}"], res["rows"][M2][key][f"limit_fb_{int(L)}"]])) - math.log(np.interp(Mx, [last, M2], [sig_xs[last], sig_xs[M2]]))
    try:
        return brentq(f, last, M2)
    except ValueError:
        return last
res["reach"] = {k: {int(L): reach(k, L) for L in LUMIS} for k in ("argmin", "pairnet_A", "pairnet_AP")}
print("expected mass reach [GeV]:", res["reach"])
json.dump(res, open(f"{out}/xs_study.json", "w"), indent=1, default=float)

# ---------------- plot
fig, ax = plt.subplots(figsize=(5.0, 3.5))
ax.plot(MASSES, [sig_xs[M] for M in MASSES], color="0.1", lw=1.6)
ax.annotate(r"$\sigma_{\rm LO}(pp\to\tilde g\tilde g)$, 13.6 TeV", (MASSES[-1], sig_xs[MASSES[-1]]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5)
for key, lab, col, ls in (("argmin", r"argmin $|m_1-m_2|$, $A<0.1$", "0.5", "--"), ("pairnet_A", r"PairNet, $A<0.1$", "#c46a5f", "--"),
                          ("pairnet_AP", r"PairNet, $A<0.1$ + $P$(signal) cut", "#9b2c2c", "-")):
    y = [res["rows"][M][key]["limit_fb_300"] for M in MASSES]
    ax.plot(MASSES, y, color=col, lw=1.3, ls=ls)
    ax.annotate(lab, (MASSES[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color=col)
ax.set_yscale("log"); ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"$\sigma$ [fb]")
ax.set_title(r"expected 95% CL limits, 300 fb$^{-1}$, observation = background, no systematics", fontsize=7.5, loc="left")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/xs_limits.pdf"); fig.savefig(f"{out}/xs_limits.png", dpi=170, bbox_inches="tight")
