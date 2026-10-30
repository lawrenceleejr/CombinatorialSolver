"""Compare PairNet models on identical events: per-mass tuned P(signal) cut after A < 0.1 and the m_avg window.

Each model: <big>.npz (big_vars.py format; signal test files + QCD files c*8, c*9 scored with that model's ONNX) and
<ht>.npz (big_vars_ht.py format; high-HT QCD files >= c1400, i.e. not in r13A's training set). QCD merged above
HT = 1.2 TeV by effective statistics (as qcdmerge.py); HT of the standard sample from ht_all.npy (big_vars order).
Thresholds tuned on one half, evaluated on the other (both ways); >= 10 QCD MC events in the tuning half.
usage: compare_models.py style ht_all.npy xs_study.json full_big_vars.npz out name=big.npz:ht.npz ...
"""
import glob, json, os, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, ht_all, xs_json, full_big, out = sys.argv[1:6]
models = [a.split("=", 1) for a in sys.argv[6:]]
sys.path.insert(0, style); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
from limits import s_up  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
L = 300.0
filt = {int(k): x for k, x in json.load(open(xs_json))["filter_eff"].items()}
P = "/scratch/llee/pairnet/data_r3/prod"
fac = {f: h5py.File(f).attrs["sigma_mb"] / h5py.File(f).attrs["weight_sum"] * 1e12 for f in sorted(glob.glob(f"{P}/qcd_c*.h5"))}
pnf = [f for f in fac if f[-4] in "89"]
k_old = np.mean([fac[f] for f in pnf]) / len(pnf)
full = np.load(full_big)
ht_full = np.load(ht_all)
keep = full["is_signal"] | np.isin(full["fileclass"], [8, 9])  # rows of a PN-only big_vars file
HT_SUB = ht_full[keep]


def load(big, htn):
    v = np.load(big); h = np.load(htn)
    s = v["is_signal"]; q = ~s
    assert len(s) == len(HT_SUB), (len(s), len(HT_SUB))
    neff = lambda x: x.sum() ** 2 / max((x ** 2).sum(), 1e-300)
    wo = np.where(q, v["w"] * k_old, 0.0)
    wn = h["w"] * h["fac"] / int(h["n_files"])
    hi = q & (HT_SUB > 1200)
    a = neff(wo[hi]); b = neff(wn); al = a / (a + b)
    wo = np.where(hi, wo * al, wo); wn = wn * (1 - al)
    cat = lambda k: np.concatenate([v[k], h[k]])
    d = {"m1": cat("pn_m1"), "m2": cat("pn_m2"), "psig": cat("psig"), "w": np.concatenate([wo, wn]),
         "sig": np.concatenate([s, np.zeros(len(wn), bool)]), "pm": np.concatenate([v["pmass"], np.zeros(len(wn))])}
    d["mavg"] = 0.5 * (d["m1"] + d["m2"]); d["A"] = np.abs(d["m1"] - d["m2"]) / np.maximum(d["m1"] + d["m2"], 1e-9)
    d["half"] = np.arange(len(d["w"])) % 2
    return d, {"alpha_old": al, "neff_old": a, "neff_new": b}


res, infos = {}, {}
for name, spec in models:
    big, htn = spec.split(":")
    d, infos[name] = load(big, htn)
    res[name] = {}
    for M in range(500, 2401, 100):
        sM = d["sig"] & (d["pm"] == M)
        base = (d["A"] < 0.1) & (d["mavg"] > 0.85 * M) & (d["mavg"] < 1.1 * M)
        FT = json.load(open(os.environ["FSYS_JSON"])) if os.environ.get("FSYS_JSON") else None
        fr = None
        if FT:
            tm_ = np.array(sorted(int(k) for k in FT)); tf_ = np.array([FT[str(k)]["f_corr"] for k in tm_])
            fr = float(np.exp(np.interp(M, tm_, np.log(tf_))))
        lims = {0.0: [], 0.05: []} | ({"run2": []} if fr else {})
        for f in lims:
            for h in (0, 1):
                ps = d["psig"][sM & base & (d["half"] == h)]
                grid = [-1.0] + list(np.quantile(ps, 1 - np.concatenate([np.linspace(0.96, 0.2, 20), np.geomspace(0.15, 0.01, 10)])))

                def lim(t, hh):
                    sel = base & (d["psig"] >= t) & (d["half"] == hh)
                    eS = (sel & sM).sum() / (sM & (d["half"] == hh)).sum() * filt[M]
                    qq = sel & ~d["sig"]
                    B = 2 * d["w"][qq].sum() * L
                    return (s_up(B, fr if f == "run2" else f) / (L * eS) if eS > 0 else np.inf), int(qq.sum())
                cand = [(lim(t, h)[0], t) for t in grid if lim(t, h)[1] >= 10]
                t = min(cand)[1] if cand else -1.0
                lims[f].append(lim(t, 1 - h)[0])
        res[name][M] = {(f"f{f:g}" if f != "run2" else "frun2"): float(np.exp(np.mean(np.log(x)))) for f, x in lims.items()}
    print(name, json.dumps(infos[name]), flush=True)
names = [n for n, _ in models]
for M in sorted(res[names[0]]):
    print(M, "  ".join(f"{n}: {res[n][M]['f0']:.3g} / {res[n][M]['f0.05']:.3g}" for n in names),
          "  ratio %s/%s: %.2f / %.2f" % (names[1], names[0], res[names[1]][M]["f0"] / res[names[0]][M]["f0"],
                                          res[names[1]][M]["f0.05"] / res[names[0]][M]["f0.05"]) if len(names) > 1 else "")
json.dump({"limits": res, "merge": infos}, open(f"{out}/compare_models.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(4.6, 3.0))
Ms = np.array(sorted(res[names[0]]))
for n, c in zip(names[1:], ("#9b2c2c", "#4a87a8", "#5f8a5f")):
    for f, ls in (("f0", "-"), ("f0.05", "--")):
        y = np.array([res[n][m][f] / res[names[0]][m][f] for m in Ms])
        ax.plot(Ms, y, color=c, ls=ls, lw=1.4, label=f"{n} / {names[0]}, " + ("no syst." if f == "f0" else "5% on B"))
ax.axhline(1, color="0.6", lw=0.6)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel("expected limit ratio")
ax.set_title("same events, PairNet + tuned P cut; QCD not used in either training", fontsize=6.5, loc="left")
ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/compare_models.pdf"); fig.savefig(f"{out}/compare_models.png", dpi=170, bbox_inches="tight")
