"""Low-mass (200-1000 GeV) optimisation with dedicated MC, for an H_T-triggered (scouting) analysis.

Adds the vlm QCD sample (pTHat > 80 GeV, H_T > 400 GeV, symmetric-split filter 150-1150 GeV): the samples generated with
pTHat > 200 GeV miss 26-95% of the QCD cross section below H_T = 900 GeV, so each sample is used only where complete:
vlm from H_T = 400 GeV, old and lm from 900 GeV, ht from 1200 GeV. The analysis always requires H_T > max(400, trig).

QCD (r13A scores), merged by H_T region, each region averaged over the samples complete there with weights set by their
effective MC statistics in the analysis region (PairNet pairing A < 0.1, 300 < m_avg < 1150):
  old   data_r3 QCD c*8, c*9 (all H_T)                         big_vars_r13A_pn.npz + ht_all.npy
  ht    high-H_T sample c1400-1999 (H_T > 1.2 TeV)             big_vars_ht_r13A_ge1400.npz
  lm    low-mass sample (H_T > 700 GeV, symmetric-split filter) prod_lowmass/qcdlm_c*.npz
Signal: prod_lowmass/siglm_m{M}_incl (all) and _boost (H_T > 2.3 M at generation), combined above 2.3 M by event
counts; _power (power-shower ISR, H_T > 2.3 M) used only as a variation of the boosted signal efficiency.
Selections (per mass, PairNet pairing, A < 0.1, m_avg in [0.85, 1.1] M, optional trigger H_T threshold):
  P        P(signal) cut
  P+boost  P cut + H_T > c M + Delta of the weaker / stronger triplet > d M
  P+all    P+boost + D2_32 (worse triplet), D2_63_32, lowest jet pT of the six
  am+all   argmin |m1 - m2| pairing with all of H_T, Delta, Dalitz and pT cuts (classical baseline)
Thresholds by coordinate descent on one half of every sample, limit on the other half, both ways (geometric mean).
Tuning requires >= MIN_NEFF effective QCD MC events, >= MIN_NB raw QCD events and >= MIN_NS signal MC events.
usage: lowmass_study.py style out tag f_sys trig_ht [masses]
"""
import glob, json, math, os, sys, time
import h5py
import numpy as np

style, out, tag = sys.argv[1:4]
FSYS = float(sys.argv[4]); TRIG = float(sys.argv[5])
MASSES = [int(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 else [200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
TRIG = max(TRIG, 400.0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limits import s_up  # noqa: E402

L = 300.0
MIN_NEFF, MIN_NB, MIN_NS = 20, 20, 50
C = "/scratch/llee/pairnet/cuts_study"
LM = "/scratch/llee/pairnet/prod_lowmass"
COLS = ["m1", "m2", "delta1", "delta2", "d32_1", "d32_2", "d2_63_32", "pt_low6", "valid"]


def std(src, pre_pn, pre_am, extra):
    """Common column names: pn_* (PairNet r13A), am_* (argmin), psig, ht, pt6, plus extra."""
    d = {f"pn_{c}": src[f"{pre_pn}_{c}"] for c in COLS}
    d.update({f"am_{c}": src[f"{pre_am}_{c}"] for c in COLS})
    d.update(extra)
    return d


# ---------------- QCD samples
P3 = "/scratch/llee/pairnet/data_r3/prod"
fac = {f: h5py.File(f).attrs["sigma_mb"] / h5py.File(f).attrs["weight_sum"] * 1e12 for f in sorted(glob.glob(f"{P3}/qcd_c*.h5"))}
pnf = [f for f in fac if f[-4] in "89"]
v = np.load(f"{C}/big_vars_r13A_pn.npz")
full = np.load(f"{C}/big_vars.npz")
ht_full = np.load(f"{C}/ht_all.npy")[full["is_signal"] | np.isin(full["fileclass"], [8, 9])]
q = ~v["is_signal"]
old = std({k: v[k][q] for k in v.files if v[k].ndim and len(v[k]) == len(q)}, "pn", "am",
          {"psig": v["psig"][q], "ht": ht_full[q], "pt6": v["pt6_lead"][q],
           "wfb": v["w"][q] * np.mean([fac[f] for f in pnf]) / len(pnf)})
h = np.load(f"{C}/big_vars_ht_r13A_ge1400.npz")
ht = std(h, "pn", "am", {"psig": h["psig"], "ht": h["ht"], "pt6": h["pt6_lead"], "wfb": h["w"] * h["fac"] / int(h["n_files"])})
parts = []
lmfiles = sorted(glob.glob(f"{LM}/qcdlm_c*.npz"))
nlm = 0
for f in lmfiles:
    try:
        if time.time() - os.path.getmtime(f) < 120:
            continue  # may still be arriving
        a = dict(np.load(f))
    except Exception:  # noqa: BLE001  (truncated transfer)
        continue
    nlm += 1  # every readable file is one independent estimate, also when it kept no events
    if len(a["w"]) == 0:
        continue
    k = float(a["attr_sigma_mb"]) * 1e12 / float(a["attr_weight_sum"])
    parts.append(std(a, "r13A", "am", {"psig": a["r13A_psig"], "ht": a["ht"], "pt6": a["pt6_lead"], "wfb": a["w"] * k}))
lm = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None
vparts, nvlm = [], 0
for f in sorted(glob.glob(f"{LM}/qcdvlm_c*.npz")):
    try:
        if time.time() - os.path.getmtime(f) < 120:
            continue
        a = dict(np.load(f))
    except Exception:  # noqa: BLE001
        continue
    nvlm += 1
    if len(a["w"]) == 0:
        continue
    k = float(a["attr_sigma_mb"]) * 1e12 / float(a["attr_weight_sum"])
    vparts.append(std(a, "r13A", "am", {"psig": a["r13A_psig"], "ht": a["ht"], "pt6": a["pt6_lead"], "wfb": a["w"] * k}))
vlm = {k: np.concatenate([p[k] for p in vparts]) for k in vparts[0]} if vparts else None
if vlm is not None:
    vlm["wfb"] = vlm["wfb"] / nvlm
if lm is not None:
    lm["wfb"] = lm["wfb"] / nlm


def derive(d):
    for p in ("pn", "am"):
        d[f"{p}_mavg"] = 0.5 * (d[f"{p}_m1"] + d[f"{p}_m2"])
        d[f"{p}_asym"] = np.abs(d[f"{p}_m1"] - d[f"{p}_m2"]) / np.maximum(d[f"{p}_m1"] + d[f"{p}_m2"], 1e-9)
        d[f"{p}_delta_min"] = np.minimum(d[f"{p}_delta1"], d[f"{p}_delta2"]); d[f"{p}_delta_max"] = np.maximum(d[f"{p}_delta1"], d[f"{p}_delta2"])
        d[f"{p}_d32_max"] = np.maximum(d[f"{p}_d32_1"], d[f"{p}_d32_2"])
    d["half"] = np.arange(len(d["wfb"])) % 2
    return d


samples = {"old": derive(old), "ht": derive(ht)}
if lm is not None:
    samples["lm"] = derive(lm)
if vlm is not None:
    samples["vlm"] = derive(vlm)
REGIONS = [(400, 900, ["vlm"]), (900, 1200, ["old", "lm", "vlm"]), (1200, 1e9, ["old", "ht", "lm", "vlm"])]
COMPLETE = {"old": 900, "lm": 900, "ht": 1200, "vlm": 400}
neff = lambda x: x.sum() ** 2 / max((x ** 2).sum(), 1e-300)
merge = {}
for lo, hi, names in REGIONS:
    names = [n for n in names if n in samples]
    ne = {}
    for n in names:
        d = samples[n]
        r = (d["ht"] >= lo) & (d["ht"] < hi) & (d["pn_asym"] < 0.1) & (d["pn_mavg"] > 150) & (d["pn_mavg"] < 1150)
        ne[n] = neff(d["wfb"][r])
    tot = sum(ne.values())
    for n in names:
        d = samples[n]
        r = (d["ht"] >= lo) & (d["ht"] < hi)
        d["wfb"] = np.where(r, d["wfb"] * ne[n] / tot, d["wfb"])
    merge[f"{lo}-{hi}"] = {n: {"neff": ne[n], "alpha": ne[n] / tot} for n in names}
for n, d in samples.items():  # events outside a sample's complete region carry no weight there
    d["wfb"] = np.where(d["ht"] >= COMPLETE[n], d["wfb"], 0.0)
QCD = {k: np.concatenate([samples[n][k] for n in samples]) for k in samples["old"]}
QCD["half"] = np.arange(len(QCD["wfb"])) % 2
print("QCD merge:", json.dumps(merge, default=float), "lm files", nlm, "vlm files", nvlm, flush=True)


# ---------------- signal
def load_sig(M, kind):
    fs = sorted(glob.glob(f"{LM}/siglm_m{M}_{kind}_c*.npz"))
    if not fs:
        return None
    ps, ntried = [], 0
    for f in fs:
        a = np.load(f)
        ps.append(std(a, "r13A", "am", {"psig": a["r13A_psig"], "ht": a["ht"], "pt6": a["pt6_lead"]}))
        ntried += float(a["attr_n_tried"])
    d = {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}
    d["wfb"] = np.full(len(d["psig"]), 1.0 / ntried)  # fraction of produced events
    return derive(d)


VARS = {"psig": ("psig", ">", None, False), "ht": ("ht", ">", [0, 1.6, 1.8, 2.0, 2.2, 2.3, 2.5, 2.7, 3.0, 3.5, 4.0], True),
        "delta_max": ("delta_max", ">", [-1e9, -.2, -.1, 0, .1, .2, .3, .4, .5, .7, 1.0, 1.3], True),
        "delta_min": ("delta_min", ">", [-1e9, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .5], True),
        "d32_max": ("d32_max", "<", [1e9, .5, .4, .3, .25, .2, .15, .1], False),
        "d2_63_32": ("d2_63_32", "<", [1e9, 2.5, 2, 1.5, 1.25, 1, .8, .6], False),
        "pt_low6": ("pt_low6", ">", [0, .05, .07, .09, .11, .13, .15, .18, .22, .26], True)}
CONF = {"P": ("pn", ["psig"]), "P+boost": ("pn", ["psig", "ht", "delta_max", "delta_min"]),
        "P+all": ("pn", list(VARS)), "am+all": ("am", [k for k in VARS if k != "psig"])}


class Cand:
    def __init__(self, d, p, M, sig_boost=None):
        sel = (d[f"{p}_asym"] < 0.1) & (d[f"{p}_mavg"] > 0.85 * M) & (d[f"{p}_mavg"] < 1.1 * M) & d[f"{p}_valid"] & (d["ht"] > TRIG) & (d["wfb"] > 0)
        self.x = {n: (d[k] if k in ("psig", "ht") else d[f"{p}_{k}"])[sel] / (M if rel else 1) for n, (k, _, _, rel) in VARS.items()}
        self.w = d["wfb"][sel]; self.h = d["half"][sel]


def passes(c, cuts):
    ok = np.ones(len(c.w), bool)
    for n, t in cuts.items():
        ok &= (c.x[n] >= t) if n == "psig" else (c.x[n] > t) if VARS[n][1] == ">" else (c.x[n] < t)
    return ok


def evaluate(S, B, cuts, h):
    ks = passes(S, cuts) & (S.h == h); kb = passes(B, cuts) & (B.h == h)
    eS = 2 * S.w[ks].sum(); xB = 2 * B.w[kb].sum()
    lim = s_up(xB * L, FSYS) / (L * eS) if eS > 0 else np.inf
    return lim, int(ks.sum()), int(kb.sum()), neff(B.w[kb]), eS, xB


def tune(S, B, free, h, rounds=3):
    grids = {}
    for n in free:
        if n == "psig":
            ps = S.x["psig"][S.h == h]
            grids[n] = [-1.0] + list(np.quantile(ps, 1 - np.concatenate([np.linspace(0.96, 0.2, 20), np.geomspace(0.15, 0.005, 14)]))) if len(ps) else [-1.0]
        else:
            grids[n] = VARS[n][2]
    cuts = {n: g[0] for n, g in grids.items()}
    best = evaluate(S, B, cuts, h)[0]
    for _ in range(rounds):
        changed = False
        for n, g in grids.items():
            for t in g:
                trial = {**cuts, n: t}
                lim, ns, nb, ne, _, _ = evaluate(S, B, trial, h)
                if ns >= MIN_NS and nb >= MIN_NB and ne >= MIN_NEFF and lim < best * (1 - 1e-9):
                    best, cuts, changed = lim, trial, True
        if not changed:
            break
    return cuts


def merged_signal(M):
    inc, bst = load_sig(M, "incl"), load_sig(M, "boost")
    if inc is None:
        return None
    if bst is not None:  # above 2.3 M both samples estimate the same efficiency: average by event counts
        hb = 2.3 * M
        n_i = (inc["ht"] > hb).sum(); n_b = len(bst["ht"])
        a = n_i / (n_i + n_b)
        inc["wfb"] = np.where(inc["ht"] > hb, inc["wfb"] * a, inc["wfb"])
        bst["wfb"] = bst["wfb"] * (1 - a)
        d = {k: np.concatenate([inc[k], bst[k]]) for k in inc if k != "half"}
        d["half"] = np.arange(len(d["wfb"])) % 2
        return d
    return inc


res = {}
for M in MASSES:
    sig = merged_signal(M)
    if sig is None:
        continue
    r = {}
    for name, (p, free) in CONF.items():
        S, B = Cand(sig, p, M), Cand(QCD, p, M)
        lims, info = [], []
        for h in (0, 1):
            cuts = tune(S, B, free, h)
            lim, ns, nb, ne, eS, xB = evaluate(S, B, cuts, 1 - h)
            lims.append(lim); info.append({"cuts": {k: float(x) for k, x in cuts.items()}, "nS": ns, "nB": nb, "neffB": float(ne), "effS": float(eS), "xsB_fb": float(xB)})
        r[name] = {"limit_fb": float(np.exp(np.mean(np.log(lims)))), "halves": info}
    # ISR-model variation: boosted signal efficiency of the tuned P+boost selection with the power-shower sample
    pw = load_sig(M, "power"); bs = load_sig(M, "boost")
    if pw is not None and bs is not None:
        effs = []
        for d in (bs, pw):
            c = Cand(d, "pn", M)
            effs.append(np.mean([c.w[passes(c, hh["cuts"])].sum() for hh in r["P+boost"]["halves"]]))
        r["power_over_default_boost_eff"] = float(effs[1] / effs[0]) if effs[0] > 0 else None
    res[M] = r
    print(f"M={M} trig>{TRIG:g} f={FSYS:g}: " + "  ".join(f"{k} {x['limit_fb']:.3g} (nS {min(hh['nS'] for hh in x['halves'])}, neffB {min(hh['neffB'] for hh in x['halves']):.0f})"
                                                     for k, x in r.items() if isinstance(x, dict)) + f"  power/default {r.get('power_over_default_boost_eff')}", flush=True)
json.dump({"merge": merge, "results": res, "trig": TRIG, "fsys": FSYS, "n_vlm_files": nvlm}, open(f"{out}/lowmass_{tag}_trig{TRIG:g}_f{FSYS:g}.json", "w"), indent=1, default=float)
