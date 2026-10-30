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
# FSYS_JSON: per-mass relative background uncertainty {M: {"f_corr": ...}} (e.g. CMS Run-2 fit uncertainty); overrides FSYS
FSYS_TAB = json.load(open(os.environ["FSYS_JSON"])) if os.environ.get("FSYS_JSON") else None
FSYS_KEY = os.environ.get("FSYS_KEY", "f_corr")
DUMP = os.environ.get("DUMP_SPECTRA")  # directory: save m_avg spectra after the chosen cuts (no window) per mass
MASSES = [int(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 else [200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
TRIG = max(TRIG, 400.0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limits import s_up  # noqa: E402

L = 300.0
MIN_NEFF, MIN_NB, MIN_NS = 20, 20, 50
C = "/scratch/llee/pairnet/cuts_study"
LM = os.environ.get("LMDIR", "/scratch/llee/pairnet/prod_lowmass")
MODEL = os.environ.get("MODEL", "r13A")  # column prefix of the PairNet model in the scored npz files
OLD_NPZ = os.environ.get("OLD_NPZ", f"{C}/big_vars_r13A_pn.npz")
HT_NPZ_FILE = os.environ.get("HT_NPZ_FILE", f"{C}/big_vars_ht_r13A_ge1400.npz")
USE_LM = not os.environ.get("NO_LM")
SIG_CLASSES = os.environ.get("SIG_CLASSES")  # e.g. "16,17,18,19": only these file indices (held out from training)
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
v = np.load(OLD_NPZ)
full = np.load(f"{C}/big_vars.npz")
ht_full = np.load(f"{C}/ht_all.npy")[full["is_signal"] | np.isin(full["fileclass"], [8, 9])]
q = ~v["is_signal"]
old = std({k: v[k][q] for k in v.files if v[k].ndim and len(v[k]) == len(q)}, "pn", "am",
          {"psig": v["psig"][q], "ht": ht_full[q], "pt6": v["pt6_lead"][q],
           "wfb": v["w"][q] * np.mean([fac[f] for f in pnf]) / len(pnf)})
h = np.load(HT_NPZ_FILE)
ht = std(h, "pn", "am", {"psig": h["psig"], "ht": h["ht"], "pt6": h["pt6_lead"], "wfb": h["w"] * h["fac"] / int(h["n_files"])})
parts = []
lmfiles = sorted(glob.glob(f"{LM}/qcdlm_c*.npz")) if USE_LM else []
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
    parts.append(std(a, MODEL, "am", {"psig": a[f"{MODEL}_psig"], "ht": a["ht"], "pt6": a["pt6_lead"], "wfb": a["w"] * k}))
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
    vparts.append(std(a, MODEL, "am", {"psig": a[f"{MODEL}_psig"], "ht": a["ht"], "pt6": a["pt6_lead"], "wfb": a["w"] * k}))
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

# Decorrelated P(signal): u = QCD fraction (cross-section weighted) with a higher P at the event's m_avg, from QCD with
# A < 0.1 and H_T > trig in 40 log-spaced m_avg bins. Cutting u < eps keeps the same QCD fraction at every m_avg.
DDT_EDGES = np.exp(np.linspace(np.log(120), np.log(3000), 41))
DDT_MAP = []
if os.environ.get("DDT"):
    qb = (QCD["pn_asym"] < 0.1) & QCD["pn_valid"] & (QCD["ht"] > max(TRIG, 400)) & (QCD["wfb"] > 0)
    for i in range(len(DDT_EDGES) - 1):
        k = qb & (QCD["pn_mavg"] >= DDT_EDGES[i]) & (QCD["pn_mavg"] < DDT_EDGES[i + 1])
        if k.sum() < 20:
            DDT_MAP.append(None); continue
        o = np.argsort(-QCD["psig"][k]); ps = QCD["psig"][k][o]; cw = np.cumsum(QCD["wfb"][k][o]); cw /= cw[-1]
        DDT_MAP.append((ps[::-1], cw[::-1]))


def ddt_u(d):
    u = np.ones(len(d["psig"]), np.float32)
    if not DDT_MAP:
        return u
    b = np.digitize(d["pn_mavg"], DDT_EDGES) - 1
    for i, mp in enumerate(DDT_MAP):
        if mp is None:
            continue
        k = b == i
        if k.any():
            u[k] = np.interp(d["psig"][k], mp[0], mp[1])  # fraction of QCD with P >= this P
    return u


if os.environ.get("DDT"):
    QCD["u"] = ddt_u(QCD)


# ---------------- signal
def load_sig(M, kind):
    fs = sorted(glob.glob(f"{LM}/siglm_m{M}_{kind}_c*.npz"))
    if SIG_CLASSES and kind != "power":
        keep = {int(x) for x in SIG_CLASSES.split(",")}
        fs = [f for f in fs if int(f.rsplit("_c", 1)[1][:-4]) in keep]
    if not fs:
        return None
    ps, ntried = [], 0
    for f in fs:
        a = np.load(f)
        ps.append(std(a, MODEL, "am", {"psig": a[f"{MODEL}_psig"], "ht": a["ht"], "pt6": a["pt6_lead"]}))
        ntried += float(a["attr_n_tried"])
    d = {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}
    d["wfb"] = np.full(len(d["psig"]), 1.0 / ntried)  # fraction of produced events
    return derive(d)


VARS = {"psig": ("psig", ">", None, False),
        "u": ("u", "<", [1.0, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 5e-4, 3e-4, 2e-4, 1e-4], False), "ht": ("ht", ">", [0, 1.6, 1.8, 2.0, 2.2, 2.3, 2.5, 2.7, 3.0, 3.5, 4.0], True),
        "delta_max": ("delta_max", ">", [-1e9, -.2, -.1, 0, .1, .2, .3, .4, .5, .7, 1.0, 1.3], True),
        "delta_min": ("delta_min", ">", [-1e9, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .5], True),
        "d32_max": ("d32_max", "<", [1e9, .5, .4, .3, .25, .2, .15, .1], False),
        "d2_63_32": ("d2_63_32", "<", [1e9, 2.5, 2, 1.5, 1.25, 1, .8, .6], False),
        "pt_low6": ("pt_low6", ">", [0, .05, .07, .09, .11, .13, .15, .18, .22, .26], True)}
CONF = {"P": ("pn", ["psig"]), "P+boost": ("pn", ["psig", "ht", "delta_max", "delta_min"]),
        "P+all": ("pn", [k for k in VARS if k != "u"]), "am+all": ("am", [k for k in VARS if k not in ("psig", "u")])}
if os.environ.get("DDT"):
    CONF.update({"DDT": ("pn", ["u"]), "DDT+boost": ("pn", ["u", "ht", "delta_max", "delta_min"])})


class Cand:
    def __init__(self, d, p, M, sig_boost=None, wide=False):
        lo, hi = (0.4, 1.6) if wide else (0.85, 1.1)
        sel = (d[f"{p}_asym"] < 0.1) & (d[f"{p}_mavg"] > lo * M) & (d[f"{p}_mavg"] < hi * M) & d[f"{p}_valid"] & (d["ht"] > TRIG) & (d["wfb"] > 0)
        self.x = {n: (d[k] if k in ("psig", "ht", "u") else d[f"{p}_{k}"])[sel] / (M if rel else 1) for n, (k, _, _, rel) in VARS.items() if k in d or f"{p}_{k}" in d}
        self.w = d["wfb"][sel]; self.h = d["half"][sel]; self.m = d[f"{p}_mavg"][sel] / M


# PEAK_MAX: require the QCD m_avg spectrum (after the cuts, 0.4-1.6 M) to peak at or below PEAK_MAX * M, i.e. the window
# [0.85, 1.1] M sits on the falling side, and the signal m_avg peak to lie inside the window. Cuts violating it are rejected.
PEAK_MAX = float(os.environ["PEAK_MAX"]) if os.environ.get("PEAK_MAX") else None
SHAPE_BINS = np.arange(0.4, 1.6001, 0.05)


def shape(Sw, Bw, cuts, h):
    """(QCD peak / M, signal peak / M) for the cuts on half h, from 3-bin-smoothed weighted histograms."""
    def peak(c, wt):
        k = passes(c, cuts) & (c.h == h)
        y, _ = np.histogram(c.m[k], SHAPE_BINS, weights=wt[k])
        if y.sum() <= 0:
            return np.nan
        y = np.convolve(y, np.ones(3) / 3, mode="same")
        return 0.5 * (SHAPE_BINS[1:] + SHAPE_BINS[:-1])[np.argmax(y)]
    return peak(Bw, Bw.w), peak(Sw, Sw.w)


def shape_ok(Sw, Bw, cuts, h):
    if PEAK_MAX is None:
        return True
    qp, sp = shape(Sw, Bw, cuts, h)
    return (qp <= PEAK_MAX) and (0.85 <= sp <= 1.1)


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


def tune(S, B, free, h, rounds=3, Sw=None, Bw=None):
    grids = {}
    for n in free:
        if n == "psig":
            ps = S.x["psig"][S.h == h]
            grids[n] = [-1.0] + list(np.quantile(ps, 1 - np.concatenate([np.linspace(0.96, 0.2, 20), np.geomspace(0.15, 0.005, 14)]))) if len(ps) else [-1.0]
        else:
            grids[n] = VARS[n][2]
    cuts = {n: g[0] for n, g in grids.items()}
    best = evaluate(S, B, cuts, h)[0] if (Sw is None or shape_ok(Sw, Bw, cuts, h)) else np.inf
    for _ in range(rounds):
        changed = False
        for n, g in grids.items():
            for t in g:
                trial = {**cuts, n: t}
                lim, ns, nb, ne, _, _ = evaluate(S, B, trial, h)
                if ns >= MIN_NS and nb >= MIN_NB and ne >= MIN_NEFF and lim < best * (1 - 1e-9) and (Sw is None or shape_ok(Sw, Bw, trial, h)):
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
        if os.environ.get("DDT"):
            d["u"] = ddt_u(d)
        return d
    if os.environ.get("DDT"):
        inc["u"] = ddt_u(inc)
    return inc


res = {}
for M in MASSES:
    sig = merged_signal(M)
    if sig is None:
        continue
    if FSYS_TAB:
        tm = np.array(sorted(int(k) for k in FSYS_TAB)); tf = np.array([FSYS_TAB[str(k)][FSYS_KEY] for k in tm])
        FSYS = float(np.exp(np.interp(M, tm, np.log(tf))))
    r = {}
    for name, (p, free) in CONF.items():
        S, B = Cand(sig, p, M), Cand(QCD, p, M)
        Sw, Bw = (Cand(sig, p, M, wide=True), Cand(QCD, p, M, wide=True)) if PEAK_MAX is not None else (None, None)
        lims, info = [], []
        for h in (0, 1):
            cuts = tune(S, B, free, h, Sw=Sw, Bw=Bw)
            feasible = Sw is None or shape_ok(Sw, Bw, cuts, h)
            lim, ns, nb, ne, eS, xB = evaluate(S, B, cuts, 1 - h)
            if not feasible:
                lim = np.inf
            qp, sp = shape(Sw, Bw, cuts, 1 - h) if Sw is not None else (np.nan, np.nan)
            lims.append(lim); info.append({"cuts": {k: float(x) for k, x in cuts.items()}, "nS": ns, "nB": nb, "neffB": float(ne), "effS": float(eS), "xsB_fb": float(xB),
                                           "feasible": bool(feasible), "qcd_peak_over_M": float(qp), "sig_peak_over_M": float(sp)})
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
    if DUMP:
        os.makedirs(DUMP, exist_ok=True)
        best = min([k for k in r if k != "am+all" and isinstance(r[k], dict) and "limit_fb" in r[k]], key=lambda k: r[k]["limit_fb"])
        cuts = r[best]["halves"][0]["cuts"]
        dumped = {}
        for name, d, p in (("sig", sig, "pn"), ("qcd", QCD, "pn")):
            base = (d[f"{p}_asym"] < 0.1) & d[f"{p}_valid"] & (d["ht"] > TRIG) & (d["wfb"] > 0) & (d[f"{p}_mavg"] > 0.4 * M) & (d[f"{p}_mavg"] < 2.0 * M)
            ok = base.copy()
            for n, t in cuts.items():
                k, direction, _, rel = VARS[n]
                xv = (d[k] if k in ("psig", "ht", "u") else d[f"{p}_{k}"]) / (M if rel else 1)
                ok &= (xv >= t) if n == "psig" else (xv > t) if direction == ">" else (xv < t)
            dumped[f"{name}_mavg"] = d[f"{p}_mavg"][ok]; dumped[f"{name}_w"] = d["wfb"][ok]
        np.savez(f"{DUMP}/spec_{MODEL}_m{M}_trig{TRIG:g}_f{FSYS:g}.npz", sel=best, cuts=json.dumps(cuts), fsys=FSYS, **dumped)
    print(f"M={M} trig>{TRIG:g} f={FSYS:g}: " + "  ".join(f"{k} {x['limit_fb']:.3g} (nS {min(hh['nS'] for hh in x['halves'])}, neffB {min(hh['neffB'] for hh in x['halves']):.0f})"
                                                     for k, x in r.items() if isinstance(x, dict)) + f"  power/default {r.get('power_over_default_boost_eff')}", flush=True)
FTAG = os.environ.get("FTAG", f"f{FSYS:g}")
json.dump({"merge": merge, "results": res, "trig": TRIG, "fsys": FSYS, "fsys_table": bool(FSYS_TAB), "n_vlm_files": nvlm, "model": MODEL}, open(f"{out}/lowmass_{tag}_{MODEL}_trig{TRIG:g}_{FTAG}.json", "w"), indent=1, default=float)
