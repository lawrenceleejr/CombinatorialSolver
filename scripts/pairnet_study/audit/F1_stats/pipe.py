"""Faithful re-implementation of lowmass_study4.py and fisher_high.py (COUNT_F path) on the local cache (data/*.npz),
with per-event sample labels, tuning-half bookkeeping and an optional bootstrap of the MC events.
The selection / tuning code is copied from repo_copy/lowmass_study4.py; only the I/O and bookkeeping differ."""
import glob, itertools, json, os, sys
import h5py
import numpy as np
D = "/scratch/llee/pairnet/audit/F1_stats"
sys.path.insert(0, f"{D}/repo_copy"); sys.dont_write_bytecode = True
from limits import s_up, mu_up_multi, mu_up_multi_corr  # noqa: E402
import tailmerge as TM  # noqa: E402

L = 300.0
MIN_NEFF, MIN_NB, MIN_NS = 20, 20, 50
C = "/scratch/llee/pairnet/cuts_study"
COLS = ["m1", "m2", "delta1", "delta2", "d32_1", "d32_2", "d2_63_32", "pt_low6", "valid"]
TRIG, FSYS, WIN, PEAK_MAX, SEED_GRID = 450.0, 0.1, (0.85, 1.1), 0.75, True
neff = lambda x: x.sum() ** 2 / max((x ** 2).sum(), 1e-300)
SRCARR = None  # per-event sample label of the QCD dict in use (set by the driver)
SRC = ["old", "ht", "vlm", "qcdvlmT", "qcdhtT", "qcd700T", "qcd700U"]


def std(src, pre_pn, pre_am, extra):
    d = {f"pn_{c}": src[f"{pre_pn}_{c}"] for c in COLS}
    d.update({f"am_{c}": src[f"{pre_am}_{c}"] for c in COLS})
    d.update(extra)
    return d


def derive(d):
    for p in ("pn", "am"):
        d[f"{p}_mavg"] = 0.5 * (d[f"{p}_m1"] + d[f"{p}_m2"])
        d[f"{p}_asym"] = np.abs(d[f"{p}_m1"] - d[f"{p}_m2"]) / np.maximum(d[f"{p}_m1"] + d[f"{p}_m2"], 1e-9)
        d[f"{p}_delta_min"] = np.minimum(d[f"{p}_delta1"], d[f"{p}_delta2"]); d[f"{p}_delta_max"] = np.maximum(d[f"{p}_delta1"], d[f"{p}_delta2"])
        d[f"{p}_d32_max"] = np.maximum(d[f"{p}_d32_1"], d[f"{p}_d32_2"])
    d["half"] = np.arange(len(d["wfb"])) % 2
    return d


def load_cache(name, first=0):
    """cached sample in the order of sorted(glob) (lexicographic file names), files c >= first.
    Returns (arrays incl. per-event weight in fb before the 1/n_files, file id per event, number of files)."""
    a = np.load(f"{D}/data/{name}.npz")
    fc = a["files_c"]; keep = fc >= first
    order = sorted(fc[keep], key=lambda c: f"{c}.npz")
    rank = np.full(int(fc.max()) + 1, -1); rank[order] = np.arange(len(order))
    k = dict(zip(fc.tolist(), (a["files_sigma_mb"] * 1e12 / a["files_weight_sum"]).tolist()))
    kf = np.zeros(int(fc.max()) + 1); kf[fc] = [k[c] for c in fc.tolist()]
    ec = a["file_c"]; ev = np.where(rank[ec] >= 0)[0]
    ev = ev[np.argsort(rank[ec[ev]], kind="stable")]
    d = {key: a[key][ev] for key in a.files if not key.startswith("file")}
    d["fid"] = ec[ev]; d["kfile"] = kf[ec[ev]]
    return d, len(order), a


def build_qcd(tails=("qcdvlmT", "qcdhtT"), verbose=False):
    P3 = "/scratch/llee/pairnet/data_r3/prod"
    fac = {f: h5py.File(f).attrs["sigma_mb"] / h5py.File(f).attrs["weight_sum"] * 1e12 for f in sorted(glob.glob(f"{P3}/qcd_c*.h5"))}
    pnf = [f for f in fac if f[-4] in "89"]
    v = np.load(f"{C}/big_vars_r14A_pn.npz"); full = np.load(f"{C}/big_vars.npz")
    ht_full = np.load(f"{C}/ht_all.npy")[full["is_signal"] | np.isin(full["fileclass"], [8, 9])]
    q = ~v["is_signal"]
    old = std({k: v[k][q] for k in v.files if v[k].ndim and len(v[k]) == len(q)}, "pn", "am",
              {"psig": v["psig"][q], "ht": ht_full[q], "pt6": v["pt6_lead"][q], "wfb": v["w"][q] * np.mean([fac[f] for f in pnf]) / len(pnf),
               "fid": v["fileclass"][q].astype(np.int64)})
    h = np.load(f"{C}/big_vars_ht_r14A_ge1400.npz")
    ht = std(h, "pn", "am", {"psig": h["psig"], "ht": h["ht"], "pt6": h["pt6_lead"], "wfb": h["w"] * h["fac"] / int(h["n_files"]), "fid": h["fileclass"].astype(np.int64)})
    a, nvlm, _ = load_cache("qcdvlm")
    vlm = std(a, "r14A", "am", {"psig": a["r14A_psig"], "ht": a["ht"], "pt6": a["pt6_lead"], "wfb": a["w"] * a["kfile"] / nvlm, "fid": a["fid"]})
    samples = {"old": derive(old), "ht": derive(ht), "vlm": derive(vlm)}
    REGIONS = [(400, 900, ["vlm"]), (900, 1200, ["old", "lm", "vlm"]), (1200, 1e9, ["old", "ht", "lm", "vlm"])]
    COMPLETE = {"old": 900, "lm": 900, "ht": 1200, "vlm": 400}
    merge = {}
    for n, d in samples.items():
        d["w0"] = d["wfb"].copy()  # cross-section weight of the sample alone (before any merge factor)
    for lo, hi, names in REGIONS:
        names = [n for n in names if n in samples]; ne = {}
        for n in names:
            d = samples[n]
            r = (d["ht"] >= lo) & (d["ht"] < hi) & (d["pn_asym"] < 0.1) & (d["pn_mavg"] > 150) & (d["pn_mavg"] < 1150)
            ne[n] = neff(d["wfb"][r])
        tot = sum(ne.values())
        for n in names:
            d = samples[n]; r = (d["ht"] >= lo) & (d["ht"] < hi)
            d["wfb"] = np.where(r, d["wfb"] * ne[n] / tot, d["wfb"])
        merge[f"{lo}-{hi}"] = {n: {"neff": ne[n], "alpha": ne[n] / tot} for n in names}
    for n, d in samples.items():
        d["wfb"] = np.where(d["ht"] >= COMPLETE[n], d["wfb"], 0.0)
    first = {"qcdvlmT": 900, "qcdhtT": 500}
    for tname in tails:
        spec = TM.TAILS[tname]
        a, nf, _ = load_cache(tname, first.get(tname, 0))
        td = derive(std(a, "r14A", "am", {"psig": a["r14A_psig"], "ht": a["ht"], "pt6": a["pt6_lead"], "wfb": a["w"] * a["kfile"] / nf, "fid": a["fid"]}))
        td["w0"] = td["wfb"].copy()
        Fd = lambda d: TM.in_filter(d["pn_valid"], d["pn_m1"], d["pn_m2"], d["psig"], d["pn_delta1"], d["pn_delta2"], spec)
        base = [(samples[n]["wfb"], samples[n]["ht"], Fd(samples[n])) for n in samples]
        td["wfb"], tinfo = TM.combine(base, (td["wfb"], td["ht"], Fd(td)), spec)
        samples[tname] = td; merge[tname] = {"files": nf, "events": int(len(td["wfb"])), **tinfo}
    for n, d in samples.items():
        d["src"] = np.full(len(d["wfb"]), SRC.index(n), np.int8)
    QCD = {k: np.concatenate([samples[n][k] for n in samples]) for k in samples["old"]}
    QCD["half"] = np.arange(len(QCD["wfb"])) % 2
    if verbose:
        print("QCD merge:", json.dumps(merge, default=float), flush=True)
    return QCD, merge


def load_sig(M, kind, extra=False):
    parts, ntried = [], 0.0
    names = [f"siglm_m{M}_{kind}"] + ([f"sig{M}X_{kind}"] if extra and kind != "power" else [])
    for n in names:
        if not os.path.exists(f"{D}/data/{n}.npz"):
            continue
        a, nf, raw = load_cache(n)
        parts.append(std(a, "r14A", "am", {"psig": a["r14A_psig"], "ht": a["ht"], "pt6": a["pt6_lead"]}))
        ntried += float(raw["files_n_tried"].sum())
    if not parts:
        return None
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d["wfb"] = np.full(len(d["psig"]), 1.0 / ntried)
    return derive(d)


def merged_signal(M, extra=False):
    inc, bst = load_sig(M, "incl", extra), load_sig(M, "boost", extra)
    if inc is None:
        return None
    hb = 2.3 * M
    n_i = (inc["ht"] > hb).sum(); n_b = len(bst["ht"]); a = n_i / (n_i + n_b)
    inc["wfb"] = np.where(inc["ht"] > hb, inc["wfb"] * a, inc["wfb"]); bst["wfb"] = bst["wfb"] * (1 - a)
    d = {k: np.concatenate([inc[k], bst[k]]) for k in inc if k != "half"}
    d["half"] = np.arange(len(d["wfb"])) % 2
    return d


VARS = {"psig": ("psig", ">", None, False),
        "ht": ("ht", ">", [0, 1.6, 1.8, 2.0, 2.2, 2.3, 2.5, 2.7, 3.0, 3.5, 4.0, 5.0, 6.0], True),
        "delta_max": ("delta_max", ">", [-1e9, -.2, -.1, 0, .1, .2, .3, .4, .5, .7, 1.0, 1.3], True),
        "delta_min": ("delta_min", ">", [-1e9, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .5, .7, 1.0], True),
        "d32_max": ("d32_max", "<", [1e9, .5, .4, .3, .25, .2, .15, .1], False),
        "d2_63_32": ("d2_63_32", "<", [1e9, 2.5, 2, 1.5, 1.25, 1, .8, .6], False),
        "pt_low6": ("pt_low6", ">", [0, .05, .07, .09, .11, .13, .15, .18, .22, .26], True),
        "asym": ("asym", "<", [1e9, .08, .06, .05, .04, .03, .02], False)}
CONF = {"P": ("pn", ["psig"]), "P+boost": ("pn", ["psig", "ht", "delta_max", "delta_min"]),
        "P+all": ("pn", [k for k in VARS]), "am+all": ("am", [k for k in VARS if k != "psig"])}


class Cand:
    def __init__(self, d=None, p=None, M=None, wide=False):
        if d is None:
            return
        lo, hi = (0.4, 1.6) if wide else WIN
        sel = (d[f"{p}_asym"] < 0.1) & (d[f"{p}_mavg"] > lo * M) & (d[f"{p}_mavg"] < hi * M) & d[f"{p}_valid"] & (d["ht"] > TRIG) & (d["wfb"] > 0)
        self.x = {n: (d[k] if k in ("psig", "ht", "u") else d[f"{p}_{k}"])[sel] / (M if rel else 1) for n, (k, _, _, rel) in VARS.items() if k in d or f"{p}_{k}" in d}
        self.w = d["wfb"][sel]; self.h = d["half"][sel]; self.m = d[f"{p}_mavg"][sel] / M
        self.i = np.where(sel)[0]; self.raw = d[f"{p}_mavg"][sel]; self.ht = d["ht"][sel]

    def take(self, idx):
        c = Cand(); c.x = {n: v[idx] for n, v in self.x.items()}
        c.w, c.h, c.m, c.i, c.raw, c.ht = self.w[idx], self.h[idx], self.m[idx], self.i[idx], self.raw[idx], self.ht[idx]
        return c


SHAPE_BINS = np.arange(0.4, 1.6001, 0.05)


def shape(Sw, Bw, cuts, h):
    def peak(c, wt):
        k = passes(c, cuts) & ((c.h == h) if h in (0, 1) else True)
        y, _ = np.histogram(c.m[k], SHAPE_BINS, weights=wt[k])
        if y.sum() <= 0:
            return np.nan
        y = np.convolve(y, np.ones(3) / 3, mode="same")
        return 0.5 * (SHAPE_BINS[1:] + SHAPE_BINS[:-1])[np.argmax(y)]
    return peak(Bw, Bw.w), peak(Sw, Sw.w)


def shape_ok(Sw, Bw, cuts, h):
    if PEAK_MAX is None:
        return True
    if PEAK_BOTH:
        h = 2
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
    if SEED_GRID and Sw is not None and not np.isfinite(best):
        dims = [n for n in ("ht", "delta_max", "delta_min") if n in grids]
        pq = [g for g in (grids.get("psig") or [None])]
        pq = [pq[0]] + pq[1::12] if pq[0] is not None else [None]
        for combo in itertools.product(*[grids[n] for n in dims], pq):
            trial = {**cuts, **dict(zip(dims, combo[:-1]))}
            if combo[-1] is not None:
                trial["psig"] = combo[-1]
            lim, ns, nb, ne, _, _ = evaluate(S, B, trial, h)
            if ns >= MIN_NS and nb >= MIN_NB and ne >= MIN_NEFF and lim < best * (1 - 1e-9) and shape_ok(Sw, Bw, trial, h):
                best, cuts = lim, trial
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


def boot(c, rng):
    return c.take(np.repeat(np.arange(len(c.w)), rng.poisson(1.0, len(c.w))))


def window_of(cw, M):
    return cw.take(np.where((cw.raw > WIN[0] * M) & (cw.raw < WIN[1] * M))[0])


PEAK_BOTH = False  # variant: falling-spectrum requirement checked on all MC events (both halves) instead of the tuning half


def prep(sig, QCD, M, rng=None):
    """wide candidates at the lowest trigger threshold (450 GeV), optionally bootstrapped (one resample for everything)"""
    global TRIG
    t = TRIG; TRIG = 450.0
    Sw, Bw = Cand(sig, "pn", M, wide=True), Cand(QCD, "pn", M, wide=True)
    TRIG = t
    if rng is not None:
        Sw, Bw = boot(Sw, rng), boot(Bw, rng)
    return Sw, Bw


def run_mass(sig, QCD, M, confs=("P", "P+boost", "P+all"), cats=3, pmin=0.8, rng=None, detail=False, cands=None):
    """lowmass_study4.py main loop for one mass (PairNet-pairing configurations). rng: bootstrap the MC events."""
    Sw0, Bw0 = cands if cands is not None else prep(sig, QCD, M, rng)
    Sw = Sw0.take(np.where(Sw0.ht > TRIG)[0]); Bw = Bw0.take(np.where(Bw0.ht > TRIG)[0])
    S, B = window_of(Sw, M), window_of(Bw, M)
    r = {}
    for name in confs:
        p, free = CONF[name]
        Sw_, Bw_ = (None, None) if PEAK_MAX is None else (Sw, Bw)
        lims, info = [], []
        for h in (0, 1):
            cuts = tune(S, B, free, h, Sw=Sw_, Bw=Bw_)
            feasible = Sw_ is None or shape_ok(Sw, Bw, cuts, h)
            lt, nst, nbt, net, eSt, xBt = evaluate(S, B, cuts, h)
            lim, ns, nb, ne, eS, xB = evaluate(S, B, cuts, 1 - h)
            if not feasible:
                lim = np.inf; lt = np.inf
            qp, sp = shape(Sw, Bw, cuts, 1 - h)
            qpa, _ = shape(Sw, Bw, cuts, 2)
            lims.append(lim)
            hh = {"cuts": {k: float(x) for k, x in cuts.items()}, "nS": ns, "nB": nb, "neffB": float(ne), "effS": float(eS), "xsB_fb": float(xB),
                  "feasible": bool(feasible), "qcd_peak_over_M": float(qp), "sig_peak_over_M": float(sp), "qcd_peak_all": float(qpa), "lim": float(lim),
                  "tune": {"lim": float(lt), "nS": nst, "nB": nbt, "neffB": float(net), "effS": float(eSt), "xsB_fb": float(xBt)}}
            if detail:  # composition of the evaluation-half background
                kb = passes(B, cuts) & (B.h == 1 - h); w = B.w[kb]; src = SRCARR[B.i[kb]]
                ws = np.sort(w)[::-1]
                hh["comp"] = {SRC[s]: {"B_frac": float(w[src == s].sum() / w.sum()), "n": int((src == s).sum()), "neff": float(neff(w[src == s])) if (src == s).any() else 0.0} for s in np.unique(src)}
                hh["top1"] = float(ws[0] / w.sum()); hh["top5"] = float(ws[:5].sum() / w.sum())
                hh["dB_rel"] = float(np.sqrt((w ** 2).sum()) / w.sum())
            info.append(hh)
        r[name] = {"limit_fb": float(np.exp(np.mean(np.log(lims)))), "halves": info}
        if cats and "psig" in free and np.all(np.isfinite(lims)):
            cl_u, cl_c, cinfo = [], [], []
            for h in (0, 1):
                cuts = info[h]["cuts"]; rest = {k: v for k, v in cuts.items() if k != "psig"}
                ms, mb = passes(S, rest), passes(B, rest)
                ps = S.x["psig"][S.h == h]
                grid = sorted(t for t in set(np.quantile(ps, 1 - np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03]))) if t >= pmin)

                def cst(bd, hh, floors=True):
                    sl, bl, edges = [], [], [np.inf] + list(bd)
                    for i in range(len(bd)):
                        ks = ms & (S.h == hh) & (S.x["psig"] >= edges[i + 1]) & (S.x["psig"] < edges[i])
                        kb = mb & (B.h == hh) & (B.x["psig"] >= edges[i + 1]) & (B.x["psig"] < edges[i])
                        if floors and (kb.sum() < MIN_NB or neff(B.w[kb]) < MIN_NEFF):
                            return None
                        sl.append(2 * L * S.w[ks].sum()); bl.append(2 * L * B.w[kb].sum())
                    return sl, bl
                bounds = [cuts["psig"]]; cur = mu_up_multi(*cst(bounds, h, False), FSYS)
                for _ in range(cats - 1):
                    cand = [(mu_up_multi(*c, FSYS), t) for t in grid if t < bounds[-1] for c in [cst(bounds + [t], h)] if c]
                    if not cand or min(cand)[0] >= cur * 0.98:
                        break
                    cur, t = min(cand); bounds.append(t)
                sl, bl = cst(bounds, 1 - h, False); slt, blt = cst(bounds, h, False)
                cl_u.append(mu_up_multi(sl, bl, FSYS)); cl_c.append(mu_up_multi_corr(sl, bl, FSYS))
                cinfo.append({"bounds": [float(b) for b in bounds], "S_per_fb": [float(x) for x in sl], "B": [float(x) for x in bl],
                              "single_corr_fb": float(mu_up_multi_corr(sl[:1], bl[:1], FSYS)), "lim_u": float(cl_u[-1]), "lim_c": float(cl_c[-1]),
                              "tune_u": float(cur), "tune_c": float(mu_up_multi_corr(slt, blt, FSYS))})
            r[name + "+cats"] = {"limit_fb": float(np.exp(np.mean(np.log(cl_u)))), "limit_corr_fb": float(np.exp(np.mean(np.log(cl_c)))),
                                 "single_corr_fb": float(np.exp(np.mean(np.log([c["single_corr_fb"] for c in cinfo])))), "halves": cinfo}
    return r


# ------------------------------------------------------------------ high-mass pipeline (fisher_high.py, COUNT_F path)
def build_high(verbose=False):
    filt = {int(k): x for k, x in json.load(open("/scratch/llee/pairnet/xs_study/xs_study.json"))["filter_eff"].items()}
    P = "/scratch/llee/pairnet/data_r3/prod"
    fac = {f: h5py.File(f).attrs["sigma_mb"] / h5py.File(f).attrs["weight_sum"] * 1e12 for f in sorted(glob.glob(f"{P}/qcd_c*.h5"))}
    pnf = [f for f in fac if f[-4] in "89"]; k_old = np.mean([fac[f] for f in pnf]) / len(pnf)
    full = np.load(f"{C}/big_vars.npz")
    HT_SUB = np.load(f"{C}/ht_all.npy")[full["is_signal"] | np.isin(full["fileclass"], [8, 9])]
    v = np.load(f"{C}/big_vars_r14A_pn.npz"); h = np.load(f"{C}/big_vars_ht_r14A_ge1400.npz")
    s = v["is_signal"]; q = ~s
    wo = np.where(q, v["w"] * k_old, 0.0); wn = h["w"] * h["fac"] / int(h["n_files"])
    hi = q & (HT_SUB > 1200); a_, b_ = neff(wo[hi]), neff(wn); al = a_ / (a_ + b_)
    wo = np.where(hi, wo * al, wo); wn = wn * (1 - al)
    cat = lambda k: np.concatenate([v[k], h[k]])
    d = {"m1": cat("pn_m1"), "m2": cat("pn_m2"), "psig": cat("psig"), "w": np.concatenate([wo, wn]),
         "sig": np.concatenate([s, np.zeros(len(wn), bool)]), "pm": np.concatenate([v["pmass"], np.zeros(len(wn))]),
         "src": np.concatenate([np.zeros(len(wo), np.int8), np.ones(len(wn), np.int8)])}
    d["mavg"] = 0.5 * (d["m1"] + d["m2"]); d["A"] = np.abs(d["m1"] - d["m2"]) / np.maximum(d["m1"] + d["m2"], 1e-9)
    d["valid"] = cat("pn_valid").astype(bool); d["ht"] = np.concatenate([HT_SUB, h["ht"]])
    info = {}
    first = {"qcdvlmT": 900, "qcdhtT": 500}
    for tname in ("qcdvlmT", "qcdhtT"):
        spec = TM.TAILS[tname]
        a, nf, _ = load_cache(tname, first[tname]); twfb = a["w"] * a["kfile"] / nf
        Fb = TM.in_filter(d["valid"], d["m1"], d["m2"], d["psig"], None, None, spec, use_delta=False) & ~d["sig"]
        Ft = TM.in_filter(a["r14A_valid"], a["r14A_m1"], a["r14A_m2"], a["r14A_psig"], None, None, spec, use_delta=False)
        d["w"] = np.where(Fb & (d["ht"] < 900), 0.0, d["w"])
        tw, ti = TM.combine([(d["w"], d["ht"], Fb)], (twfb, a["ht"], Ft), spec)
        n = len(tw); info[tname] = {"files": nf, "events": n, **ti}
        add = {"m1": a["r14A_m1"], "m2": a["r14A_m2"], "psig": a["r14A_psig"], "w": np.where(Ft, tw, 0.0), "sig": np.zeros(n, bool), "pm": np.zeros(n),
               "valid": a["r14A_valid"].astype(bool), "ht": a["ht"], "src": np.full(n, SRC.index(tname), np.int8)}
        add["mavg"] = 0.5 * (add["m1"] + add["m2"]); add["A"] = np.abs(add["m1"] - add["m2"]) / np.maximum(add["m1"] + add["m2"], 1e-9)
        d = {k: np.concatenate([d[k], add[k]]) for k in add}
    d["half"] = np.arange(len(d["w"])) % 2
    if verbose:
        print("high-mass tail merge:", info, "alpha_old", al, flush=True)
    return d, filt, {"alpha_old": al, "tail_merge": info}


def high_arrays(d, filt, M):
    """reduced arrays for one mass: signal / QCD events with A < 0.1 and 0.4 < m_avg / M < 2 (no valid, no H_T cut, as published)"""
    sM = d["sig"] & (d["pm"] == M)
    base = (d["A"] < 0.1) & (d["mavg"] > 0.4 * M) & (d["mavg"] < 2.0 * M)
    A = {"ntot": np.array([(sM & (d["half"] == 0)).sum(), (sM & (d["half"] == 1)).sum()]), "filt": filt[M], "M": M,
         "pgrid": [d["psig"][sM & (d["A"] < 0.1) & (d["half"] == hh)] for hh in (0, 1)]}
    for nm, k in (("s", base & sM), ("q", base & ~d["sig"] & (d["w"] > 0))):
        A[nm] = {"m": d["mavg"][k] / M, "p": d["psig"][k], "h": d["half"][k], "w": d["w"][k], "src": d["src"][k], "ht": d["ht"][k], "valid": d["valid"][k]}
    return A


def hpeak(m, w):
    y, _ = np.histogram(m, np.arange(0.4, 1.6001, 0.05), weights=w)
    y = np.convolve(y, np.ones(3) / 3, mode="same")
    return 0.425 + 0.05 * np.argmax(y) if y.sum() > 0 else np.nan


def run_high(A, f=None, rng=None, cats=True, detail=False, peak_max=None, extra=None):
    """fisher_high.py main loop for one mass, counting with relative uncertainty f. extra: optional mask function(q or s dict) for added requirements."""
    f = FSYS if f is None else f; peak_max = PEAK_MAX if peak_max is None else peak_max
    s, q = dict(A["s"]), dict(A["q"]); ntot = A["ntot"].astype(float); pg = A["pgrid"]
    if extra is not None:
        ks, kq = extra(s), extra(q)
        s = {k: v[ks] for k, v in s.items()}; q = {k: v[kq] for k, v in q.items()}
    if rng is not None:
        for dct in (s, q):
            idx = np.repeat(np.arange(len(dct["m"])), rng.poisson(1.0, len(dct["m"])))
            for k in dct:
                dct[k] = dct[k][idx]
    sw_ = (s["m"] > 0.85) & (s["m"] < 1.1); qw_ = (q["m"] > 0.85) & (q["m"] < 1.1)

    def fom(t, hh):
        ss = (s["h"] == hh) & (s["p"] >= t); qq = (q["h"] == hh) & (q["p"] >= t)
        qw = 2 * q["w"][qq & qw_] * L; S = (ss & sw_).sum() * A["filt"] / ntot[hh] * L
        info = {"nS": int((ss & sw_).sum()), "nB": int((qq & qw_).sum()), "neffB": float(neff(qw)), "S_per_fb": float(S), "B": float(qw.sum())}
        aq = q["p"] >= t; as_ = s["p"] >= t
        info["qcd_peak_over_M"] = float(hpeak(q["m"][aq], q["w"][aq])); info["sig_peak_over_M"] = float(hpeak(s["m"][as_], np.ones(as_.sum())))
        feas = info["qcd_peak_over_M"] <= peak_max and 0.85 <= info["sig_peak_over_M"] <= 1.1
        lim = s_up(info["B"], f) / S if S > 0 else np.inf
        return lim, info, feas
    lims, halves = [], []
    for hh in (0, 1):
        grid = [-1.0] + list(np.quantile(pg[hh], 1 - np.concatenate([np.linspace(0.96, 0.2, 20), np.geomspace(0.15, 0.01, 10)])))
        best, bt = np.inf, None
        for t in grid:
            l, info, feas = fom(t, hh)
            if feas and info["nS"] >= 50 and info["nB"] >= 10 and info["neffB"] >= 20 and l < best:
                best, bt, it = l, t, info
        if bt is None:
            lims.append(np.inf); halves.append({"feasible": False, "lim": np.inf, "tune": {"lim": np.inf}}); continue
        l, info, feas = fom(bt, 1 - hh)
        info["tune"] = {"lim": float(best), "nB": it["nB"], "neffB": it["neffB"], "B": it["B"], "S_per_fb": it["S_per_fb"]}
        if cats:
            def cst(bd, h_, floors=True):
                sl, bl = [], []
                for lo, hi in zip(bd, [2.0] + bd[:-1]):
                    kq = qw_ & (q["h"] == h_) & (q["p"] >= lo) & (q["p"] < hi); qw = 2 * q["w"][kq] * L
                    if floors and (kq.sum() < 10 or neff(qw) < 20):
                        return None
                    sl.append((sw_ & (s["h"] == h_) & (s["p"] >= lo) & (s["p"] < hi)).sum() / ntot[h_] * A["filt"] * L); bl.append(qw.sum())
                return sl, bl
            cb = [bt]; cur = mu_up_multi(*cst(cb, hh, False), f)
            for _ in range(2):
                cand = [(mu_up_multi(*c, f), t) for t in grid if 0.8 <= t < cb[-1] for c in [cst(cb + [t], hh)] if c]
                if not cand or min(cand)[0] >= cur * 0.98:
                    break
                cur, t = min(cand); cb.append(t)
            sl, bl = cst(cb, 1 - hh, False)
            info.update({"count_cats_bounds": [float(b) for b in cb], "count_cats_fb": float(mu_up_multi(sl, bl, f)), "count_cats_corr_fb": float(mu_up_multi_corr(sl, bl, f)),
                         "cats_S": [float(x) for x in sl], "cats_B": [float(x) for x in bl], "cats_tune_u": float(cur)})
        if detail:
            kq = qw_ & (q["h"] == 1 - hh) & (q["p"] >= bt); w = q["w"][kq]; src = q["src"][kq]; ws = np.sort(w)[::-1]
            info["comp"] = {SRC[i]: {"B_frac": float(w[src == i].sum() / w.sum()), "n": int((src == i).sum()), "neff": float(neff(w[src == i]))} for i in np.unique(src)}
            info["top1"] = float(ws[0] / w.sum()); info["top5"] = float(ws[:5].sum() / w.sum()); info["dB_rel"] = float(np.sqrt((w ** 2).sum()) / w.sum())
            info["B_ht_lt900_frac"] = float(w[q["ht"][kq] < 900].sum() / w.sum()); info["B_invalid_frac"] = float(w[~q["valid"][kq]].sum() / w.sum())
            ks = sw_ & (s["h"] == 1 - hh) & (s["p"] >= bt); info["S_invalid_frac"] = float((~s["valid"][ks]).mean())
        lims.append(l if feas else np.inf); halves.append({"psig": float(bt), "feasible": bool(feas), "lim": float(l if feas else np.inf), **info})
    ccl = [x.get("count_cats_fb", np.inf) if x.get("feasible") else np.inf for x in halves]
    ccc = [x.get("count_cats_corr_fb", np.inf) if x.get("feasible") else np.inf for x in halves]
    return {"limit_fb": float(np.exp(np.mean(np.log(lims)))), "count_cats_fb": float(np.exp(np.mean(np.log(ccl)))), "count_cats_corr_fb": float(np.exp(np.mean(np.log(ccc)))), "halves": halves}
