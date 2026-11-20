"""F6 common code: load the re-scored held-out samples, QCD merge (as lowmass_study4.py + tailmerge.py for
M <= 1000 GeV, old + ht + htT for M >= 1100 GeV), signal normalisation, nested two-fold tuning, limits (limits.py
formulas) and the paired evaluation-half bootstrap.

Conventions: every event has a fixed fold (parity of its index inside its sample). "Tuning" uses fold h only,
"evaluation" fold 1-h only; yields are multiplied by 2. Limits in fb for L = 300 fb^-1.
"""
import glob, itertools, math, os
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("F6_RES", f"{HERE}/rescored")
L = 300.0
TRIG = 450.0
MIN_NEFF, MIN_NB, MIN_NS = (2, 2, 5) if os.environ.get("F6_TEST") else (20, 20, 50)
PEAK_MAX = 0.75
WIN = (0.85, 1.1)
SHAPE_BINS = np.arange(0.4, 1.6001, 0.05)
Z = 1.96
LOW_MASSES = [300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
HIGH_MASSES = list(range(1100, 2401, 100))
QTAGS = ["old", "ht", "vlm", "vlmT", "htT", "q7T", "q7U"]
TAILS = {"vlmT": dict(complete=400, pmin=0.8, dmin=0.4, mlo=150, mhi=1150),
         "htT": dict(complete=900, pmin=0.8, dmin=0.4, mlo=800, mhi=3000),
         "q7T": dict(complete=900, pmin=0.9, dmin=0.4, mlo=450, mhi=1000),
         "q7U": dict(complete=1200, pmin=0.9, dmin=0.4, mlo=450, mhi=1000)}
TREG = [(400, 900), (900, 1200), (1200, 1e9)]


# ------------------------------------------------------------------ limits (same formulas as limits.py)
def z_excl(s, b, f):
    if f <= 0:
        return math.sqrt(max(2 * (s - b * math.log1p(s / b)), 0))
    d2 = (f * b) ** 2
    x = math.sqrt((s + b) ** 2 - 4 * s * b * d2 / (b + d2))
    val = 2 * (s - b * math.log((b + s + x) / (2 * b)) - b * b / d2 * math.log((b - s + x) / (2 * b))) - (b + s - x) * (1 + b / d2)
    return math.sqrt(max(val, 0))


def s_up(B, f=0.0):
    if B <= 0:
        return 3.0
    g = lambda s: z_excl(s, B, f) - Z
    hi = 10 + 10 * math.sqrt(B) + 10 * f * B
    while g(hi) < 0:
        hi *= 2
    return max(brentq(g, 1e-9, hi), 3.0)


def mu_up_multi(S, B, f=0.0):
    pairs = [(s, b) for s, b in zip(S, B) if s > 0 and b > 0]
    if not pairs:
        return math.inf
    g = lambda mu: math.sqrt(sum(z_excl(mu * s, b, f) ** 2 for s, b in pairs)) - Z
    tot = sum(p[0] for p in pairs)
    hi = 1.0 / tot
    while g(hi) < 0:
        hi *= 2
    return max(brentq(g, 1e-12 * hi, hi), 3.0 / tot)


_SUP = {}


def s_up_fast(B, f):
    """Vectorised s_up by log-log interpolation of a table (relative error < 1e-3)."""
    if f not in _SUP:
        g = np.logspace(-3, 11, 1401)
        _SUP[f] = (np.log(g), np.log([s_up(b, f) for b in g]))
    lg, ls = _SUP[f]
    B = np.asarray(B, float)
    return np.where(B > 1e-3, np.exp(np.interp(np.log(np.maximum(B, 1e-3)), lg, ls)), 3.0)


def neff(w):
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300)) if len(w) else 0.0


# ------------------------------------------------------------------ loading
def load_tag(tag, with_S=False, cols=None):
    """Concatenate rescored/<tag>_NNN.npz. Returns dict of per-event arrays (+ S_* and rich_idx if with_S) and
    'files' = dict of per-file arrays; 'fidx' = running file index of each event is NOT available for QCD (events are
    filtered), only for signal (all events kept)."""
    fs = sorted(glob.glob(f"{RES}/{tag}_[0-9][0-9][0-9].npz"))
    parts, files, off, nbad = [], [], 0, 0
    for f in fs:
        a = np.load(f)
        files.append({k: a[k] for k in a.files if k.startswith("file_")}); nbad += int(a["n_bad"])
        if "w" not in a.files:
            continue
        n = len(a["w"])
        d = {k: a[k] for k in a.files if not k.startswith("file_") and k != "n_bad" and (with_S or not (k.startswith("S_") or k == "rich_idx"))
             and (cols is None or k in cols or k.startswith("S_") or k == "rich_idx")}
        if with_S:
            d["rich_idx"] = d["rich_idx"] + off
        d["_fseq"] = np.full(n, len(files) - 1, np.int32)
        if tag.startswith("sig"):  # signal keeps every event: recover the file of each event (incl / boost)
            fn = a["file_n"]; assert fn.sum() == n
            d["_floc"] = np.repeat(np.arange(len(fn)), fn).astype(np.int16)
        off += n; parts.append(d)
    if not parts:
        return None
    ev = [k for k in parts[0] if not (k.startswith("S_") or k == "rich_idx")]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ev}
    if with_S:
        for k in parts[0]:
            if k.startswith("S_") or k == "rich_idx":
                d[k] = np.concatenate([p[k] for p in parts])
    d["files"] = {k: np.concatenate([x[k] for x in files]) for k in files[0]}
    d["n_files"] = len(d["files"]["file_fid"]); d["n_bad"] = nbad
    return d


def derive(d):
    for p in ("pn", "am"):
        if f"{p}_m1" not in d:
            continue
        d[f"{p}_mavg"] = 0.5 * (d[f"{p}_m1"] + d[f"{p}_m2"])
        d[f"{p}_asym"] = np.abs(d[f"{p}_m1"] - d[f"{p}_m2"]) / np.maximum(d[f"{p}_m1"] + d[f"{p}_m2"], 1e-9)
        d[f"{p}_delta_min"] = np.minimum(d[f"{p}_delta1"], d[f"{p}_delta2"]); d[f"{p}_delta_max"] = np.maximum(d[f"{p}_delta1"], d[f"{p}_delta2"])
        d[f"{p}_d32_max"] = np.maximum(d[f"{p}_d32_1"], d[f"{p}_d32_2"])
    return d


def load_qcd(with_S=False, tags=QTAGS, cols=None):
    """All QCD samples concatenated; adds sid (index into tags), wraw (fb per event, averaged over files),
    half, wL (low-mass merge) and wH (high-mass merge)."""
    parts = []
    for i, t in enumerate(tags):
        d = load_tag(t, with_S, cols)
        if d is None:
            continue
        n = len(d["w"])
        d["wraw"] = d["w"] * d["fb_per_w"] / d["n_files"]
        d["sid"] = np.full(n, QTAGS.index(t), np.int8); d["half"] = (np.arange(n) % 2).astype(np.int8); d["sub"] = ((np.arange(n) // 2) % 2).astype(np.int8)
        print(f"  {t}: {d['n_files']} files ({d['n_bad']} unreadable), {n} events", flush=True)
        parts.append(d)
    ev = [k for k in parts[0] if isinstance(parts[0][k], np.ndarray) and not (k.startswith("S_") or k == "rich_idx")]
    Q = {k: np.concatenate([p[k] for p in parts]) for k in ev}
    if with_S:
        off = np.cumsum([0] + [len(p["w"]) for p in parts])
        Q["rich_idx"] = np.concatenate([p["rich_idx"] + o for p, o in zip(parts, off)])
        for k in parts[0]:
            if k.startswith("S_"):
                Q[k] = np.concatenate([p[k] for p in parts])
    derive(Q)
    Q["wL"], Q["mergeL"] = merge_low(Q)
    Q["wH"], Q["mergeH"] = merge_high(Q)
    return Q


def in_filter(Q, spec):
    ma = Q["pn_mavg"]
    ok = Q["pn_valid"].astype(bool) & (Q["pn_asym"] < 0.1) & (ma > spec["mlo"]) & (ma < spec["mhi"])
    return ok & ((Q["psig"] >= spec["pmin"]) | (Q["pn_delta_min"] / np.maximum(ma, 1e-9) >= spec["dmin"]))


def _add_tail(Q, w, t, info):
    """Combine tail sample t into the weights w (in place) inside its filter region, per H_T region, by neff."""
    spec = TAILS[t]; it = Q["sid"] == QTAGS.index(t)
    if not it.any():
        return
    F = in_filter(Q, spec); ht = Q["ht"]
    tw = Q["wraw"].copy(); new = np.zeros(len(w))
    for lo, hi in TREG:
        if lo < spec["complete"]:
            continue
        r = F & (ht >= lo) & (ht < hi)
        bm = r & ~it & (w > 0); tm = r & it
        nb = neff(w[bm]); nt = neff(tw[tm])
        if nt <= 0:
            continue
        a = nb / (nb + nt)
        w[bm] *= a; new[tm] = tw[tm] * (1 - a)
        info[f"{t} {lo}-{hi:g}"] = (round(nb, 1), round(nt, 1))
    w[it] = new[it]


def merge_low(Q):
    sid, ht = Q["sid"], Q["ht"]
    w = np.where(np.isin(sid, [0, 1, 2]), Q["wraw"], 0.0)
    ar = (Q["pn_asym"] < 0.1) & (Q["pn_mavg"] > 150) & (Q["pn_mavg"] < 1150)
    info = {}
    for lo, hi, names in [(400, 900, [2]), (900, 1200, [0, 2]), (1200, 1e9, [0, 1, 2])]:
        r = (ht >= lo) & (ht < hi)
        ne = {n: neff(Q["wraw"][r & ar & (sid == n)]) for n in names}
        tot = sum(ne.values())
        for n in names:
            w[r & (sid == n)] *= ne[n] / tot
        info[f"{lo}-{hi:g}"] = {QTAGS[n]: round(ne[n], 1) for n in names}
    for n, c in ((0, 900), (1, 1200), (2, 400)):
        w[(sid == n) & (ht < c)] = 0.0
    for t in ("vlmT", "htT", "q7T", "q7U"):
        _add_tail(Q, w, t, info)
    return w, info


def merge_high(Q):
    """For m_avg hypotheses >= 1100 GeV: old (H_T >= 900) and ht (H_T >= 1200) combined above 1200 by neff in
    (A < 0.1, m_avg > 900), plus the htT tail inside its filter. Only valid for H_T >= 900."""
    sid, ht = Q["sid"], Q["ht"]
    w = np.where(np.isin(sid, [0, 1]), Q["wraw"], 0.0)
    ar = (Q["pn_asym"] < 0.1) & (Q["pn_mavg"] > 900)
    r = ht >= 1200
    ne = {n: neff(Q["wraw"][r & ar & (sid == n)]) for n in (0, 1)}
    tot = sum(ne.values()); info = {"1200-": {QTAGS[n]: round(ne[n], 1) for n in ne}}
    for n in (0, 1):
        w[r & (sid == n)] *= ne[n] / tot
    w[(sid == 0) & (ht < 900)] = 0.0; w[(sid == 1) & (ht < 1200)] = 0.0
    _add_tail(Q, w, "htT", info)
    w[ht < 900] = 0.0
    return w, info


def load_sig(M, with_S=False, cols=None):
    """Held-out signal of mass M with wfb = fraction of produced events per event. M <= 1000: siglm incl + boost
    (merged above H_T > 2.3 M by event counts, as lowmass_study4.py); M >= 1100: data_r3 sig c8, c9, c18, c19."""
    low = M <= 1000
    d = load_tag(f"siglm{M}" if low else f"sig{M}", with_S, cols)
    if d is None:
        return None
    n = len(d["w"]); nt = d["files"]["file_n_tried"]
    if low:
        fmin = np.array([d["ht"][d["_floc"] == i].min() if (d["_floc"] == i).any() else 0 for i in range(len(nt))])
        isb = fmin > 2.3 * M - 1e-3  # boost files only contain H_T > 2.3 M
        assert 0 < isb.sum() < len(nt), (M, fmin)
        evb = isb[d["_floc"]]
        wi, wb = 1.0 / nt[~isb].sum(), 1.0 / nt[isb].sum()
        hb = 2.3 * M
        n_i = ((d["ht"] > hb) & ~evb).sum(); n_b = evb.sum(); a = n_i / (n_i + n_b)
        d["wfb"] = np.where(evb, wb * (1 - a), np.where(d["ht"] > hb, wi * a, wi))
        d["is_boost"] = evb
    else:
        d["wfb"] = np.full(n, 1.0 / nt.sum())
    d["half"] = (np.arange(n) % 2).astype(np.int8); d["sub"] = ((np.arange(n) // 2) % 2).astype(np.int8)
    d["M"] = np.full(n, M, np.float32)
    return derive(d)


# ------------------------------------------------------------------ tables and cuts
# name: (column, direction, grid, relative to M)
VARS = {"psig": ("psig", ">=", None, False),
        "ht": ("ht", ">", [0, 1.6, 1.8, 2.0, 2.2, 2.3, 2.5, 2.7, 3.0, 3.5, 4.0, 5.0, 6.0], True),
        "delta_max": ("pn_delta_max", ">", [-1e9, -.2, -.1, 0, .1, .2, .3, .4, .5, .7, 1.0, 1.3], True),
        "delta_min": ("pn_delta_min", ">", [-1e9, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .5, .7, 1.0], True),
        "d32_max": ("pn_d32_max", "<", [1e9, .5, .4, .3, .25, .2, .15, .1], False),
        "d2_63_32": ("pn_d2_63_32", "<", [1e9, 2.5, 2, 1.5, 1.25, 1, .8, .6], False),
        "pt_low6": ("pn_pt_low6", ">", [0, .05, .07, .09, .11, .13, .15, .18, .22, .26], True),
        "asym": ("pn_asym", "<", [1e9, .08, .06, .05, .04, .03, .02], False)}
CONF = {"P": ["psig"], "P+boost": ["psig", "ht", "delta_max", "delta_min"], "P+all": list(VARS)}
PQ = 1 - np.concatenate([np.linspace(0.96, 0.2, 20), np.geomspace(0.15, 0.005, 14)])  # signal quantiles of score grids


class Tab:
    """Event table for one mass hypothesis: w (fb or produced fraction), h (fold), m (search variable / M), eid
    (index into the parent arrays), x[name] (cut variables), DIR[name]."""

    def __init__(self, w, h, m, eid, x=None, dirs=None):
        self.w, self.h, self.m, self.eid = w, h, m, eid
        self.x = x or {}; self.dir = dirs or {}

    def add(self, name, val, direction=">="):
        self.x[name] = val; self.dir[name] = direction


def weights(Q, M):
    return Q["wL"] if M <= 1000 else Q["wH"]


def base_sel(d, M, w, lo=0.4, hi=1.6):
    return d["pn_valid"].astype(bool) & (d["pn_asym"] < 0.1) & (d["pn_mavg"] > lo * M) & (d["pn_mavg"] < hi * M) & (d["ht"] > TRIG) & (w > 0)


def make_tab(d, M, w, sel=None, m=None):
    """Standard table (r14A argmax pairing, A < 0.1, 0.4 < m_avg / M < 1.6, H_T > TRIG) with the cut variables."""
    if sel is None:
        sel = base_sel(d, M, w)
    eid = np.where(sel)[0]
    T = Tab(w[eid], d["half"][eid], (d["pn_mavg"][eid] if m is None else m[eid]) / M, eid)
    for n, (col, direction, _, rel) in VARS.items():
        T.add(n, d[col][eid] / (M if rel else 1), direction)
    return T


def passes(T, cuts):
    ok = np.ones(len(T.w), bool)
    for n, t in cuts.items():
        if n in ("wlo", "whi"):
            continue
        dr = T.dir[n]
        ok &= (T.x[n] >= t) if dr == ">=" else (T.x[n] > t) if dr == ">" else (T.x[n] < t)
    return ok


def inwin(T, cuts):
    return (T.m > cuts.get("wlo", WIN[0])) & (T.m < cuts.get("whi", WIN[1]))


def peaks(S, B, cuts, h):
    def peak(T):
        k = passes(T, cuts) & (T.h == h)
        y, _ = np.histogram(T.m[k], SHAPE_BINS, weights=T.w[k])
        if y.sum() <= 0:
            return np.nan
        y = np.convolve(y, np.ones(3) / 3, mode="same")
        return 0.5 * (SHAPE_BINS[1:] + SHAPE_BINS[:-1])[np.argmax(y)]
    return peak(B), peak(S)


def shape_ok(S, B, cuts, h):
    if PEAK_MAX is None:
        return True
    qp, sp = peaks(S, B, cuts, h)
    return bool((qp <= PEAK_MAX) and (0.85 <= sp <= 1.1))


def evaluate(S, B, cuts, h, f, exact=False):
    ks = passes(S, cuts) & inwin(S, cuts) & (S.h == h); kb = passes(B, cuts) & inwin(B, cuts) & (B.h == h)
    eS = 2 * S.w[ks].sum(); xB = 2 * B.w[kb].sum()
    su = s_up(xB * L, f) if exact else float(s_up_fast(xB * L, f))
    lim = su / (L * eS) if eS > 0 else np.inf
    return lim, int(ks.sum()), int(kb.sum()), neff(B.w[kb]), eS, xB


def grid_for(S, n, h, extra=None):
    if n in (extra or {}):
        g = extra[n]
        return g(S, h) if callable(g) else g
    g = VARS.get(n, (None, None, None))[2]
    if g is None:  # score-like variable: signal quantiles in the window on the tuning fold
        k = (S.h == h) & (S.m > WIN[0]) & (S.m < WIN[1])
        v = S.x[n][k]
        if S.dir[n] == "<":
            return [np.inf] + list(np.quantile(v, 1 - PQ)) if len(v) else [np.inf]
        return [-np.inf] + list(np.quantile(v, PQ)) if len(v) else [-np.inf]
    return g


def tune(S, B, free, h, f, rounds=3, grids=None, shape=True, seed=True, start=None):
    """Coordinate descent (as lowmass_study4.tune) on fold h. free: variable names; grids: optional {name: list}
    (first entry = no cut); 'wlo'/'whi' may be given in grids to tune the window."""
    G = {n: grid_for(S, n, h, grids) for n in free}
    cuts = {n: g[0] for n, g in G.items()}
    if start:
        cuts.update(start)
    ok = lambda c: (not shape) or shape_ok(S, B, c, h)

    def good(trial, best):
        lim, ns, nb, ne, _, _ = evaluate(S, B, trial, h, f)
        return ns >= MIN_NS and nb >= MIN_NB and ne >= MIN_NEFF and lim < best * (1 - 1e-9) and ok(trial), lim
    lim0, ns, nb, ne, _, _ = evaluate(S, B, cuts, h, f)
    best = lim0 if (ok(cuts) and ns >= MIN_NS and nb >= MIN_NB and ne >= MIN_NEFF) else np.inf
    if seed and shape and not np.isfinite(best):
        dims = [n for n in ("ht", "delta_max", "delta_min") if n in G]
        sc = [n for n in G if n not in VARS or VARS[n][2] is None]
        pq = [G[sc[0]][0]] + list(G[sc[0]][1::12]) if sc else [None]
        for combo in itertools.product(*[G[n] for n in dims], pq):
            trial = {**cuts, **dict(zip(dims, combo[:-1]))}
            if combo[-1] is not None:
                trial[sc[0]] = combo[-1]
            g, lim = good(trial, best)
            if g:
                best, cuts = lim, trial
    for _ in range(rounds):
        changed = False
        for n, g in G.items():
            for t in g:
                trial = {**cuts, n: t}
                gd, lim = good(trial, best)
                if gd:
                    best, cuts, changed = lim, trial, True
        if not changed:
            break
    return cuts, best


def tune_cats(S, B, cuts, score, h, f, ncat=3, smin=None):
    """Extra categories in `score` below the tuned threshold (other cuts unchanged), greedy boundaries on fold h with
    uncorrelated relative uncertainty f per category (as lowmass_study4.py, counting version)."""
    rest = {k: v for k, v in cuts.items() if k != score}
    ms, mb = passes(S, rest) & inwin(S, cuts), passes(B, rest) & inwin(B, cuts)
    v = S.x[score][(S.h == h) & inwin(S, cuts)]
    grid = sorted(t for t in set(np.quantile(v, 1 - np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03])))
                  if smin is None or t >= smin)

    def cst(bd, floors=True):
        sl, bl, edges = [], [], [np.inf] + list(bd)
        for i in range(len(bd)):
            ks = ms & (S.h == h) & (S.x[score] >= edges[i + 1]) & (S.x[score] < edges[i])
            kb = mb & (B.h == h) & (B.x[score] >= edges[i + 1]) & (B.x[score] < edges[i])
            if floors and (kb.sum() < MIN_NB or neff(B.w[kb]) < MIN_NEFF):
                return None
            sl.append(2 * L * S.w[ks].sum()); bl.append(2 * L * B.w[kb].sum())
        return sl, bl
    bounds = [cuts[score]]; cur = mu_up_multi(*cst(bounds, False), f)
    for _ in range(ncat - 1):
        cand = [(mu_up_multi(*c, f), t) for t in grid if t < bounds[-1] for c in [cst(bounds + [t])] if c]
        if not cand or min(cand)[0] >= cur * 0.98:
            break
        cur, t = min(cand); bounds.append(t)
    return bounds


class Sel:
    """A selection frozen on the tuning fold, represented on the evaluation fold h_eval: per category the global
    event ids and weights of the selected in-window signal and QCD events."""

    def __init__(self, S, B, cuts, h_eval, score=None, bounds=None, info=None):
        rest = {k: v for k, v in cuts.items() if k != score} if bounds else cuts
        ks = passes(S, rest) & inwin(S, cuts) & (S.h == h_eval); kb = passes(B, rest) & inwin(B, cuts) & (B.h == h_eval)
        edges = [np.inf] + list(bounds) if bounds else [None, None]
        self.cats = []
        for i in range(len(edges) - 1):
            if bounds:
                cs = ks & (S.x[score] >= edges[i + 1]) & (S.x[score] < edges[i]); cb = kb & (B.x[score] >= edges[i + 1]) & (B.x[score] < edges[i])
            else:
                cs, cb = ks, kb
            self.cats.append((S.eid[cs], S.w[cs], B.eid[cb], B.w[cb]))
        self.cuts = cuts; self.bounds = bounds; self.info = info or {}
        self.feasible = True

    def stats(self):
        es, ws, eb, wb = self.cats[0]
        allb = np.concatenate([c[3] for c in self.cats])
        return {"nS": int(sum(len(c[0]) for c in self.cats)), "nB": int(len(allb)), "neffB": neff(allb), "nB_top": int(len(wb)), "neffB_top": neff(wb),
                "effS": float(2 * sum(c[1].sum() for c in self.cats)), "xsB_fb": float(2 * allb.sum())}

    def limit(self, f, rs=None, rb=None):
        """rs / rb: optional dict-like arrays of bootstrap multiplicities indexed by global event id."""
        Ss, Bs = [], []
        for es, ws, eb, wb in self.cats:
            Ss.append(2 * L * (ws * (rs[es] if rs is not None else 1)).sum()); Bs.append(2 * L * (wb * (rb[eb] if rb is not None else 1)).sum())
        if not self.feasible:
            return np.inf
        if len(Ss) == 1:
            return s_up(Bs[0], f) / Ss[0] if Ss[0] > 0 else np.inf
        return mu_up_multi(Ss, Bs, f)


def boot(sels, ref, f, nS, nB, R=200, seed=1):
    """Paired bootstrap of the evaluation fold: Poisson(1) multiplicities per event, shared by all selections.
    sels: {name: Sel}; ref: name. Returns {name: (log-limit replicas)}."""
    rng = np.random.default_rng(seed)
    us = np.unique(np.concatenate([c[0] for s in sels.values() for c in s.cats])); ub = np.unique(np.concatenate([c[2] for s in sels.values() for c in s.cats]))
    out = {k: np.zeros(R) for k in sels}
    rs = np.zeros(nS, np.float32); rb = np.zeros(nB, np.float32)
    for r in range(R):
        rs[us] = rng.poisson(1.0, len(us)); rb[ub] = rng.poisson(1.0, len(ub))
        for k, s in sels.items():
            out[k][r] = np.log(max(s.limit(f, rs, rb), 1e-300))
    return out


def gm(x):
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(x)))) if np.all(np.isfinite(x)) and np.all(x > 0) else float("inf")


def summarize(per_fold, f, nS, nB, ref, R=200):
    """per_fold: [ {name: Sel} for eval fold 1, {name: Sel} for eval fold 0 ]. Returns {name: {limit, ratio, err,
    nS, neff...}} with ratio = limit / limit(ref), err = bootstrap std of the ratio (paired, both folds)."""
    names = [k for k in per_fold[0] if k in per_fold[1]]
    lim = {k: [per_fold[i][k].limit(f) for i in (0, 1)] for k in names}
    bs = [boot({k: per_fold[i][k] for k in names}, ref, f, nS, nB, R, seed=11 + i) for i in (0, 1)]
    res = {}
    for k in names:
        lr = 0.5 * ((bs[0][k] - bs[0][ref]) + (bs[1][k] - bs[1][ref]))
        ll = 0.5 * (bs[0][k] + bs[1][k])
        st = [per_fold[i][k].stats() for i in (0, 1)]
        g = gm(lim[k]); gr = gm(lim[ref])
        fin = np.isfinite(lr)
        res[k] = {"limit_fb": g, "ratio": g / gr if np.isfinite(g) and np.isfinite(gr) else float("nan"),
                  "ratio_err": float(np.std(np.exp(lr[fin]))) if fin.sum() > 10 else float("nan"),
                  "limit_relerr": float(np.std(ll[np.isfinite(ll)])) if np.isfinite(ll).sum() > 10 else float("nan"),
                  "nS": min(s["nS"] for s in st), "nB": min(s["nB"] for s in st), "neffB": min(s["neffB"] for s in st),
                  "neffB_top": min(s["neffB_top"] for s in st), "effS": float(np.mean([s["effS"] for s in st])),
                  "xsB_fb": float(np.mean([s["xsB_fb"] for s in st])),
                  "feasible": bool(all(per_fold[i][k].feasible for i in (0, 1))),
                  "cuts": [{a: float(b) for a, b in per_fold[i][k].cuts.items()} for i in (0, 1)],
                  "bounds": [[float(b) for b in (per_fold[i][k].bounds or [])] for i in (0, 1)]}
    return res


def run_conf(S, B, free, f, grids=None, cats=None, smin=None, score="psig", shape=True, start=None):
    """Tune on each fold, freeze, represent on the other. Returns [Sel(eval fold 1), Sel(eval fold 0)] (and with
    categories if cats)."""
    out, outc = [], []
    for h in (0, 1):
        cuts, best = tune(S, B, free, h, f, grids=grids, shape=shape, start=start)
        s = Sel(S, B, cuts, 1 - h)
        s.feasible = bool(np.isfinite(best)) and ((not shape) or shape_ok(S, B, cuts, h))
        out.append(s)
        if cats:
            if s.feasible:
                bounds = tune_cats(S, B, cuts, score, h, f, cats, smin)
                sc = Sel(S, B, cuts, 1 - h, score, bounds)
            else:
                sc = Sel(S, B, cuts, 1 - h); sc.feasible = False
            outc.append(sc)
    return (out, outc) if cats else out
