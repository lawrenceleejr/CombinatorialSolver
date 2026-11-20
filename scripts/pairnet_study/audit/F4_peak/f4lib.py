"""F4 audit: shared helpers for the signal-in-peak ceiling study (ancestry-labelled samples) and its background cost."""
import glob
import itertools

import h5py
import numpy as np

K = 12  # jets kept per event in a jet collection
WIN = {"w85": (0.85, 1.10), "w92": (0.92, 1.04)}
ACUT = 0.1


def p4_from_ptetaphim(a):
    """(..., 4) (pt, eta, phi, m) -> (E, px, py, pz)"""
    pt, eta, phi, m = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return np.stack([np.sqrt(px**2 + py**2 + pz**2 + m**2), px, py, pz], -1)


def mass(p):
    return np.sqrt(np.maximum(p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(-1), 0.0))


def load_signal(pattern, max_files=None):
    files = sorted(glob.glob(pattern))[:max_files]
    keys = ["ANC/true", "ANC/smeared", "ANC/label_p4", "ANC/label_pt", "ANC/std_index", "ANC/n", "truth/gluino_p4",
            "truth/desc_p4", "truth/quarks_before_fsr", "truth/quarks_after_fsr", "truth/jet_index",
            "EventVars/pass_sel", "INPUTS/Source/pt", "INPUTS/Source/eta", "INPUTS/Source/phi", "INPUTS/Source/mass"]
    d = {k: [] for k in keys}
    ntried = 0
    for fn in files:
        with h5py.File(fn, "r") as f:
            for k in keys:
                d[k].append(f[k][:])
            ntried += int(f.attrs["n_tried"])
    d = {k: np.concatenate(v) for k, v in d.items()}
    d["n_tried"], d["files"] = ntried, files
    return d


def jet_collection(d, ptmin=30.0, etamax=3.0, true_frac=0.8, level="smeared"):
    """Jets passing (smeared pT > ptmin, |eta| < etamax, true pT > true_frac * ptmin), sorted by smeared pT, first K.
    Returns dict: p4 (n,K,4) at `level`, tp4 (true), lpt (n,K,3), lp4 (n,K,3,4), valid (n,K)."""
    sm, tr = d["ANC/smeared"], d["ANC/true"]
    n, AJ = sm.shape[:2]
    exists = np.arange(AJ)[None] < d["ANC/n"][:, None]
    sel = exists & (sm[..., 0] > ptmin) & (np.abs(sm[..., 1]) < etamax) & (tr[..., 0] >= true_frac * ptmin)
    key = np.where(sel, sm[..., 0], -1.0)
    o = np.argsort(-key, axis=1, kind="stable")[:, :K]
    rows = np.arange(n)[:, None]
    valid = sel[rows, o]
    out = {"valid": valid, "sp4": p4_from_ptetaphim(sm[rows, o]) * valid[..., None],
           "tp4": p4_from_ptetaphim(tr[rows, o]) * valid[..., None],
           "lpt": d["ANC/label_pt"][rows, o] * valid[..., None],
           "lp4": d["ANC/label_p4"][rows, o] * valid[..., None, None],
           "ptetaphim": sm[rows, o] * valid[..., None]}
    out["p4"] = out["sp4"] if level == "smeared" else out["tp4"]
    return out


def recluster(jc, R=0.8, ptmin=30.0):
    """Anti-kT recluster the jets of a collection with radius R (labels are summed). Returns a new collection."""
    import fastjet
    jd = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    n = len(jc["valid"])
    out = {"valid": np.zeros((n, K), bool), "p4": np.zeros((n, K, 4)), "tp4": np.zeros((n, K, 4)),
           "lpt": np.zeros((n, K, 3)), "lp4": np.zeros((n, K, 3, 4)), "nsub": np.zeros((n, K), int)}
    for e in range(n):
        idx = np.where(jc["valid"][e])[0]
        if len(idx) == 0:
            continue
        pj = []
        for i in idx:
            E, px, py, pz = jc["p4"][e, i]
            j = fastjet.PseudoJet(float(px), float(py), float(pz), float(E))
            j.set_user_index(int(i))
            pj.append(j)
        cs = fastjet.ClusterSequence(pj, jd)
        fat = fastjet.sorted_by_pt(cs.inclusive_jets(ptmin))[:K]
        for k, fj in enumerate(fat):
            ci = [c.user_index() for c in fj.constituents()]
            out["valid"][e, k] = True
            out["p4"][e, k] = jc["p4"][e, ci].sum(0)
            out["tp4"][e, k] = jc["tp4"][e, ci].sum(0)
            out["lpt"][e, k] = jc["lpt"][e, ci].sum(0)
            out["lp4"][e, k] = jc["lp4"][e, ci].sum(0)
            out["nsub"][e, k] = len(ci)
    out["sp4"] = out["p4"]
    return out


# ---------------------------------------------------------------- ancestry-defined groupings
def flex_assign(jc, N, key="p4", min_frac=0.0):
    """Each of the leading N jets goes to gluino 1 / gluino 2 / neither by ancestry majority.
    Returns m1, m2, n1, n2 (group sizes)."""
    lpt = jc["lpt"][:, :N]
    v = jc["valid"][:, :N]
    lab = np.where(v, lpt.argmax(-1), 2)
    if min_frac > 0:
        fr = lpt.max(-1) / np.maximum(lpt.sum(-1), 1e-9)
        lab = np.where(fr >= min_frac, lab, 2)
    p = jc[key][:, :N]
    g = [(p * (lab == k)[..., None]).sum(1) for k in (0, 1)]
    return mass(g[0]), mass(g[1]), (lab == 0).sum(1), (lab == 1).sum(1)


def ordered_33(N):
    out = []
    for t1 in itertools.combinations(range(N), 3):
        rest = [i for i in range(N) if i not in t1]
        for t2 in itertools.combinations(rest, 3):
            out.append(t1 + t2)
    return np.array(out)


def best33_assign(jc, N, key="p4", chunk=1000):
    """Ancestry-best 3+3 among the leading N jets: the two disjoint triplets capturing the largest labelled pT
    (triplet 1 scored with gluino-1 pT, triplet 2 with gluino-2 pT). Returns m1, m2, ok (>= 6 jets), captured frac."""
    T = ordered_33(N)
    n = len(jc["valid"])
    m1, m2, cap = np.zeros(n), np.zeros(n), np.zeros(n)
    ok = jc["valid"][:, :N].sum(1) >= 6
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        lpt, v, p = jc["lpt"][sl, :N], jc["valid"][sl, :N], jc[key][sl, :N]
        sc = lpt[:, T[:, :3], 0].sum(-1) + lpt[:, T[:, 3:], 1].sum(-1)
        sc = np.where(v[:, T].all(-1), sc, -1.0)
        b = sc.argmax(1)
        r = np.arange(len(b))[:, None]
        m1[sl] = mass(p[r, T[b, :3]].sum(1))
        m2[sl] = mass(p[r, T[b, 3:]].sum(1))
        cap[sl] = sc[r[:, 0], b] / np.maximum(jc["lpt"][sl][..., :2].sum((1, 2)), 1e-9)
    return m1, m2, ok, cap


# ---------------------------------------------------------------- exhaustive schemes (mass oracle, min-asymmetry)
_PAIRS = {}


def pair_list(N, smin, smax, sizes=None):
    """Unordered disjoint subset pairs (bitmasks) of N jets with both sizes in [smin, smax] (or (a, b) in `sizes`)."""
    key = (N, smin, smax, sizes)
    if key not in _PAIRS:
        pc = np.array([bin(i).count("1") for i in range(1 << N)])
        a, b = [], []
        for s1 in range(1, 1 << N):
            if not smin <= pc[s1] <= smax:
                continue
            rest = ((1 << N) - 1) ^ s1
            s2 = rest
            while s2 > s1:  # enumerate submasks of rest larger than s1 (unordered)
                if smin <= pc[s2] <= smax and (sizes is None or tuple(sorted((pc[s1], pc[s2]))) in sizes):
                    a.append(s1)
                    b.append(s2)
                s2 = (s2 - 1) & rest
        _PAIRS[key] = (np.array(a), np.array(b), pc)
    return _PAIRS[key]


def subset_masses(p4, valid, N):
    """Masses of all 2^N subsets of the leading N jets; subsets containing an invalid jet get -1."""
    n = len(p4)
    S = np.zeros((n, 1, 4), dtype=np.float32)
    bad = np.zeros((n, 1), dtype=bool)
    for j in range(N):
        S = np.concatenate([S, S + p4[:, j:j + 1].astype(np.float32)], 1)
        bad = np.concatenate([bad, bad | ~valid[:, j:j + 1]], 1)
    m = mass(S)
    m[bad] = -1.0
    return m


def exhaustive(p4, valid, N, smin, smax, windows, sizes=None, chunk=None, weights=None):
    """For every event: (a) does any grouping in the scheme have A < ACUT and m_avg in window w (per window);
    (b) the minimum-asymmetry grouping's (m_avg, A). windows: dict name -> (lo, hi) in GeV. Returns dict."""
    a, b, _ = pair_list(N, smin, smax, sizes)
    n = len(p4)
    chunk = chunk or max(100, int(4e6 // max(len(a), 1)))
    res = {"minA_m": np.zeros(n, np.float32), "minA_A": np.ones(n, np.float32), "has": np.zeros(n, bool),
           "nsym": np.zeros(n, np.int32)}
    for w in windows:
        res["any_" + w] = np.zeros(n, bool)
        res["nwin_" + w] = np.zeros(n, np.int32)
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        sm = subset_masses(p4[sl], valid[sl], N)
        m1, m2 = sm[:, a], sm[:, b]
        ok = (m1 >= 0) & (m2 >= 0)
        tot = m1 + m2
        A = np.where(ok, np.abs(m1 - m2) / np.maximum(tot, 1e-6), 9.0)
        mavg = 0.5 * tot
        sym = A < ACUT
        res["nsym"][sl] = sym.sum(1)
        res["has"][sl] = ok.any(1)
        i = A.argmin(1)
        r = np.arange(len(i))
        res["minA_m"][sl], res["minA_A"][sl] = mavg[r, i], A[r, i]
        for w, (lo, hi) in windows.items():
            inw = sym & (mavg > lo) & (mavg < hi)
            res["any_" + w][sl] = inw.any(1)
            res["nwin_" + w][sl] = inw.sum(1)
    return res


# ---------------------------------------------------------------- metrics
def shortest_halfwidth(x, frac=0.68):
    x = np.sort(x)
    n = len(x)
    if n < 20:
        return float("nan")
    k = int(np.ceil(frac * n))
    w = x[k - 1:] - x[:n - k + 1]
    return float(0.5 * w.min())


def metrics(m1, m2, ok, M, ntot):
    """Fractions of all produced events (ntot) with A < 0.1, and in each window; core width of m_avg / M."""
    tot = m1 + m2
    A = np.where(ok & (tot > 0), np.abs(m1 - m2) / np.maximum(tot, 1e-9), 9.0)
    x = 0.5 * tot / M
    sym = A < ACUT
    out = {"fA": sym.sum() / ntot}
    for w, (lo, hi) in WIN.items():
        k = (sym & (x > lo) & (x < hi)).sum()
        out[w] = k / ntot
        out[w + "_err"] = np.sqrt(max(k, 1) * (1 - k / ntot)) / ntot
    out["w68"] = shortest_halfwidth(x[sym])
    out["med"] = float(np.median(x[sym])) if sym.any() else float("nan")
    return out
