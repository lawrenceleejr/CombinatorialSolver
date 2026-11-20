"""Item 1: limits.py (s_up, mu_up_multi, mu_up_multi_corr) against independent median expected 95% CL CLs limits.

Independent calculations (none uses limits.py):
  exact      f = 0: Poisson CLs with the observed count at the background-only median (integer n)
  asimov-G   profile likelihood, n ~ Pois(s + b), auxiliary measurement m ~ Gauss(b, f B); Asimov q_s = 1.96^2
  asimov-LN  same with a log-normal constraint b = B (1 + f)^theta, theta ~ Gauss(0, 1)
  asimov-on/off  same with m ~ Pois(tau b), tau = 1 / (f^2 B)  (the model behind the BMW formula)
  toys-G     CLs with the LHC test statistic q~_s, toys for s+b and b-only (auxiliary measurement fluctuated), observed
             test statistic at the background-only median; s scanned to CLs = 0.05
Multi-category: Asimov profile likelihood with independent (and with one common) Gaussian nuisance, and toys.
"""
import json, math, sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import poisson
sys.path.insert(0, "/scratch/llee/pairnet/audit/F1_stats/repo_copy"); sys.dont_write_bytecode = True
import limits as LIM

rng = np.random.default_rng(1)
Z = 1.959964


def xlogy(n, lam):
    return np.where(n > 0, n * np.log(np.maximum(lam, 1e-300)), 0.0)


# ---------- single category, Gaussian auxiliary measurement: closed-form profile
def bhat_G(n, m, s, sig):
    """conditional MLE of b for fixed s (vectorised): root of b^2 - b (m - s - sig^2) - (n sig^2 - sig^2 s + m s) = 0"""
    p = m - s - sig ** 2; q = n * sig ** 2 - sig ** 2 * s + m * s
    return np.maximum(0.5 * (p + np.sqrt(p * p + 4 * q)), 1e-12)


def nll_G(n, m, s, b, sig):
    return (s + b) - xlogy(n, s + b) + 0.5 * ((m - b) / sig) ** 2


def qtilde_G(n, m, s, sig):
    """LHC test statistic q~_s: 0 <= s_hat <= s"""
    shat = np.clip(n - m, 0, s)  # unconstrained MLE: s = n - m, b = m; the profile is concave in s so clipping is exact
    bh = bhat_G(n, m, shat, sig)
    return np.maximum(2 * (nll_G(n, m, s, bhat_G(n, m, s, sig), sig) - nll_G(n, m, shat, bh, sig)), 0)


def asimov_G(B, f):
    if f <= 0:
        return brentq(lambda s: 2 * (s - B * math.log1p(s / B)) - Z * Z, 1e-9, 20 + 20 * math.sqrt(B))
    sig = f * B
    g = lambda s: float(qtilde_G(np.array(float(B)), np.array(float(B)), s, sig)) - Z * Z
    return brentq(g, 1e-9, 20 + 20 * math.sqrt(B) + 20 * sig)


def asimov_LN(B, f):
    k = math.log1p(f)
    def q(s):
        nll = lambda th: (s + B * math.exp(k * th)) - B * math.log(s + B * math.exp(k * th)) + 0.5 * th * th
        return 2 * (minimize_scalar(nll, bounds=(-8, 8), method="bounded", options={"xatol": 1e-10}).fun - (B - B * math.log(B)))
    return brentq(lambda s: q(s) - Z * Z, 1e-9, 20 + 20 * math.sqrt(B) + 40 * f * B)


def asimov_onoff(B, f):
    tau = 1 / (f * f * B); m = tau * B
    def q(s):
        # conditional MLE of b: maximise n ln(s+b) - (s+b) + m ln(tau b) - tau b with n = B
        nll = lambda b: (s + b) - B * math.log(s + b) + tau * b - m * math.log(tau * b)
        r = minimize_scalar(nll, bounds=(1e-9 * B, 3 * B), method="bounded", options={"xatol": 1e-12 * B})
        return 2 * (r.fun - nll(B) + 0 * s) - 2 * (0 - 0) + 2 * (nll(B) - (B - B * math.log(B) + tau * B - m * math.log(tau * B)))
    return brentq(lambda s: q(s) - Z * Z, 1e-9, 20 + 20 * math.sqrt(B) + 40 * f * B)


def exact_f0(B):
    n0 = int(poisson.median(B))
    g = lambda s: poisson.cdf(n0, s + B) / poisson.cdf(n0, B) - 0.05
    return brentq(g, 1e-6, 30 + 30 * math.sqrt(B)), n0


def toys_G(B, f, ntoy=200000, guess=None):
    """median expected CLs limit from toys; returns (s_up, statistical error from the toy counts)"""
    sig = f * B if f > 0 else None
    guess = guess or asimov_G(B, f)
    def cls(s):
        nb = rng.poisson(B, ntoy).astype(float); ns = rng.poisson(s + B, ntoy).astype(float)
        if f > 0:
            mb = np.maximum(rng.normal(B, sig, ntoy), 1e-9); ms = np.maximum(rng.normal(B, sig, ntoy), 1e-9)
            qb = qtilde_G(nb, mb, s, sig); qs = qtilde_G(ns, ms, s, sig)
        else:
            qf = lambda n: np.where(n < s + B, 2 * ((s + B) - xlogy(n, s + B) - (np.maximum(n, B) - xlogy(n, np.maximum(n, B)))), 0.0)
            qb = qf(nb); qs = qf(ns)
        qmed = np.median(qb)
        psb = np.mean(qs >= qmed); pb = np.mean(qb >= qmed)
        return psb / pb, math.sqrt(psb * (1 - psb) / ntoy) / pb
    ss = guess * np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35])
    c = np.array([cls(s) for s in ss])
    y = np.log(np.maximum(c[:, 0], 1e-6)) - math.log(0.05)
    i = np.where(y < 0)[0]
    if len(i) == 0 or i[0] == 0:
        return float("nan"), float("nan")
    i = i[0]
    slope = (y[i] - y[i - 1]) / (ss[i] - ss[i - 1])
    s0 = ss[i - 1] - y[i - 1] / slope
    err = (c[i, 1] / c[i, 0]) / abs(slope)
    return float(s0), float(err)


# ---------- multi-category, general nuisance structure: B_i (1 + fc theta_c + fu theta_i)
def asimov_multi(S, B, fc, fu):
    """mu at Asimov q_mu = 1.96^2 (no floor), numerical profile over (theta_c, theta_1..n) by Newton iterations"""
    S, B = np.asarray(S, float), np.asarray(B, float); n = len(S)

    def q(mu):
        th = np.zeros(n + 1)  # [common, per-category]
        J = np.zeros((n, n + 1)); J[:, 0] = fc * B; J[np.arange(n), 1 + np.arange(n)] = fu * B
        for _ in range(200):
            lam = mu * S + B + J @ th
            if np.any(lam <= 0):
                th *= 0.5; continue
            g = J.T @ (1 - B / lam) + th
            H = J.T @ ((B / lam ** 2)[:, None] * J) + np.eye(n + 1)
            step = np.linalg.solve(H, g)
            t = 1.0
            while np.any(mu * S + B + J @ (th - t * step) <= 0):
                t *= 0.5
            th = th - t * step
            if np.abs(step).max() < 1e-12:
                break
        lam = mu * S + B + J @ th
        return 2 * np.sum(lam - B - B * np.log(lam / B)) + th @ th
    hi = 1.0 / S.sum()
    while q(hi) < Z * Z:
        hi *= 2
    return brentq(lambda mu: q(mu) - Z * Z, 1e-12 * hi, hi)


def toys_multi(S, B, f, corr, ntoy=60000, guess=None):
    """CLs toys for several categories, Gaussian nuisances (independent per category, or one common), test statistic q~_mu"""
    S, B = np.asarray(S, float), np.asarray(B, float); n = len(S)

    def prof(N, A, mu):
        """min over theta of the NLL for fixed mu; N (ntoy, n) counts, A auxiliary (ntoy, n) or (ntoy,)"""
        if not corr:  # closed form per category: lam = mu S + b', b' = B (1 + f th), aux a: th ~ a
            sig = f * B; m = B * (1 + f * A); s = mu * S
            b = bhat_G(N, m, s, sig)
            return np.sum(nll_G(N, m, s, b, sig), 1)
        th = A.copy()
        for _ in range(60):
            lam = np.maximum(mu * S + B * (1 + f * th[:, None]), 1e-9)
            g = np.sum((1 - N / lam) * f * B, 1) + (th - A)
            H = np.sum(N * (f * B) ** 2 / lam ** 2, 1) + 1
            th = np.maximum(th - g / H, (-(mu * S + B) / (f * B)).max() + 1e-6)
        lam = np.maximum(mu * S + B * (1 + f * th[:, None]), 1e-9)
        return np.sum(lam - xlogy(N, lam), 1) + 0.5 * (th - A) ** 2

    def qt(N, A, mu):
        # mu_hat in [0, mu]: golden-section on the profile (unimodal)
        lo = np.zeros(len(N)); hi = np.full(len(N), mu); gr = 0.6180339887
        c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
        fc_ = profv(N, A, c); fd_ = profv(N, A, d)
        for _ in range(30):
            left = fc_ < fd_
            hi = np.where(left, d, hi); lo = np.where(left, lo, c)
            c2 = hi - gr * (hi - lo); d2 = lo + gr * (hi - lo)
            c, d = c2, d2
            fc_ = profv(N, A, c); fd_ = profv(N, A, d)
        best = np.minimum(np.minimum(fc_, fd_), np.minimum(profv(N, A, np.zeros(len(N))), prof(N, A, mu)))
        return np.maximum(2 * (prof(N, A, mu) - best), 0)

    def profv(N, A, muv):
        if not corr:
            sig = f * B; m = B * (1 + f * A); s = muv[:, None] * S
            return np.sum(nll_G(N, m, s, bhat_G(N, m, s, sig), sig), 1)
        th = A.copy()
        for _ in range(60):
            lam = np.maximum(muv[:, None] * S + B * (1 + f * th[:, None]), 1e-9)
            g = np.sum((1 - N / lam) * f * B, 1) + (th - A)
            H = np.sum(N * (f * B) ** 2 / lam ** 2, 1) + 1
            th = th - g / H
        lam = np.maximum(muv[:, None] * S + B * (1 + f * th[:, None]), 1e-9)
        return np.sum(lam - xlogy(N, lam), 1) + 0.5 * (th - A) ** 2

    def cls(mu):
        shp = (ntoy, n) if not corr else (ntoy,)
        Ab, As = rng.normal(0, 1, shp), rng.normal(0, 1, shp)
        Nb = rng.poisson(B, (ntoy, n)).astype(float); Ns = rng.poisson(mu * S + B, (ntoy, n)).astype(float)
        qb, qs = qt(Nb, Ab, mu), qt(Ns, As, mu)
        qmed = np.median(qb); psb = np.mean(qs >= qmed); pb = np.mean(qb >= qmed)
        return psb / pb, math.sqrt(psb * (1 - psb) / ntoy) / pb
    mus = guess * np.array([0.75, 0.9, 1.0, 1.1, 1.25])
    c = np.array([cls(m) for m in mus]); y = np.log(np.maximum(c[:, 0], 1e-6)) - math.log(0.05)
    i = np.where(y < 0)[0]
    if len(i) == 0 or i[0] == 0:
        return float("nan"), float("nan")
    i = i[0]; slope = (y[i] - y[i - 1]) / (mus[i] - mus[i - 1])
    return float(mus[i - 1] - y[i - 1] / slope), float((c[i, 1] / c[i, 0]) / abs(slope))


if __name__ == "__main__":
    out = {"single": [], "multi": []}
    print("== single category: s_up(B, f) of limits.py vs independent median expected 95% CL CLs limits (signal events)")
    print(f"{'B':>9} {'f':>5} {'s_up':>10} {'no-floor':>10} {'asimov-G':>10} {'asimov-LN':>10} {'on/off':>10} {'exact f=0':>10} {'toys-G':>16} {'s_up/toys':>12}")
    for B in (1, 3, 10, 30, 100, 300, 1e3, 1e4, 1e5, 1e6):
        for f in (0.0, 0.05, 0.1):
            su = LIM.s_up(B, f)
            raw = brentq(lambda s: LIM.z_excl(s, B, f) - 1.96, 1e-9, 1e3 + 100 * math.sqrt(B) + 100 * f * B)
            aG = asimov_G(B, f); aL = asimov_LN(B, f) if f > 0 else float("nan"); oo = asimov_onoff(B, f) if f > 0 else float("nan")
            ex = exact_f0(B)[0] if f == 0 else float("nan")
            do_toys = B in (1, 3, 10, 30, 100, 1e3, 1e4, 1e6)
            ty, te = toys_G(B, f, guess=ex if f == 0 else None) if do_toys else (float("nan"), float("nan"))
            ref = ty if do_toys else aG
            out["single"].append(dict(B=B, f=f, s_up=su, s_up_nofloor=raw, asimov_gauss=aG, asimov_lognormal=aL, asimov_onoff=oo, exact_f0=ex, toys=ty, toys_err=te))
            print(f"{B:9g} {f:5.2f} {su:10.4g} {raw:10.4g} {aG:10.4g} {aL:10.4g} {oo:10.4g} {ex:10.4g} {ty:9.4g} ± {te:6.2g} {su / ref:12.3f}", flush=True)
    print("\n== several categories: mu_up_multi (uncorrelated) / mu_up_multi_corr (one common nuisance) vs independent Asimov and toys")
    cases = {"2 cat, equal S/B": ([10.0, 10.0], [300.0, 300.0]), "2 cat, steep": ([6.0, 9.0], [150.0, 900.0]),
             "3 cat (700 GeV P+all+cats, half 0)": ([0.4224835662743348, 0.7756306118568866, 0.4408020398385036], [268.4542912924659, 1069.2189373850451, 758.7673082504851]),
             "3 cat, low B": ([2.0, 3.0, 4.0], [8.0, 40.0, 200.0]), "3 cat, high B": ([50.0, 80.0, 100.0], [3e3, 1.2e4, 5e4])}
    for name, (S, B) in cases.items():
        for f in (0.05, 0.1):
            u = LIM.mu_up_multi(S, B, f); c = LIM.mu_up_multi_corr(S, B, f)
            au = asimov_multi(S, B, 0.0, f); ac = asimov_multi(S, B, f, 0.0)
            tu, tue = toys_multi(S, B, f, False, guess=au); tc, tce = toys_multi(S, B, f, True, guess=ac)
            single = LIM.mu_up_multi(S[:1], B[:1], f)
            out["multi"].append(dict(name=name, f=f, S=S, B=B, mu_up_multi=u, mu_up_multi_corr=c, asimov_uncorr=au, asimov_corr=ac, toys_uncorr=tu, toys_uncorr_err=tue, toys_corr=tc, toys_corr_err=tce, single_first=single))
            print(f"{name:36s} f={f:4.2f}  uncorr: limits.py {u:.4g}  asimov {au:.4g}  toys {tu:.4g} ± {tue:.2g}  (limits/toys {u / tu:.3f})   corr: limits.py {c:.4g}  asimov {ac:.4g}  toys {tc:.4g} ± {tce:.2g}  (limits/toys {c / tc:.3f})   first category alone {single:.4g}", flush=True)
    json.dump(out, open("/scratch/llee/pairnet/audit/F1_stats/item1_formula.json", "w"), indent=1)
