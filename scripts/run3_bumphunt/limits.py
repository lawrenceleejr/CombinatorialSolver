"""Median expected 95% CL upper limit on signal events for background B with a relative background uncertainty f.

f = 0: Asimov counting, 2[s - B ln(1 + s/B)] = 1.96^2 (as xs_study.py).
f > 0: Asimov exclusion significance with background uncertainty Delta = f B
(Bhattiprolu, Martin, Wells, arXiv:2009.07249, eq. 2.11), solved for Z = 1.96. Floor of 3 events throughout.
"""
import math
from scipy.optimize import brentq

Z = 1.96


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
    """Expected 95% CL limit on a common signal scale for several independent counting categories: S[i] = expected
    signal per unit scale, B[i] = background, uncorrelated relative uncertainty f per category; solves
    sum_i Z_i(mu S_i, B_i, f)^2 = 1.96^2. Floor: the total signal at the limit is at least 3 events."""
    pairs = [(s, b) for s, b in zip(S, B) if s > 0 and b > 0]
    if not pairs:
        return math.inf
    S, B = [p[0] for p in pairs], [p[1] for p in pairs]
    g = lambda mu: math.sqrt(sum(z_excl(mu * s, b, f) ** 2 for s, b in pairs)) - Z
    hi = 1.0 / sum(S)
    while g(hi) < 0:
        hi *= 2
    return max(brentq(g, 1e-12 * hi, hi), 3.0 / sum(S))


def mu_up_multi_corr(S, B, f=0.0):
    """As mu_up_multi but with one fully correlated relative uncertainty f on all backgrounds (common nuisance theta,
    B_i -> B_i (1 + f theta), Gaussian constraint): Asimov background-only data, profile-likelihood q_mu = 1.96^2."""
    from scipy.optimize import minimize_scalar
    pairs = [(s, b) for s, b in zip(S, B) if s > 0 and b > 0]
    if not pairs:
        return math.inf

    def q(mu):
        def nll(th):
            v = th * th
            for s, b in pairs:
                lam = mu * s + b * (1 + f * th)
                if lam <= 0:
                    return 1e30
                v += 2 * (lam - b - b * math.log(lam / b))
            return v
        return minimize_scalar(nll, bounds=(-5, 5), method="bounded").fun if f > 0 else nll(0.0)
    tot = sum(s for s, b in pairs)
    g = lambda mu: math.sqrt(max(q(mu), 0)) - Z
    hi = 1.0 / tot
    while g(hi) < 0:
        hi *= 2
    return max(brentq(g, 1e-12 * hi, hi), 3.0 / tot)


if __name__ == "__main__":
    for B in (1, 10, 100, 1e4, 1e6):
        print(B, [round(s_up(B, f), 2) for f in (0, 0.001, 0.01, 0.05, 0.1)])
