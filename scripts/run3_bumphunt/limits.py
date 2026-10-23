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


if __name__ == "__main__":
    for B in (1, 10, 100, 1e4, 1e6):
        print(B, [round(s_up(B, f), 2) for f in (0, 0.001, 0.01, 0.05, 0.1)])
