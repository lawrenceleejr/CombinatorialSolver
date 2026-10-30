"""Stat-only sensitivity of a binned m_avg shape fit (signal strength + free 4-par dijet background) vs window counting.
Fisher information at 300 fb^-1 with the background parameters profiled; uses the spectra dumps."""
import json, sys, glob
import numpy as np, h5py
from scipy.optimize import least_squares
dd, trig, ftag = sys.argv[1:4]; masses = [int(x) for x in sys.argv[4].split(",")]
L = 300.0
th = json.load(open("/scratch/llee/pairnet/xs_theory/pp13600_gluino_NNLO+NNLL.json"))["data"]
def sigma(M):
    if M >= 500: return th[str(M)]["xsec_pb"] * 1e3
    fs = sorted(glob.glob(f"/scratch/llee/pairnet/prod_lowmass/siglm_m{M}_incl_c*.h5"))[:3]
    return 2.07 * np.mean([h5py.File(f).attrs["sigma_mb"] for f in fs]) * 1e12
def dijet(p, x): return np.exp(p[0]) * (1 - x) ** p[1] / x ** (p[2] + p[3] * np.log(x))
res = {}
for M in masses:
    a = np.load(f"{dd}/spec_r14A_m{M}_trig{trig}_{ftag}.npz", allow_pickle=True)
    b = np.linspace(0.4 * M, 2.0 * M, 49); c = 0.5 * (b[1:] + b[:-1]); x = c / 13600
    q, _ = np.histogram(a["qcd_mavg"], b, weights=a["qcd_w"] * L); q2, _ = np.histogram(a["qcd_mavg"], b, weights=(a["qcd_w"] * L) ** 2)
    s, _ = np.histogram(a["sig_mavg"], b, weights=a["sig_w"] * L * sigma(M))
    i0 = int(np.argmax(q)); ok = (np.arange(len(q)) >= i0) & (q2 > 0)
    f = lambda p: (dijet(p, x[ok]) - q[ok]) / np.sqrt(q2[ok])
    p = least_squares(f, [np.log(q[ok][0] * x[ok][0] ** 5), 10, 5, 0], method="lm", max_nfev=20000).x
    bb = dijet(p, x[ok]); ss = s[ok] / sigma(M)  # signal per fb
    J = np.zeros((ok.sum(), 5)); J[:, 0] = ss
    for k in range(4):
        e = np.zeros(4); e[k] = 1e-5 * max(1, abs(p[k])); J[:, k + 1] = (dijet(p + e, x[ok]) - dijet(p - e, x[ok])) / (2 * e[k])
    F = J.T @ (J / bb[:, None])
    sig_fit = np.sqrt(np.linalg.inv(F)[0, 0]); sig_fixed = 1 / np.sqrt(F[0, 0])
    w = (c > 0.85 * M) & (c < 1.1 * M) & ok
    sig_win = np.sqrt(dijet(p, x[w]).sum()) / ss[w[ok]].sum()
    res[M] = {"window_counting": sig_win, "shape_bkg_known": sig_fixed, "shape_bkg_profiled": sig_fit}
    print(M, "sigma_mu [fb]: window %.3g, shape (bkg known) %.3g, shape (dijet params free) %.3g -> profiled/window %.2f" % (sig_win, sig_fixed, sig_fit, sig_fit / sig_win))
json.dump(res, open(f"{dd}/../fisher_trig{trig}_{ftag}.json", "w"), indent=1)
