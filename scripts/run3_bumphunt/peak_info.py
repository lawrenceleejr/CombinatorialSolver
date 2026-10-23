"""Share of the profiled shape-fit information carried by correctly paired signal, and the ceiling if every selected
signal event were correctly paired (same efficiency, narrow-peak shape). Background from the Fisher-tuned QCD dump."""
import json, sys
import h5py, numpy as np
sys.argv += []
P = "/scratch/llee/pairnet/prod_lowmass"; E = "/ospool/uc-shared/project/futurecolliders/llee/pairnet/eval_r14"
fj = json.load(open("lowmass14fisher_pk0.75/lowmass_eval_r14A_trig450_fisher.json"))["results"]
FB = np.linspace(0.4, 2.0, 49); FC = 0.5 * (FB[1:] + FB[:-1]); L = 300.0
def basis(x): return np.stack([np.ones_like(x), np.log1p(-x), np.log(x), np.log(x) ** 2], 1)
out = {}
for M in (300, 500, 700, 1000):
    x = fj[str(M)]; conf = min([k for k in ("P", "P+boost", "P+all")], key=lambda k: x[k]["limit_fb"]); cuts = x[conf]["halves"][0]["cuts"]
    q = np.load(f"lowmass14fisher_pk0.75/dump/spec_r14A_m{M}_trig450_f0.npz", allow_pickle=True)
    qh, _ = np.histogram(q["qcd_mavg"] / M, FB, weights=q["qcd_w"] * L); q2, _ = np.histogram(q["qcd_mavg"] / M, FB, weights=(q["qcd_w"] * L) ** 2)
    i0 = int(np.argmax(np.convolve(qh, np.ones(3) / 3, mode="same"))); fit = (np.arange(48) >= i0) & (qh > 0); use = np.arange(48) >= i0
    X = basis(FC * M / 13600); wt = qh[fit] / np.sqrt(q2[fit])
    th, *_ = np.linalg.lstsq(X[fit] * wt[:, None], np.log(qh[fit]) * wt, rcond=None); b = np.exp(X[use] @ th)
    allm, cm, ntried = [], [], 0
    for c in (16, 17, 18, 19):
        h = h5py.File(f"{P}/siglm_m{M}_incl_c{c}.h5"); a = np.load(f"{E}/siglm_m{M}_incl_c{c}.npz"); ntried += float(a["attr_n_tried"])
        S = h["INPUTS/Source"]; pt, eta, phi, m = [S[k][:] for k in ("pt", "eta", "phi", "mass")]
        J = np.stack([np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], -1)
        t1 = np.stack([h[f"TARGETS/g1/j{k}"][:] for k in (1, 2, 3)], 1); t2 = np.stack([h[f"TARGETS/g2/j{k}"][:] for k in (1, 2, 3)], 1)
        full = (t1 >= 0).all(1) & (t2 >= 0).all(1)
        m3 = lambda idx: (lambda v: np.sqrt(np.maximum(v[:, 0] ** 2 - (v[:, 1:] ** 2).sum(1), 0)))(np.take_along_axis(J, np.maximum(idx, 0)[:, :, None], 1).sum(1))
        tm1, tm2 = m3(t1), m3(t2); m1, m2 = a["r14A_m1"], a["r14A_m2"]
        same = full & ((np.isclose(m1, tm1, rtol=1e-4) & np.isclose(m2, tm2, rtol=1e-4)) | (np.isclose(m1, tm2, rtol=1e-4) & np.isclose(m2, tm1, rtol=1e-4)))
        A = np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)
        sel = (A < 0.1) & a["r14A_valid"].astype(bool) & (a["ht"] > 450) & (a["r14A_psig"] >= cuts.get("psig", -1)) & (a["ht"] / M > cuts.get("ht", 0))
        allm.append(0.5 * (m1 + m2)[sel] / M); cm.append(0.5 * (m1 + m2)[sel & same] / M)
    allm, cm = np.concatenate(allm), np.concatenate(cm)
    def info(vals, scale=1.0):
        s, _ = np.histogram(vals, FB); s = s / ntried * L * scale
        A_ = X[use] * np.sqrt(b)[:, None]; v = s[use] / np.sqrt(b); r = v - A_ @ np.linalg.lstsq(A_, v, rcond=None)[0]; return float(r @ r)
    ia, ic = info(allm), info(cm)
    ceil = info(cm, scale=len(allm) / max(len(cm), 1))  # all selected events with the correct-pairing shape
    out[M] = {"info_correct_over_all": ic / ia, "ceiling_sigma_ratio": float(np.sqrt(ia / ceil)), "correct_frac": len(cm) / len(allm)}
    print(M, conf, "correct frac %.2f; share of fit information from correct pairings %.2f; sigma_mu if all were correctly paired: x%.2f" % (len(cm) / len(allm), ic / ia, np.sqrt(ia / ceil)))
json.dump(out, open("falling/anatomy/peak_info.json", "w"), indent=1)
