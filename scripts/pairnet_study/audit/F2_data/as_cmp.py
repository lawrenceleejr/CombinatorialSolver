"""F2 item 4a/5: effect of the shower alpha_s (FSR=ISR = 0.123 / 0.1365 nominal / 0.150) on QCD rate and signal efficiency
along the r14A cut ladder. QCD: prod-type samples (pTHat>200, no filter), H_T>1200 (where pTHat>200 is complete)."""
import numpy as np, glob, json
def load(pat):
    fs = sorted(glob.glob(pat)); o = {k: [] for k in ("ht", "p", "m", "A", "v", "w")}; nt = 0; nall = 0
    for f in fs:
        a = np.load(f); k = float(a["attr_sigma_mb"]) * 1e12 / float(a["attr_weight_sum"])
        m1, m2 = a["r14A_m1"], a["r14A_m2"]; o["ht"].append(a["ht"]); o["p"].append(a["r14A_psig"]); o["m"].append(0.5 * (m1 + m2))
        o["A"].append(np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)); o["v"].append(a["r14A_valid"] > 0); o["w"].append(a["w"] * k)
        nt += float(a["attr_n_tried"]); nall += int(a["n_all"])
    o = {k: np.concatenate(v) for k, v in o.items()}; o["w"] = o["w"] / max(len(fs), 1); o["nf"] = len(fs); o["nt"] = nt; o["nall"] = nall
    return o
def rate(o, M, ht, p, sig=False):
    s = o["v"] & (o["A"] < 0.1) & (o["m"] > 0.85 * M) & (o["m"] < 1.1 * M) & (o["ht"] > ht) & (o["p"] > p)
    if sig: n = int(s.sum()); return n / o["nt"], np.sqrt(max(n, 1)) / o["nt"], n      # efficiency per generated event
    w = o["w"][s]; return w.sum(), np.sqrt((w ** 2).sum()), int(s.sum())
res = {}
Q = {t: load(f"as/{t}/qcd_c*.npz") for t in ("as123", "nom", "as150")}
print("QCD files:", {t: (q["nf"], len(q["w"])) for t, q in Q.items()})
for M, ht in ((700, 1200), (1500, 1200)):
    print(f"\nQCD M={M}: valid, A<0.1, {0.85*M:.0f}<m_avg<{1.1*M:.0f}, H_T>{ht}  [fb]")
    for p in (-1, 0.5, 0.9, 0.97, 0.99):
        r = {t: rate(q, M, ht, p) for t, q in Q.items()}; line = f"  P>{p:5.2f}: " + "  ".join(f"{t} {r[t][0]:10.2f}+-{r[t][1]:8.2f} (n={r[t][2]})" for t in r)
        for a, b in (("as123", "nom"), ("as150", "nom"), ("as150", "as123")):
            if r[a][2] > 2 and r[b][2] > 2:
                x = r[a][0] / r[b][0]; e = x * np.hypot(r[a][1] / r[a][0], r[b][1] / r[b][0]); line += f"  {a}/{b} {x:.2f}+-{e:.2f}"; res[f"qcd_{M}_{p}_{a}/{b}"] = (x, e)
        print(line)
print("\ninclusive QCD: no cut besides >=6 jets: ", {t: round(float(q["w"].sum()), 1) for t, q in Q.items()})
for M, pf in ((500, 0.9965), (1000, 0.9982), (1500, 0.998), (2000, 0.998)):
    S = {t: load(f"as/{t}/sig_m{M}_c*.npz") for t in ("as123", "nom", "as150")}
    print(f"\nsignal M={M}: efficiency per generated event (files {[s['nf'] for s in S.values()]})")
    for p in (-1, 0.5, 0.9, 0.99, pf):
        r = {t: rate(s, M, 0, p, True) for t, s in S.items() if s["nf"]}; line = f"  P>{p:6.4f}: " + "  ".join(f"{t} {r[t][0]:.4f}+-{r[t][1]:.4f}" for t in r)
        for a, b in (("as123", "nom"), ("as150", "nom"), ("as150", "as123")):
            if a in r and b in r and r[a][2] > 2 and r[b][2] > 2:
                x = r[a][0] / r[b][0]; e = x * np.hypot(r[a][1] / r[a][0], r[b][1] / r[b][0]); line += f"  {a}/{b} {x:.2f}+-{e:.2f}"; res[f"sig_{M}_{p}_{a}/{b}"] = (x, e)
        print(line)
    # peak position / resolution of the selected candidates (P>0.9)
    for t, s in S.items():
        if not s["nf"]: continue
        k = s["v"] & (s["A"] < 0.1) & (s["p"] > 0.9) & (s["m"] > 0.6 * M) & (s["m"] < 1.3 * M)
        if k.sum() > 20: q = np.percentile(s["m"][k] / M, [16, 50, 84]); print(f"    {t}: m_avg/M median {q[1]:.3f}, half-width68 {(q[2]-q[0])/2:.3f} (n={k.sum()})")
json.dump(res, open("as_cmp.json", "w"), indent=1)
