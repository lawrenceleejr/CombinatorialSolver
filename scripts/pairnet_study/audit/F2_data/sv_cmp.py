"""F2 item 5: signal-model variations (width, cascade decay via squark, heavy flavour) vs nominal, r14A cut ladder.
Efficiency per generated event: valid, A<0.1, 0.85M<m_avg<1.1M, H_T>450, P>cut.  usage: sv_cmp.py dir M pfinal tag1 tag2 ..."""
import sys, glob, json, numpy as np
D, M, pf = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); tags = sys.argv[4:]
def load(t):
    o = {k: [] for k in ("ht", "p", "m", "A", "v")}; nt = 0
    for f in sorted(glob.glob(f"{D}/sv_{t}_m{M:.0f}*.npz")):
        a = np.load(f); m1, m2 = a["r14A_m1"], a["r14A_m2"]; o["ht"].append(a["ht"]); o["p"].append(a["r14A_psig"]); o["m"].append(0.5 * (m1 + m2))
        o["A"].append(np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)); o["v"].append(a["r14A_valid"] > 0); nt += float(a["attr_n_tried"])
    o = {k: np.concatenate(v) for k, v in o.items()}; o["nt"] = nt; return o
S = {t: load(t) for t in tags}; res = {}
def eff(o, p, lo=0.85, hi=1.1):
    s = o["v"] & (o["A"] < 0.1) & (o["m"] > lo * M) & (o["m"] < hi * M) & (o["ht"] > 450) & (o["p"] > p); n = int(s.sum()); return n / o["nt"], np.sqrt(max(n, 1)) / o["nt"], n
print(f"M={M:.0f}  generated events:", {t: int(s["nt"]) for t, s in S.items()})
for p in (-1, 0.5, 0.9, 0.99, pf):
    r = {t: eff(s, p) for t, s in S.items()}; line = f" P>{p:7.4f}: " + " ".join(f"{t} {100*r[t][0]:.3f}+-{100*r[t][1]:.3f}%" for t in r) + "  | ratio to nom:"
    for t in tags[1:]:
        x = r[t][0] / r[tags[0]][0]; e = x * np.hypot(r[t][1] / max(r[t][0], 1e-12), r[tags[0]][1] / r[tags[0]][0]); line += f" {t} {x:.2f}+-{e:.2f}"; res[f"{t}_P{p}"] = (x, e)
    print(line)
for t, s in S.items():
    k = s["v"] & (s["A"] < 0.1) & (s["p"] > 0.9) & (s["m"] > 0.6 * M) & (s["m"] < 1.3 * M); q = np.percentile(s["m"][k] / M, [16, 50, 84])
    print(f"   {t}: P>0.9 candidates m_avg/M median {q[1]:.3f} half-width68 {(q[2]-q[0])/2:.3f} (n={k.sum()}); P median of valid A<0.1 window events {np.median(s['p'][s['v'] & (s['A']<0.1) & (s['m']>0.85*M) & (s['m']<1.1*M)]):.3f}")
json.dump(res, open(f"sv_cmp_m{M:.0f}.json", "w"), indent=1)
