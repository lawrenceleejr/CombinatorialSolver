"""F2 item 4b: MC (CMS-EXO-21-004 emulation, 13.6 TeV) x 128 fb-1 vs HEPData observed events (13 TeV).
usage: cms_cmp.py emul.npz [variant ...]"""
import sys, yaml, numpy as np
d = np.load(sys.argv[1]); R = "/home/llee/pairnet_study/results/run3/"
def hep(f):
    y = yaml.safe_load(open(R + f)); x = y["independent_variables"][0]["values"]
    lo = np.array([v.get("low", v.get("value")) for v in x], float); hi = np.array([v.get("high", v.get("value")) for v in x], float)
    dv = {q["header"]["name"]: np.array([v["value"] for v in q["values"]], float) for q in y["dependent_variables"]}
    return lo, hi, dv
L = 128.0
for var in sys.argv[2:] or ["noqgd"]:
  for r, f, rng in ((1, "fig3a.yaml", [(200, 300), (300, 400), (400, 500)]), (2, "fig3b.yaml", [(500, 700), (700, 900)]), (3, "fig3c.yaml", [(900, 1200), (1200, 1600), (1600, 2000)])):
    lo, hi, dv = hep(f); ev = dv["Events"]; e = d[f"edges_r{r}"]; mc = d[f"{var}_r{r}"] * L; er = d[f"{var}_r{r}_err"] * L
    if lo[0] == hi[0]: w = np.diff(lo)[0]; lo, hi = lo - w / 2, lo + w / 2
    for a, b in rng:
        sd = (lo >= a - 1e-6) & (hi <= b + 1e-6); sm = (e[:-1] >= a - 1e-6) & (e[1:] <= b + 1e-6)
        D = ev[sd].sum(); Mc = mc[sm].sum(); Me = np.sqrt((er[sm] ** 2).sum())
        print(f"{var} region{r} m_jjj {a}-{b}: data {D:.0f}  MC {Mc:.0f} +- {Me:.0f}  MC/data {Mc/D:.2f} +- {Me/D:.2f}  (data bins {lo[sd].min():.0f}-{hi[sd].max():.0f}, mc {e[:-1][sm].min():.0f}-{e[1:][sm].max():.0f})")
    print(f"   pthat-bin sums [fb] {var} r{r}:", np.round(d[f"{var}_r{r}_pth"], 2))
