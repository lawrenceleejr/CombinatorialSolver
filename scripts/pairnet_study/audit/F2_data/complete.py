"""F2 item 3: p_T-hat threshold completeness by comparing independent samples with different thresholds in common regions.
Rates in fb after: valid, A<0.1, m_avg window, H_T>cut, P>cut. Errors: sqrt(sum w^2)."""
import numpy as np, json
C = {}
def L(n):
    if n not in C:
        d = np.load(f"cache/{n}.npz"); C[n] = {k: d[k] for k in ("ht", "r14A_psig", "r14A_m1", "r14A_m2", "r14A_valid", "wfb")}
        c = C[n]; c["mavg"] = 0.5 * (c["r14A_m1"] + c["r14A_m2"]); c["A"] = np.abs(c["r14A_m1"] - c["r14A_m2"]) / np.maximum(c["r14A_m1"] + c["r14A_m2"], 1e-9)
    return C[n]
def rate(n, lo, hi, ht, p, ht_hi=1e9):
    c = L(n); s = (c["r14A_valid"] > 0) & (c["A"] < 0.1) & (c["mavg"] > lo) & (c["mavg"] < hi) & (c["ht"] > ht) & (c["ht"] < ht_hi) & (c["r14A_psig"] > p)
    w = c["wfb"][s]; return w.sum(), np.sqrt((w ** 2).sum()), int(s.sum())
out = []
def cmp(a, b, lo, hi, hts, ps, note):
    print(f"\n{note}: {a} / {b}, {lo}<m_avg<{hi}")
    for ht in hts:
        for p in ps:
            ra, ea, na = rate(a, lo, hi, ht, p); rb, eb, nb = rate(b, lo, hi, ht, p)
            if na < 5 or nb < 5: continue
            r = ra / rb; e = r * np.hypot(ea / ra, eb / rb)
            print(f"  H_T>{ht:5.0f} P>{p:6.4f}: {a} {ra:10.3f}+-{ea:8.3f} fb (n={na})  {b} {rb:10.3f}+-{eb:8.3f} fb (n={nb})  ratio {r:.3f} +- {e:.3f}")
            out.append(dict(a=a, b=b, lo=lo, hi=hi, ht=ht, p=p, ra=ra, ea=ea, rb=rb, eb=eb, ratio=r, err=e))
import sys
tests = sys.argv[1:] or ["1", "2", "3", "4"]
if "1" in tests: cmp("htT", "vlmT", 850, 1100, (900, 1050, 1200, 1500, 2000), (0.9, 0.99, 0.997), "pTHat>200 (htT) vs pTHat>80 (vlmT)")
if "2" in tests: cmp("q700T", "vlmT", 595, 770, (900, 1050, 1200, 1500), (0.9, 0.99, 0.9966), "pTHat>120 (700T) vs pTHat>80 (vlmT)")
if "3" in tests: cmp("q700U", "q700T", 595, 770, (1200, 1500, 2000), (0.9, 0.99, 0.9966), "pTHat>250 (700U) vs pTHat>120 (700T)")
if "4" in tests: cmp("vlmT", "vlm", 595, 770, (450, 900), (0.9, 0.99, 0.9966), "tail-filter closure (vlmT vs vlm, same generator settings)")
json.dump(out, open("complete_%s.json" % "".join(tests), "w"), indent=1)
