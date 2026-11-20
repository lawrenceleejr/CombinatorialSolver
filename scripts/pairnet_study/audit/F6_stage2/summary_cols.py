"""Compact per-mass view of the key ratios (stdout) for REPORT.md."""
import json, glob
def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        for M, v in json.load(open(f)).items(): d.setdefault(M, {}).update(v)
    return dict(sorted(d.items(), key=lambda kv: int(kv[0])))
B, CO, S2 = load("out/full/b34_*.json"), load("out/full/combo_*.json"), load("out/full/s2*.json")
c = lambda v, k="ratio": ("inf" if not v.get("feasible", True) else f"{v[k]:.2f}±{v[k + '_err']:.2f}")
for f in ("f0.1", "f0.05"):
    print(f"== {f}: M | P+all lim(neff) | cats | win | mest|win | wavg|win | best4 | combo ref lim | win+cats")
    for M in sorted(set(B) | set(CO), key=int):
        out = [M]
        if M in B and f in B[M]:
            s = B[M][f]["sel"]; r = s["P+all"]
            x4 = {k: v for k, v in s.items() if k.startswith("P+all+") and k != "P+all+cats" and v.get("feasible", True)}
            b4 = min(x4, key=lambda k: x4[k]["ratio"]) if x4 else None
            out += [f"{r['limit_fb']:.4g}({r['neffB']:.0f})", c(s["P+all+cats"]), c(s["P+all|win"]), c(s["P+all|mest|win"]), c(s["P+all|wavg|win"]), f"{b4[6:]}:{c(x4[b4])}" if b4 else "-"]
        else: out += ["-"] * 6
        if M in CO and f in CO[M]:
            s = CO[M][f]["sel"]; out += [f"{s['P+all+cats']['limit_fb']:.4g}±{100 * s['P+all+cats']['limit_relerr']:.0f}%", c(s["P+all|win+cats"])]
        print(" | ".join(out))
for f in ("f0.1", "f0.05"):
    print(f"== stage2 {f} (ratio vs P+all; dr2/lin2 also vs P)")
    for M, r in S2.items():
        if f not in r: continue
        s = r[f]["sel"]
        print(M, f"P+all {s['P+all']['limit_fb']:.4g}({s['P+all']['neffB']:.0f}) P {c(s['P'])} cats {c(s['P+all+cats'])} | " + " ".join(f"{k} {c(v)}" for k, v in s.items() if not k.startswith("P")) + f" | dr2/P {c(s['dr2'], 'ratio_vs_P')} lin2/P {c(s['lin2'], 'ratio_vs_P')}")
