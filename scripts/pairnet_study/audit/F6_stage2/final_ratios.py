"""Item 1 vs the current recipe (P+all+cats): r(X)/r(P+all+cats), errors added in quadrature (ignores the positive
correlation of the paired bootstrap, hence conservative)."""
import json, glob, math
def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        for M, v in json.load(open(f)).items(): d.setdefault(M, {}).update(v)
    return dict(sorted(d.items(), key=lambda kv: int(kv[0])))
S2 = load("out/full/s2*.json")
for X in ("mlp_core+cats", "gbdt_ext+cats", "dr2+cats", "lin2"):
    print("==", X, "/ P+all+cats   f0.1 | f0.05   [lim fb new, eval neff]")
    gm = {"f0.1": [], "f0.05": []}
    for M, r in S2.items():
        out = [M]
        for f in ("f0.1", "f0.05"):
            if f not in r: out.append("-"); continue
            s = r[f]["sel"]; a, b = s[X], s["P+all+cats"]
            if not a.get("feasible", True): out.append("infeas"); continue
            q = a["ratio"] / b["ratio"]; e = q * math.hypot(a["ratio_err"] / a["ratio"], b["ratio_err"] / b["ratio"])
            gm[f].append(math.log(q)); out.append(f"{q:.2f}±{e:.2f} [{a['limit_fb']:.3g}, {a['neffB']:.0f}/{a.get('neffB_top', float('nan')):.0f}]")
        print(" | ".join(out))
    print("geo-mean", {f: round(math.exp(sum(v) / len(v)), 3) for f, v in gm.items() if v})
