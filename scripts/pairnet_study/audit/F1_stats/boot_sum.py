"""Low-mass bootstrap summary: spread of the published procedure (best-of on evaluated limits), nested choice, bias."""
import glob, json, sys
import numpy as np
orig = json.load(open("rerun_main.json"))
gm = lambda v: float(np.exp(np.mean(np.log(v))))
for M in (400, 700, 1000):
    R = [json.loads(l) for f in sorted(glob.glob(f"boot_low_m{M}_s*.jsonl")) for l in open(f)]
    if not R:
        continue
    pub, nest, pubc, nestc, fix = [], [], [], [], []
    for r in R:
        E = [(t, c, x) for t, v in r.items() for c, x in v.items() if not c.endswith("+cats")]
        C = [(t, c, x) for t, v in r.items() for c, x in v.items() if c.endswith("+cats")]
        pub.append(min(x["limit_fb"] for _, _, x in E))
        nest.append(gm([min(E, key=lambda e: e[2]["tune"][h])[2]["lim"][h] for h in (0, 1)]))
        fix.append(r["450"]["P+all"]["limit_fb"])
        if C:
            pubc.append(min(x["limit_fb"] for _, _, x in C))
            nestc.append(gm([min(C, key=lambda e: e[2]["tune_u"][h])[2]["lim_u"][h] for h in (0, 1)]))
    o = orig[str(M)]; trigs = list(r)
    po = min(x["limit_fb"] for t in trigs for c, x in o[t].items() if not c.endswith("+cats"))
    pco = min(x["limit_fb"] for t in trigs for c, x in o[t].items() if c.endswith("+cats"))
    print(f"M {M}: N {len(R)}, triggers {trigs}")
    for name, v, ref in (("published single", pub, po), ("nested single", nest, None), ("fixed P+all@450", fix, o["450"]["P+all"]["limit_fb"]), ("published cats", pubc, pco), ("nested cats", nestc, None)):
        v = np.array(v); f = np.isfinite(v); lv = np.log(v[f])
        print(f"   {name:18s}: median {np.median(v):.4g}  std(log) {lv.std():.3f}  mean {np.exp(lv.mean()):.4g}" + (f"  original {ref:.4g}" if ref else "") + f"  inf {1 - f.mean():.2f}")
    lr = np.log(np.array(pub) / np.array(nest)); lc = np.log(np.array(pubc) / np.array(nestc)) if pubc else np.array([np.nan])
    print(f"   best-of bias: mean log(pub/nested) single {lr.mean():+.3f} ± {lr.std() / np.sqrt(len(lr)):.3f}, cats {np.nanmean(lc):+.3f} ± {np.nanstd(lc) / np.sqrt(len(lc)):.3f}")
    rr = np.log([l / t for r in R for v in r.values() for c, x in v.items() if not c.endswith("+cats") for l, t in zip(x["lim"], x["tune"]) if np.isfinite(l) and np.isfinite(t)])
    rb = np.log([b / bt for r in R for v in r.values() for c, x in v.items() if not c.endswith("+cats") for b, bt in zip(x["B"], x["Bt"]) if b > 0 and bt > 0])
    print(f"   eval/tune: geo-mean limit {np.exp(rr.mean()):.3f}, B {np.exp(rb.mean()):.3f} (n {len(rr)})")
