"""F2 item 5: sensitivity of r14A signal efficiency to the ISR model: siglm 'boost' (default wimpy ISR) vs 'power' (SpaceShower:pTmaxMatch=2),
same generator-level H_T > 2.3 M filter.  Efficiencies are per generated (tried) event."""
import glob, json, numpy as np
O = "/ospool/uc-shared/project/futurecolliders/llee/pairnet/eval_r14"
res = json.load(open("/scratch/llee/pairnet/lowmass14win10t_pk0.75/lowmass_eval_r14A_trig450_f0.1.json"))["results"]
def load(pat):
    d = {}; nt = 0
    for f in sorted(glob.glob(pat)):
        a = np.load(f); nt += float(a["attr_n_tried"])
        for k in ("ht", "r14A_psig", "r14A_m1", "r14A_m2", "r14A_valid", "pt6_lead"): d.setdefault(k, []).append(a[k])
    return {k: np.concatenate(v) for k, v in d.items()}, nt
out = {}
print(f"{'M':>5} {'cut':>22} {'eff boost':>20} {'eff power':>20} {'power/boost':>16}")
for M in (200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000):
    b, nb = load(f"{O}/siglm_m{M}_boost_c1[6-9].npz"); p, np_ = load(f"{O}/siglm_m{M}_power_c*.npz")
    try: pc = res[str(M)]["P"]["halves"][0]["cuts"]["psig"]
    except Exception: pc = None
    row = {}
    def eff(d, n, pcut, htmin=0):
        m = 0.5 * (d["r14A_m1"] + d["r14A_m2"]); a = np.abs(d["r14A_m1"] - d["r14A_m2"]) / np.maximum(d["r14A_m1"] + d["r14A_m2"], 1e-9)
        s = d["r14A_valid"] & (a < 0.1) & (m > 0.85 * M) & (m < 1.1 * M) & (d["r14A_psig"] > pcut) & (d["ht"] > htmin)
        k = s.sum(); return k / n, np.sqrt(k) / n
    for name, pcut in (("gen HT filter only", None), ("A<0.1 & window", -1), ("+P>0.5", 0.5), ("+P>0.9", 0.9), ("+P>0.99", 0.99), (f"+P>final", pc)):
        if name.startswith("gen"):
            eb, ep = (len(b["ht"]) / nb, np.sqrt(len(b["ht"])) / nb), (len(p["ht"]) / np_, np.sqrt(len(p["ht"])) / np_)
        elif pcut is None: continue
        else: eb, ep = eff(b, nb, pcut), eff(p, np_, pcut)
        r = ep[0] / eb[0] if eb[0] > 0 else float("nan"); dr = r * np.sqrt((eb[1] / eb[0]) ** 2 + (ep[1] / ep[0]) ** 2) if eb[0] > 0 and ep[0] > 0 else float("nan")
        row[name] = [eb[0], eb[1], ep[0], ep[1], r, dr]
        print(f"{M:5d} {name + ('(%.4f)' % pcut if name=='+P>final' else ''):>22} {eb[0]:10.3e}±{eb[1]:8.1e} {ep[0]:10.3e}±{ep[1]:8.1e} {r:8.3f}±{dr:5.3f}")
    out[M] = row
json.dump(out, open("sig_real.json", "w"), indent=1)
