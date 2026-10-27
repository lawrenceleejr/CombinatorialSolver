"""Closure: cross section of QCD in tail-filter subregions from the existing samples vs the enriched tail samples."""
import glob, json, sys
import h5py, numpy as np
sys.path.insert(0, "/tmp/claude-12421/-scratch-llee/a35d08e7-9d87-5923-8644-0d53c5b0f767/scratchpad")
import tailmerge as TM
C = "/scratch/llee/pairnet/cuts_study"; T = "/ospool/uc-shared/project/futurecolliders/llee/pairnet/tail"
P3 = "/scratch/llee/pairnet/data_r3/prod"; E = "/ospool/uc-shared/project/futurecolliders/llee/pairnet/eval_r14"
fac = {f: h5py.File(f).attrs["sigma_mb"] / h5py.File(f).attrs["weight_sum"] * 1e12 for f in sorted(glob.glob(f"{P3}/qcd_c*.h5"))}
pnf = [f for f in fac if f[-4] in "89"]; k_old = np.mean([fac[f] for f in pnf]) / len(pnf)
full = np.load(f"{C}/big_vars.npz"); HT = np.load(f"{C}/ht_all.npy")[full["is_signal"] | np.isin(full["fileclass"], [8, 9])]
v = np.load(f"{C}/big_vars_r14A_pn.npz"); q = ~v["is_signal"]
old = dict(m1=v["pn_m1"][q], m2=v["pn_m2"][q], p=v["psig"][q], valid=v["pn_valid"][q].astype(bool), ht=HT[q], w=v["w"][q] * k_old)
h = np.load(f"{C}/big_vars_ht_r14A_ge1400.npz")
ht = dict(m1=h["pn_m1"], m2=h["pn_m2"], p=h["psig"], valid=h["pn_valid"].astype(bool), ht=h["ht"], w=h["w"] * h["fac"] / int(h["n_files"]))
vparts, nv = [], 0
for f in sorted(glob.glob(f"{E}/qcdvlm_c*.npz")):
    a = np.load(f); nv += 1
    if len(a["w"]) == 0: continue
    vparts.append(dict(m1=a["r14A_m1"], m2=a["r14A_m2"], p=a["r14A_psig"], valid=a["r14A_valid"].astype(bool), ht=a["ht"],
                       w=a["w"] * float(a["attr_sigma_mb"]) * 1e12 / float(a["attr_weight_sum"])))
vlm = {k: np.concatenate([x[k] for x in vparts]) for k in vparts[0]}; vlm["w"] /= nv
tails = {}
for n in ("qcdvlmT", "qcdhtT"):
    t, nf = TM.load_tail(T, n, "r14A", 900 if n == "qcdvlmT" else 500)
    tails[n] = dict(m1=t["r14A_m1"], m2=t["r14A_m2"], p=t["r14A_psig"], valid=t["r14A_valid"].astype(bool), ht=t["ht"], w=t["wfb"])


def xs(d, htlo, hthi, plo, mlo, mhi):
    ma = 0.5 * (d["m1"] + d["m2"]); A = np.abs(d["m1"] - d["m2"]) / np.maximum(d["m1"] + d["m2"], 1e-9)
    k = d["valid"] & (A < 0.1) & (d["p"] >= plo) & (ma > mlo) & (ma < mhi) & (d["ht"] >= htlo) & (d["ht"] < hthi)
    w = d["w"][k]; ne = w.sum() ** 2 / max((w ** 2).sum(), 1e-300)
    return w.sum(), ne


tests = [(400, 900, 0.8, 300, 600, ["vlm", "qcdvlmT"]), (400, 900, 0.95, 300, 600, ["vlm", "qcdvlmT"]),
         (900, 1200, 0.8, 400, 800, ["old", "vlm", "qcdvlmT", "qcdhtT"]), (1200, 1e9, 0.8, 800, 1600, ["old", "ht", "vlm", "qcdvlmT", "qcdhtT"]),
         (1200, 1e9, 0.99, 800, 1600, ["old", "ht", "vlm", "qcdvlmT", "qcdhtT"]), (1200, 1e9, 0.99, 1600, 3000, ["old", "ht", "vlm", "qcdvlmT", "qcdhtT"])]
S = {"old": old, "ht": ht, "vlm": vlm, **tails}
for lo, hi, pl, ml, mh, names in tests:
    out = []
    for n in names:
        x, ne = xs(S[n], lo, hi, pl, ml, mh)
        out.append(f"{n}: {x:.3g} fb (neff {ne:.0f}, ±{100 / np.sqrt(max(ne, 1)):.0f}%)")
    print(f"H_T {lo}-{hi:g}, P>={pl}, m_avg {ml}-{mh}:  " + "  ".join(out))
