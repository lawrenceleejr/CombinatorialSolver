import glob, json, h5py, numpy as np
th = json.load(open("/scratch/llee/pairnet/xs_theory/pp13600_gluino_NNLO+NNLL.json"))
print({k: (v if k != "data" else "...") for k, v in th.items()})
d = th["data"]; tm = np.array(sorted(int(float(k)) for k in d)); tx = np.array([d[str(m)]["xsec_pb"] * 1e3 for m in tm])
print("table masses", tm[:8], "...", tm[-3:], "keys", list(d[str(tm[0])].keys()))
LM = "/scratch/llee/pairnet/prod_lowmass"
def lo(M, n=5, kind="incl"):
    fs = sorted(glob.glob(f"{LM}/siglm_m{M}_{kind}_c*.h5"))[:n]
    v = [h5py.File(f).attrs["sigma_mb"] * 1e12 for f in fs]
    return np.mean(v), np.std(v) / np.sqrt(len(v)), len(fs)
KM = [500, 600, 700, 800, 900, 1000]
ks = []
for m in [200, 250, 300, 350, 400, 450] + KM:
    l, e, n = lo(m); nn = np.exp(np.interp(m, tm, np.log(tx))) if m >= tm[0] else np.nan
    print(f"M {m}: LO {l:.4g} ± {e:.2g} fb (n={n})  NNLL {nn:.4g}  K {nn / l:.3f}  rel unc table {d.get(str(m), {})}")
    if m in KM: ks.append(np.log(nn / l))
kfit = np.polyfit(KM, ks, 1)
for m in [200, 250, 300, 350, 400, 450, 500]:
    print(m, "K extrapolated", round(float(np.exp(np.polyval(kfit, m))), 3), "sigma_th", lo(m)[0] * np.exp(np.polyval(kfit, m)))
f = h5py.File(sorted(glob.glob(f"{LM}/siglm_m700_incl_c*.h5"))[0]); print(dict(f.attrs))
f = h5py.File(sorted(glob.glob(f"{LM}/siglm_m700_boost_c*.h5"))[0]); print(dict(f.attrs))
