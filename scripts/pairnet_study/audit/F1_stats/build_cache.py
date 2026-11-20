"""Cache the scored npz files (many small files on OSDF) as one npz per sample, keeping the file index of every event."""
import glob, sys, re, os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
O = "/ospool/uc-shared/project/futurecolliders/llee/pairnet"
out = "/scratch/llee/pairnet/audit/F1_stats/data"
COLS = ["m1", "m2", "delta1", "delta2", "d32_1", "d32_2", "d2_63_32", "pt_low6", "valid"]
KEEP = ["ht", "pt6_lead", "w", "r14A_psig"] + [f"r14A_{c}" for c in COLS] + [f"am_{c}" for c in COLS]


def one(f):
    try:
        a = np.load(f)
        d = {k: a[k] for k in KEEP}
        at = (float(a["attr_sigma_mb"]), float(a["attr_weight_sum"]), float(a["attr_n_tried"]), float(a["attr_mass"]), int(a["n_all"]))
        return f, d, at
    except Exception as e:  # noqa: BLE001
        return f, None, repr(e)


def build(name, pattern):
    if os.path.exists(f"{out}/{name}.npz"):
        return
    fs = sorted(glob.glob(pattern), key=lambda f: int(re.search(r"_c(\d+)\.npz$", f).group(1)))
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, fs))
    bad = [(f, at) for f, d, at in res if d is None]
    res = [r for r in res if r[1] is not None]
    cs = np.array([int(re.search(r"_c(\d+)\.npz$", f).group(1)) for f, _, _ in res])
    d = {k: np.concatenate([r[1][k] for r in res]) for k in KEEP}
    d["file_c"] = np.concatenate([np.full(len(r[1]["w"]), c) for r, c in zip(res, cs)])
    at = np.array([r[2] for r in res])
    np.savez(f"{out}/{name}.npz", **d, files_c=cs, files_sigma_mb=at[:, 0], files_weight_sum=at[:, 1], files_n_tried=at[:, 2], files_mass=at[:, 3], files_n_all=at[:, 4])
    print(name, len(fs), "files", len(bad), "bad", len(d["w"]), "events", flush=True)
    for b in bad[:5]:
        print("  bad", b)


jobs = [("qcdvlm", f"{O}/eval_r14/qcdvlm_c*.npz"), ("qcdvlmT", f"{O}/tail/qcdvlmT_c*.npz"), ("qcdhtT", f"{O}/tail/qcdhtT_c*.npz"),
        ("qcd700T", f"{O}/t700/qcd700T_c*.npz"), ("qcd700U", f"{O}/t700/qcd700U_c*.npz")]
for M in (200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000):
    for kind in ("incl", "boost", "power"):
        jobs.append((f"siglm_m{M}_{kind}", f"{O}/eval_r14/siglm_m{M}_{kind}_c*.npz"))
for M in (650, 700, 750):
    for kind in ("incl", "boost"):
        jobs.append((f"sig{M}X_{kind}", f"{O}/t700/sig{M}X_{kind}_c*.npz"))
for n, p in jobs:
    build(n, p)
print("done")
