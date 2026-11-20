"""Build job chunks (npz: pt, eta, phi, m, mask (N,20), w, ht0, aux) for the OSPool variation scan.
usage: prep.py <group>   group in sig, T, U, UX"""
import glob, os, sys, re
import numpy as np, h5py
from multiprocessing import Pool

OUT = "/scratch/llee/pairnet/audit/F3_robust/cache/chunks"
os.makedirs(OUT, exist_ok=True)
P = "/scratch/llee/pairnet"
B = "/ospool/uc-shared/project/futurecolliders/llee/pairnet"
CH = 4000


def cnum(f):
    return int(re.search(r"_c(\d+)", f).group(1))


def read(f, rows=None):
    with h5py.File(f, "r") as h:
        s = h["INPUTS/Source"]
        d = {k: s[n][:] for k, n in (("pt", "pt"), ("eta", "eta"), ("phi", "phi"), ("m", "mass"))}
        d["mask"] = s["MASK"][:].astype(bool)
        d["w"] = h["EventVars/normweight"][:].astype(np.float64)
        at = dict(h.attrs)
    d["row"] = np.arange(len(d["w"]), dtype=np.int32)
    if rows is not None:
        d = {k: v[rows] for k, v in d.items()}
    return d, at


def qcd_file(a):
    f, kind, cut = a
    try:
        if kind == "T":  # tail sample: preselect on the stored nominal scores
            z = np.load(f.replace("_tail.h5", ".npz"))
            ma = 0.5 * (z["r14A_m1"] + z["r14A_m2"])
            rows = np.where((z["r14A_psig"] >= cut[0]) & (ma > cut[1]) & (ma < cut[2]))[0]
            d, at = read(f, rows)
            assert np.allclose((d["pt"] * d["mask"]).sum(1), z["ht"][rows], rtol=1e-4)
            d["p0"] = z["r14A_psig"][rows].astype(np.float32)
        else:
            d, at = read(f)
            ht = (d["pt"] * d["mask"]).sum(1)
            d = {k: v[ht >= cut] for k, v in d.items()}
        d["w"] = d["w"] * float(at["sigma_mb"]) * 1e12 / float(at["weight_sum"])
        d["fidx"] = np.full(len(d["w"]), cnum(f), np.int32)
        return d
    except Exception as e:  # noqa: BLE001
        print("FAILED", f, e, flush=True)
        return None


def write(name, parts, extra=None):
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = len(d["w"])
    rng = np.random.default_rng(7); perm = rng.permutation(n)  # mix so every chunk is a random subset
    d = {k: v[perm] for k, v in d.items()}
    nch = max(1, int(np.ceil(n / CH)))
    for i in range(nch):
        np.savez(f"{OUT}/{name}_{i}.npz", **{k: v[i * CH:(i + 1) * CH] for k, v in d.items()}, **(extra or {}))
    print(name, "events", n, "chunks", nch, "sum w", d["w"].sum(), flush=True)


def qcd(name, files, kind, cut):
    with Pool(6) as pool:
        parts = pool.map(qcd_file, [(f, kind, cut) for f in files], chunksize=8)
    ok = [p for p in parts if p is not None]
    nf = len(ok)
    ok = [p for p in ok if len(p["w"])]
    for p in ok:
        p["w"] = p["w"] / nf
    write(name, ok, {"nfiles": nf})


grp = sys.argv[1]
if grp == "sig":
    for M in (400, 700, 1000):
        parts = []
        nt = {}
        for kind in ("incl", "boost"):
            fs = [f"{P}/prod_lowmass/siglm_m{M}_{kind}_c{c}.h5" for c in (16, 17, 18, 19)]
            pp = []
            nt[kind] = 0
            for f in fs:
                d, at = read(f); nt[kind] += int(at["n_tried"])
                d["fidx"] = np.full(len(d["w"]), cnum(f), np.int32); d["boost"] = np.full(len(d["w"]), kind == "boost"); pp.append(d)
            parts += pp
        for p in parts:
            ht = (p["pt"] * p["mask"]).sum(1)
            hi = ht >= 2.3 * M
            # boost sample is complete for stored-jet H_T >= 2.3 M: combine by tried events there, inclusive only below
            p["w"] = np.where(hi, 1.0 / (nt["incl"] + nt["boost"]), np.where(p["boost"], 0.0, 1.0 / nt["incl"]))
        parts = [{k: v[p["w"] > 0] for k, v in p.items()} for p in parts]
        write(f"sig{M}", parts, {"ntried_incl": nt["incl"], "ntried_boost": nt["boost"]})
    for M in (1500, 2000):
        parts, nt = [], 0
        for f in [f"{P}/prod/sig_m{M}_c9.h5", f"{P}/prod/sig_m{M}_c19.h5"]:
            d, at = read(f); nt += int(at["n_tried"]); d["fidx"] = np.full(len(d["w"]), cnum(f), np.int32); d["boost"] = np.zeros(len(d["w"]), bool)
            parts.append(d)
        for p in parts:
            p["w"] = np.full(len(p["w"]), 1.0 / nt)
        write(f"sig{M}", parts, {"ntried_incl": nt, "ntried_boost": 0})
elif grp == "T":
    qcd("TvlmT", sorted(glob.glob(f"{B}/tail/qcdvlmT_c*_tail.h5")), "T", (0.9, 280, 1150))
    qcd("ThtT", sorted(glob.glob(f"{B}/tail/qcdhtT_c*_tail.h5")), "T", (0.9, 1100, 2600))
    qcd("T700T", sorted(glob.glob(f"{B}/t700/qcd700T_c*_tail.h5")), "T", (0.9, 480, 920))
    qcd("T700U", sorted(glob.glob(f"{B}/t700/qcd700U_c*_tail.h5")), "T", (0.9, 480, 920))
elif grp == "U":
    qcd("Uvlm", [f"{P}/prod_lowmass/qcdvlm_c{c}.h5" for c in range(1000, 1100)], "U", 600.0)
    qcd("Uht", [f"{P}/prod_ht/qcdht_c{c}.h5" for c in range(1400, 2000)], "U", 2000.0)
elif grp == "UX":
    qcd("UXvlm", [f"{P}/prod_lowmass/qcdvlm_c{c}.h5" for c in range(0, 1000)], "U", 600.0)
