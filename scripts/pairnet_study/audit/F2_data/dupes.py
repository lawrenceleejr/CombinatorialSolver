"""F2 item 1: event-level duplicate search over every sample used for training, validation, test and limits.

Keys (uint64 hashes):
  KJ  h5 only: bytes of (pt, eta, phi) of the six leading stored jets          -> exact event identity
  KU  h5 and scored npz: (H_T float32, pT of 6th leading jet float32, generator weight float64) -> links npz to h5
      (for unweighted signal the weight is constant and KU is weaker; signal families are also tested with KM)
  KM  scored npz only: (H_T, pt6, m1, m2 of the argmin pairing)                 -> exact identity among npz
usage: dupes.py outdir nproc
"""
import glob, json, os, sys
from multiprocessing import Pool
import h5py
import numpy as np

out, nproc = sys.argv[1], int(sys.argv[2])
P = "/scratch/llee/pairnet"; O = "/ospool/uc-shared/project/futurecolliders/llee/pairnet"
C1 = np.uint64(0x9E3779B97F4A7C15); C2 = np.uint64(0xC2B2AE3D27D4EB4F); C3 = np.uint64(0x165667B19E3779F9)
rs = np.random.RandomState(12345).randint(1, 2 ** 62, size=64).astype(np.uint64) * np.uint64(2) + np.uint64(1)


def mix(cols):
    """cols: (N, k) uint64 -> (N,) uint64 multiply-xorshift hash."""
    h = np.zeros(len(cols), np.uint64)
    with np.errstate(over="ignore"):
        for i in range(cols.shape[1]):
            h = (h ^ (cols[:, i] * rs[i])) * C1
            h ^= h >> np.uint64(29)
        h = h * C2; h ^= h >> np.uint64(32)
    return h


def u32(x):
    return np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)


def u64(x):
    return np.ascontiguousarray(x, np.float64).view(np.uint64)


def one(job):
    fam, kind, f = job
    try:
        if kind == "h5":
            with h5py.File(f, "r") as h:
                s = h["INPUTS/Source"]; pt, eta, phi, mask = s["pt"][:], s["eta"][:], s["phi"][:], s["MASK"][:].astype(bool)
                w = h["EventVars/normweight"][:]
            order = np.argsort(np.where(mask, -pt, np.inf), axis=1, kind="stable")
            pts = np.take_along_axis(pt, order, 1); ms = np.take_along_axis(mask, order, 1)
            kj = mix(np.concatenate([u32(x[:, :6]) for x in (pt, eta, phi)], 1))
            ht = (pts * ms).sum(1).astype(np.float32); pt6 = pts[:, 5].astype(np.float32)
            ku = mix(np.stack([u32(ht), u32(pt6), u64(w)], 1))
            return fam, f, len(pt), kj, ku, None
        z = np.load(f)
        if len(z["w"]) == 0:
            return fam, f, 0, None, np.zeros(0, np.uint64), np.zeros(0, np.uint64)
        ku = mix(np.stack([u32(z["ht"]), u32(z["pt6_lead"]), u64(z["w"])], 1))
        km = mix(np.stack([u32(z["ht"]), u32(z["pt6_lead"]), u32(z["am_m1"]), u32(z["am_m2"])], 1))
        return fam, f, len(ku), None, ku, km
    except Exception as e:  # noqa: BLE001
        return fam, f, -1, None, None, str(e)


def fam_of(path):
    b = os.path.basename(path)
    for suf in ("_tail.h5", ".h5", ".npz"):
        if b.endswith(suf):
            b = b[: -len(suf)]; break
    return b.rsplit("_c", 1)[0]


jobs = []
def add(tag, pattern, kind):
    for f in sorted(glob.glob(pattern)):
        jobs.append((f"{tag}:{fam_of(f)}", kind, f))


add("prod", f"{P}/prod/*.h5", "h5"); add("data_r3", f"{P}/data_r3/prod/*.h5", "h5"); add("data_r1", f"{P}/data_r1/prod/*.h5", "h5")
add("prod_ht", f"{P}/prod_ht/*.h5", "h5"); add("prod_lowmass", f"{P}/prod_lowmass/*.h5", "h5"); add("prod_sighi", f"{P}/prod_sighi/*.h5", "h5")
add("as123", f"{P}/prod_as123/*.h5", "h5"); add("as150", f"{P}/prod_as150/*.h5", "h5")
add("spill", f"{O}/prod_lowmass_spill/*.h5", "h5")
add("lmnpz", f"{P}/prod_lowmass/*.npz", "npz"); add("eval_r14", f"{O}/eval_r14/*.npz", "npz")
add("tail", f"{O}/tail/*_c*[0-9].npz", "npz"); add("t700", f"{O}/t700/*_c*[0-9].npz", "npz")
add("tailh5", f"{O}/tail/*_tail.h5", "h5"); add("t700h5", f"{O}/t700/*_tail.h5", "h5")
print(len(jobs), "files", flush=True)
res = {}
bad = []
with Pool(nproc) as pool:
    for i, (fam, f, n, kj, ku, km) in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
        if n < 0:
            bad.append((f, km)); continue
        d = res.setdefault(fam, {"files": [], "n": [], "kj": [], "ku": [], "km": []})
        d["files"].append(f); d["n"].append(n)
        if kj is not None: d["kj"].append(kj)
        if ku is not None: d["ku"].append(ku)
        if km is not None and not isinstance(km, str): d["km"].append(km)
        if i % 5000 == 0:
            print(i, flush=True)
np.save(f"{out}/dupes_bad.npy", np.array(bad, dtype=object), allow_pickle=True)
summary = {"bad_files": len(bad), "families": {}}
allk = {"kj": [], "ku": [], "km": []}
for fam, d in sorted(res.items()):
    s = {"files": len(d["files"]), "events": int(sum(d["n"]))}
    fid = np.repeat(np.arange(len(d["files"])), d["n"])
    for k in ("kj", "ku", "km"):
        if d[k]:
            a = np.concatenate(d[k]); u, c = np.unique(a, return_counts=True)
            s[f"{k}_dupes_within"] = int((c - 1).sum())
            if (c > 1).any() and len(a) == len(fid):
                # how many of the repeated keys are within one file vs across files
                rep = np.isin(a, u[c > 1]); o = np.argsort(a[rep], kind="stable"); aa, ff = a[rep][o], fid[rep][o]
                same = (aa[1:] == aa[:-1]) & (ff[1:] == ff[:-1]); s[f"{k}_dupes_same_file"] = int(same.sum())
                ex = [(d["files"][ff[j]], d["files"][ff[j + 1]]) for j in np.where((aa[1:] == aa[:-1]) & (ff[1:] != ff[:-1]))[0][:3]]
                s[f"{k}_examples_across_files"] = ex
            allk[k].append((fam, u))
    summary["families"][fam] = s
    print(fam, s, flush=True)
# cross-family overlaps (single sort per key type)
cross = {}
for k in ("kj", "ku", "km"):
    L = allk[k]
    if not L: continue
    a = np.concatenate([u for _, u in L]); fi = np.concatenate([np.full(len(u), i, np.int32) for i, (_, u) in enumerate(L)])
    o = np.argsort(a, kind="stable"); a, fi = a[o], fi[o]
    eq = a[1:] == a[:-1]
    # group runs of equal keys; count every family pair inside a run (runs are short)
    idx = np.where(eq)[0]
    # run starts
    starts = idx[np.r_[True, idx[1:] != idx[:-1] + 1]] if len(idx) else idx
    ends = idx[np.r_[idx[1:] != idx[:-1] + 1, True]] + 1 if len(idx) else idx
    from collections import Counter
    cnt = Counter()
    ln = ends - starts + 1
    # fast path: runs of length 2
    two = ln == 2
    p = np.stack([fi[starts[two]], fi[starts[two] + 1]], 1)
    if len(p):
        up, c = np.unique(p, axis=0, return_counts=True)
        for (x, y), n in zip(up, c): cnt[(int(x), int(y))] += int(n)
    for s0, e0 in zip(starts[~two], ends[~two]):
        fs = sorted(set(fi[s0:e0 + 1].tolist()))
        for ii in range(len(fs)):
            for jj in range(ii + 1, len(fs)): cnt[(fs[ii], fs[jj])] += 1
    for (x, y), n in cnt.items():
        cross[f"{k}|{L[x][0]}|{L[y][0]}"] = [n, len(L[x][1]), len(L[y][1])]
    print(k, "cross done", len(a), flush=True)
summary["cross"] = cross
json.dump(summary, open(f"{out}/dupes_summary.json", "w"), indent=1)
print("cross-family overlaps:"); [print(" ", k, v) for k, v in sorted(cross.items())]
