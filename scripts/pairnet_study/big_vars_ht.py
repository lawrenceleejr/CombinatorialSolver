"""Score the high-HT QCD sample (prod_ht/qcdht_c*.h5: pTHat > 200, bias power 4, HT > 1200 GeV) like big_vars.py:
classical pairings + PairNet (ONNX), CMS-style variables, plus HT and the per-file normalisation. One npz per file."""
import glob, os, sys
import multiprocessing as mp
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import big_vars  # noqa: E402

PARTS = os.environ.get("BIGVARS_HT_PARTS", "/scratch/llee/pairnet/cuts_study/parts_ht")


def run_one(fp):
    dst = os.path.join(PARTS, os.path.basename(fp).replace(".h5", ".npz"))
    if os.path.exists(dst):
        return fp, "cached"
    try:
        with h5py.File(fp, "r") as h:
            pt = h["INPUTS/Source/pt"][:]; m = h["INPUTS/Source/MASK"][:].astype(bool)
            fac = h.attrs["sigma_mb"] / h.attrs["weight_sum"] * 1e12
        _, res = big_vars.work((fp, False, True))
        res["ht"] = (pt * m).sum(1)[m.sum(1) >= 6].astype(np.float32)
        assert len(res["ht"]) == len(res["w"])
        res["fac"] = np.full(len(res["w"]), fac, np.float64)
    except Exception as ex:  # noqa: BLE001
        return fp, f"error {ex!r}"
    np.savez(dst[:-4] + ".tmp.npz", **res)
    os.replace(dst[:-4] + ".tmp.npz", dst)
    return fp, "ok"


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(PARTS, exist_ok=True)
    files = sorted(f for f in glob.glob("/scratch/llee/pairnet/prod_ht/qcdht_c*.h5") if os.path.getsize(f) > 0)
    lo = int(os.environ.get("HT_MIN_INDEX", "0"))  # e.g. 1400: files not used to train r13A
    files = [f for f in files if int(f.rsplit("_c", 1)[1][:-3]) >= lo]
    with mp.get_context("spawn").Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 16, maxtasksperchild=20) as pool:
        bad = [(f, st) for f, st in pool.imap_unordered(run_one, files, chunksize=1) if st not in ("ok", "cached")]
    for f, st in bad:
        print("FAILED", f, st)
    good = [f for f in files if os.path.exists(os.path.join(PARTS, os.path.basename(f).replace(".h5", ".npz")))]
    parts = [dict(np.load(os.path.join(PARTS, os.path.basename(f).replace(".h5", ".npz")))) for f in good]
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    res["n_files"] = np.array(len(good))
    np.savez(out, **res)
    print("files", len(good), "events", len(res["w"]), "HT min %.0f" % res["ht"].min())
