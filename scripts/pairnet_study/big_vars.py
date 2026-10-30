"""Per-event pairings and CMS-style trijet variables over many files, in parallel (one worker per file).

Pairings among the leading 7 jets:
  am    argmin |m1 - m2|
  dm    argmax Delta_min, Delta_t = (scalar pT sum of triplet t) - m_t, Delta_min = min over the two triplets
  dma   argmax Delta_min among interpretations with A < 0.1 (argmin |m1 - m2| if none)
  pn    PairNet (ONNX r8B; files listed in --pn only): argmax of the gated logits, plus P(signal)
Per pairing: m1, m2, Delta and D2_[3,2] of each triplet, D2_[(6,3)+(3,2)] of the six jets, lowest pT of the six.
Definitions as cms_vars.py (CMS-EXO-17-030). Output: one npz with all events in sorted file order (signal files first).
"""
import glob, os, re, sys
from itertools import combinations
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, "/scratch/llee/CombinatorialSolver")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ONNX = os.environ.get("PAIRNET_ONNX", "/scratch/llee/pairnet/deliverables/pairnet_gated_12jet_win5_f1d279f.onnx")
T3 = np.array(list(combinations(range(6), 3)))
PR = [(0, 1), (0, 2), (1, 2)]
mm = lambda v: np.sqrt(np.maximum(v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1), 0))


def d2_32(t4):
    den = mm(t4.sum(-2)) ** 2 + (mm(t4) ** 2).sum(-1)
    mh = np.stack([np.sqrt(mm(t4[..., i, :] + t4[..., j, :]) ** 2 / np.maximum(den, 1e-12)) for i, j in PR], -1)
    return ((mh - 1 / np.sqrt(3)) ** 2).sum(-1)


def variables(p4, pt, G, best):
    ar = np.arange(len(best))
    j6 = G[best]
    v = p4[ar[:, None], j6]
    ptc = pt[ar[:, None], j6]
    m1, m2 = mm(v[:, :3].sum(1)), mm(v[:, 3:].sum(1))
    den63 = 4 * mm(v.sum(1)) ** 2 + 6 * (mm(v) ** 2).sum(1)
    tr = v[:, T3]
    d63 = ((np.sqrt(mm(tr.sum(2)) ** 2 / np.maximum(den63, 1e-12)[:, None] + d2_32(tr)) - 1 / np.sqrt(20)) ** 2).sum(1)
    return {"m1": m1, "m2": m2, "delta1": ptc[:, :3].sum(1) - m1, "delta2": ptc[:, 3:].sum(1) - m2,
            "d32_1": d2_32(v[:, :3]), "d32_2": d2_32(v[:, 3:]), "d2_63_32": d63, "pt_low6": ptc.min(1)}


def work(job):
    fp, is_sig, do_pn = job
    import torch
    from src.pairdata import load_events
    from src.pairnet import to_cartesian
    from src.combinatorics import enumerate_assignments
    torch.set_num_threads(1)
    d = load_events([fp], 12, 3, is_sig, None, min_jets=6, truth="none", n_assign=7)
    n = len(d["pt"])
    p4_12 = (to_cartesian(*(torch.from_numpy(d[k]).double() for k in ("pt", "eta", "phi", "m"))) * torch.from_numpy(d["mask"]).unsqueeze(-1)).numpy()
    p4, pt, mask = p4_12[:, :7], d["pt"][:, :7], d["mask"][:, :7]
    asg = enumerate_assignments(7)
    G = np.array([list(a[1]) + list(a[2]) for a in asg])
    m1 = mm(p4[:, G[:, :3]].sum(2)); m2 = mm(p4[:, G[:, 3:]].sum(2))
    ok = mask[:, G].all(-1)
    A = np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)
    dmin = np.minimum(pt[:, G[:, :3]].sum(2) - m1, pt[:, G[:, 3:]].sum(2) - m2)
    picks = {"am": np.where(ok, np.abs(m1 - m2), np.inf).argmin(1), "dm": np.where(ok, dmin, -np.inf).argmax(1)}
    okA = ok & (A < 0.1)
    picks["dma"] = np.where(okA.any(1), np.where(okA, dmin, -np.inf).argmax(1), picks["am"])
    res = {"pt6_lead": pt[:, 5].astype(np.float32), "pmass": d["pmass"], "w": d["w"].astype(np.float32),
           "is_signal": np.full(n, is_sig), "fileclass": np.full(n, int(re.search(r"_c(\d+)\.h5$", fp).group(1)) % 10, np.int8)}
    if do_pn:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
        lg, ps = [], []
        for i in range(0, n, 2048):
            o = sess.run(["logits", "p_signal"], {"four_momenta": p4_12[i:i + 2048].astype(np.float32)})
            lg.append(o[0].argmax(1)); ps.append(o[1])
        picks["pn"] = np.concatenate(lg); res["psig"] = np.concatenate(ps).astype(np.float32)
    for name, b in picks.items():
        for k, x in variables(p4, pt, G, b).items():
            res[f"{name}_{k}"] = x.astype(np.float32)
        res[f"{name}_valid"] = ok[np.arange(n), b]
    return fp, res


def run_one(job):
    """Worker entry: write one npz per input file (skipped if present), so a crash loses only that file."""
    fp = job[0]
    dst = os.path.join(PARTS, os.path.basename(fp).replace(".h5", ".npz"))
    if os.path.exists(dst):
        return fp, "cached"
    try:
        _, res = work(job)
    except Exception as ex:  # noqa: BLE001
        return fp, f"error {ex!r}"
    np.savez(dst[:-4] + ".tmp.npz", **res)
    os.replace(dst[:-4] + ".tmp.npz", dst)
    return fp, "ok"


PARTS = os.environ.get("BIGVARS_PARTS", "/scratch/llee/pairnet/cuts_study/parts")

if __name__ == "__main__":
    import multiprocessing as mp
    out = sys.argv[1]
    os.makedirs(PARTS, exist_ok=True)
    P = "/scratch/llee/pairnet/data_r3/prod"
    sig = sorted(glob.glob(f"{P}/sig_*_c*9.h5"))
    qcd = sorted(glob.glob(f"{P}/qcd_c*.h5"))
    jobs = [(f, True, True) for f in sig] + [(f, False, f[-4] in "89") for f in qcd]
    if os.environ.get("BIGVARS_PN_ONLY"):  # model evaluation: only files PairNet is scored on
        jobs = [j for j in jobs if j[2]]
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 16, maxtasksperchild=20) as pool:
        bad = [(f, st) for f, st in pool.imap_unordered(run_one, jobs, chunksize=1) if st not in ("ok", "cached")]
    for f, st in bad:
        print("FAILED", f, st)
    parts = {f: dict(np.load(os.path.join(PARTS, os.path.basename(f).replace(".h5", ".npz")))) for f, _, _ in jobs}
    keys = set().union(*(p.keys() for p in parts.values()))
    res = {}
    for k in sorted(keys):
        cols = []
        for f, _, _ in jobs:
            p = parts[f]
            if k in p:
                cols.append(p[k])
            else:  # PairNet columns are absent for the QCD training files
                n = len(p["w"])
                cols.append(np.full(n, False) if k.endswith("_valid") else np.full(n, np.nan, np.float32))
        res[k] = np.concatenate(cols)
    res["has_pn"] = np.concatenate([np.full(len(parts[f]["w"]), pn) for f, _, pn in jobs])
    np.savez(out, **res)
    print("events", len(res["w"]), "signal", int(res["is_signal"].sum()), "QCD", int((~res["is_signal"]).sum()),
          "QCD with PairNet", int((res["has_pn"] & ~res["is_signal"]).sum()))
