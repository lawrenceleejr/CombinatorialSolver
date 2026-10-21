"""
Score a generated h5 file with one or more PairNet ONNX models and the argmin |m1 - m2| rule, and write a compact npz
with the per-event variables of a low-mass search. Self-contained (numpy, h5py, onnxruntime) so it runs on grid workers.

  python scripts/score_lowmass.py in.h5 out.npz name1=model1.onnx [name2=model2.onnx ...] [--window LO,HI]

Per pairing p (each model name and "am"): p_m1, p_m2, p_delta1, p_delta2 (scalar pT sum minus mass of each triplet),
p_d32_1, p_d32_2 (triplet Dalitz distances), p_d2_63_32 (six-jet Dalitz distance), p_pt_low6 (lowest jet pT of the six),
p_valid; per model: <name>_psig; per event: ht, pt6_lead, w (normweight), pmass. Definitions as CMS-EXO-17-030.
With --window, only events where some pairing has A < 0.1 and LO < m_avg < HI are kept (n_all records the total).
File attributes are copied (sigma_mb, weight_sum, n_tried, mass, process).
"""
import argparse
import json
from itertools import combinations

import h5py
import numpy as np
import onnxruntime as ort

T3 = np.array(list(combinations(range(6), 3)))
PR = [(0, 1), (0, 2), (1, 2)]


def mm(v):
    return np.sqrt(np.maximum(v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1), 0))


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
    d63 = ((np.sqrt(mm(v[:, T3].sum(2)) ** 2 / np.maximum(den63, 1e-12)[:, None] + d2_32(v[:, T3])) - 1 / np.sqrt(20)) ** 2).sum(1)
    return {"m1": m1, "m2": m2, "delta1": ptc[:, :3].sum(1) - m1, "delta2": ptc[:, 3:].sum(1) - m2,
            "d32_1": d2_32(v[:, :3]), "d32_2": d2_32(v[:, 3:]), "d2_63_32": d63, "pt_low6": ptc.min(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("models", nargs="+")
    ap.add_argument("--window", default=None)
    a = ap.parse_args()
    with h5py.File(a.inp, "r") as f:
        src = f["INPUTS/Source"]
        pt, eta, phi, m = (src[x][:] for x in ("pt", "eta", "phi", "mass"))
        mask = src["MASK"][:].astype(bool)
        w = f["EventVars/normweight"][:]
        pmass = f["EventVars/parent_mass"][:] if "EventVars/parent_mass" in f else np.zeros(len(pt), np.float32)
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    order = np.argsort(np.where(mask, -pt, np.inf), axis=1, kind="stable")
    take = lambda x: np.take_along_axis(x, order, 1)
    pt, eta, phi, m, mask = map(take, (pt, eta, phi, m, mask))
    keep6 = mask.sum(1) >= 6
    pt, eta, phi, m, mask, w, pmass = (x[keep6] for x in (pt, eta, phi, m, mask, w, pmass))
    n = len(pt)
    pad = max(0, 12 - pt.shape[1])
    pt12, eta12, phi12, m12, mask12 = (np.pad(x[:, :12], ((0, 0), (0, pad))) for x in (pt, eta, phi, m, mask))
    px, py, pz = pt12 * np.cos(phi12), pt12 * np.sin(phi12), pt12 * np.sinh(eta12)
    p4_12 = np.stack([np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m12 ** 2), px, py, pz], -1) * mask12[..., None]
    p4, pt7, mask7 = p4_12[:, :7].astype(np.float64), pt12[:, :7], mask12[:, :7]
    res = {"ht": (pt * mask).sum(1).astype(np.float32), "pt6_lead": pt7[:, 5].astype(np.float32), "w": w, "pmass": pmass}
    picks, G = {}, None
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    for spec in a.models:
        name, path = spec.split("=", 1)
        if G is None:
            asg = json.load(open(path[:-5] + ".json"))["assignments"]
            G = np.array([list(x[1]) + list(x[2]) for x in asg])
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        lg, ps = [], []
        for i in range(0, n, 2048):
            o = sess.run(["logits", "p_signal"], {"four_momenta": p4_12[i:i + 2048].astype(np.float32)})
            lg.append(o[0].argmax(1)); ps.append(o[1])
        picks[name] = np.concatenate(lg) if lg else np.zeros(0, int)
        res[f"{name}_psig"] = np.concatenate(ps).astype(np.float32) if ps else np.zeros(0, np.float32)
    m1 = mm(p4[:, G[:, :3]].sum(2)); m2 = mm(p4[:, G[:, 3:]].sum(2))
    ok = mask7[:, G].all(-1)
    picks["am"] = np.where(ok, np.abs(m1 - m2), np.inf).argmin(1) if n else np.zeros(0, int)
    sel = np.zeros(n, bool)
    for name, b in picks.items():
        for k, x in variables(p4, pt7, G, b).items():
            res[f"{name}_{k}"] = x.astype(np.float32)
        res[f"{name}_valid"] = ok[np.arange(n), b] if n else np.zeros(0, bool)
        if a.window:
            lo, hi = (float(x) for x in a.window.split(","))
            ma = 0.5 * (res[f"{name}_m1"] + res[f"{name}_m2"]); A = np.abs(res[f"{name}_m1"] - res[f"{name}_m2"]) / np.maximum(2 * ma, 1e-9)
            sel |= (A < 0.1) & (ma > lo) & (ma < hi)
    if a.window:
        res = {k: x[sel] for k, x in res.items()}
    res["n_all"] = np.array(n)
    for k in ("sigma_mb", "weight_sum", "n_tried", "mass"):
        res[f"attr_{k}"] = np.array(float(attrs.get(k, np.nan)))
    np.savez_compressed(a.out, **res)
    print(f"scored {n} events, kept {len(res['w'])}, models {list(picks)}")


if __name__ == "__main__":
    main()
