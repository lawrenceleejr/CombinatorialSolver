"""F3 audit library: event loading, perturbations, r14A ONNX scoring, search variables (definitions as scripts/score_lowmass.py)."""
import json, os
from itertools import combinations
try:
    import h5py
except ImportError:
    h5py = None
import numpy as np
import onnxruntime as ort

HOME = os.path.expanduser("~")
ONNX = os.environ.get("F3_ONNX", f"{HOME}/CombinatorialSolver/models/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx")
G = np.array([list(x[1]) + list(x[2]) for x in json.load(open(ONNX[:-5] + ".json"))["assignments"]])
ISR = np.array([x[0] for x in json.load(open(ONNX[:-5] + ".json"))["assignments"]])
T3 = np.array(list(combinations(range(6), 3)))
PR = [(0, 1), (0, 2), (1, 2)]
NSLOT = 24  # stored jets (20) + room for added pile-up jets


def session(threads=4, path=ONNX):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def mm(v):
    return np.sqrt(np.maximum(v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1), 0))


def d2_32(t4):
    den = mm(t4.sum(-2)) ** 2 + (mm(t4) ** 2).sum(-1)
    mh = np.stack([np.sqrt(mm(t4[..., i, :] + t4[..., j, :]) ** 2 / np.maximum(den, 1e-12)) for i, j in PR], -1)
    return ((mh - 1 / np.sqrt(3)) ** 2).sum(-1)


def variables(p4, pt, best):
    ar = np.arange(len(best))
    j6 = G[best]
    v = p4[ar[:, None], j6]
    ptc = pt[ar[:, None], j6]
    m1, m2 = mm(v[:, :3].sum(1)), mm(v[:, 3:].sum(1))
    den63 = 4 * mm(v.sum(1)) ** 2 + 6 * (mm(v) ** 2).sum(1)
    d63 = ((np.sqrt(mm(v[:, T3].sum(2)) ** 2 / np.maximum(den63, 1e-12)[:, None] + d2_32(v[:, T3])) - 1 / np.sqrt(20)) ** 2).sum(1)
    return {"m1": m1, "m2": m2, "delta1": ptc[:, :3].sum(1) - m1, "delta2": ptc[:, 3:].sum(1) - m2,
            "d32_1": d2_32(v[:, :3]), "d32_2": d2_32(v[:, 3:]), "d2_63_32": d63, "pt_low6": ptc.min(1)}


def load_h5(f, rows=None):
    with h5py.File(f, "r") as h:
        s = h["INPUTS/Source"]
        pt, eta, phi, m = (s[x][:].astype(np.float64) for x in ("pt", "eta", "phi", "mass"))
        mask = s["MASK"][:].astype(bool)
        w = h["EventVars/normweight"][:].astype(np.float64)
        at = {k: (v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items()}
    if rows is not None:
        pt, eta, phi, m, mask, w = (x[rows] for x in (pt, eta, phi, m, mask, w))
    return pt, eta, phi, m, mask, w, at


def sortpad(pt, eta, phi, m, mask, n=12):
    """pT-sort (stable), keep the leading n, zero-pad. Returns sorted full arrays and the (N, n, 4) float64 four-momenta."""
    order = np.argsort(np.where(mask, -pt, np.inf), axis=1, kind="stable")
    pt, eta, phi, m, mask = (np.take_along_axis(x, order, 1) for x in (pt, eta, phi, m, mask))
    pt = pt * mask; m = m * mask
    pad = max(0, n - pt.shape[1])
    a = [np.pad(x[:, :n], ((0, 0), (0, pad))) for x in (pt, eta, phi, m, mask)]
    px, py, pz = a[0] * np.cos(a[2]), a[0] * np.sin(a[2]), a[0] * np.sinh(a[1])
    p4 = np.stack([np.sqrt(px ** 2 + py ** 2 + pz ** 2 + a[3] ** 2), px, py, pz], -1) * a[4][..., None]
    return pt, mask, p4


def run(sess, p4, bs=2048, outs=("logits", "p_signal", "m_est")):
    res = [[] for _ in outs]
    for i in range(0, len(p4), bs):
        o = sess.run(list(outs), {"four_momenta": p4[i:i + bs].astype(np.float32)})
        for r, x in zip(res, o):
            r.append(x)
    return [np.concatenate(r) if r else np.zeros(0) for r in res]


def evaluate(sess, pt, eta, phi, m, mask):
    """Full search variables for (N, njet) raw arrays. Events with < 6 jets get valid = False."""
    pts, masks, p4 = sortpad(pt, eta, phi, m, mask)
    lg, ps, me = run(sess, p4)
    best = lg.argmax(1)
    p47, pt7, mask7 = p4[:, :7], pts[:, :7], masks[:, :7]
    v = variables(p47, pt7, best)
    ar = np.arange(len(best))
    valid = mask7[ar[:, None], G[best]].all(1)
    lgs = np.sort(lg, 1)
    out = {"psig": ps.astype(np.float64), "mest": me.astype(np.float64), "best": best.astype(np.int8), "valid": valid,
           "ht": (pts * masks).sum(1), "nj": masks.sum(1).astype(np.int8), "gap": (lgs[:, -1] - lgs[:, -2]).astype(np.float32),
           "mavg": 0.5 * (v["m1"] + v["m2"]), "asym": np.abs(v["m1"] - v["m2"]) / np.maximum(v["m1"] + v["m2"], 1e-9),
           "dmax": np.maximum(v["delta1"], v["delta2"]), "dmin": np.minimum(v["delta1"], v["delta2"]),
           "d32": np.maximum(v["d32_1"], v["d32_2"]), "d63": v["d2_63_32"], "ptlow": v["pt_low6"]}
    return {k: (x.astype(np.float32) if x.dtype == np.float64 and k != "psig" else x) for k, x in out.items()}


# ------------------------------------------------------------------ perturbations
def widen(pt, eta, phi, m, mask):
    pad = NSLOT - pt.shape[1]
    return [np.pad(x, ((0, 0), (0, pad))) for x in (pt, eta, phi, m, mask)]


def perturb(name, pt, eta, phi, m, mask, seed=1):
    """Return perturbed copies. name: 'nom' or '+'-joined atoms; a trailing '@T' applies a jet pT threshold T at the end."""
    thr = None
    if "@" in name:
        name, thr = name.split("@"); thr = float(thr)
    pt, eta, phi, m, mask = (x.copy() for x in (pt, eta, phi, m, mask))
    for atom in name.split("+"):
        import zlib
        rng = np.random.default_rng([seed, zlib.crc32(atom.encode())])
        k, _, x = atom.partition(":"); x = float(x) if x else None
        if k == "nom":
            pass
        elif k == "m0":
            m[:] = 0
        elif k == "msm":
            m *= np.maximum(1 + x * rng.standard_normal(m.shape), 0)
        elif k == "msc":
            m *= x
        elif k == "moff":
            m = (m + x) * mask
        elif k == "mq":  # m^2 -> m^2 + (x pT R)^2, R = 0.4
            m = np.sqrt(m ** 2 + (x * pt * 0.4) ** 2)
        elif k == "jes":
            pt *= x; m *= x
        elif k == "jer":
            f = np.maximum(1 + x * rng.standard_normal(m.shape), 0.05); pt *= f; m *= f
        elif k == "thr":
            mask &= pt >= x
        elif k == "eta":
            mask &= np.abs(eta) < x
        elif k == "drop":
            mask &= rng.random(mask.shape) >= x
        elif k == "pu":  # add int(x) jets, pT 30-60, |eta| < 3 flat, random phi, m/pT = 0.15
            n = int(x)
            for _ in range(n):
                slot = mask.sum(1)  # first free slot after compaction below
                order = np.argsort(~mask, axis=1, kind="stable")
                pt, eta, phi, m, mask = (np.take_along_axis(a, order, 1) for a in (pt, eta, phi, m, mask))
                ok = slot < mask.shape[1]; ar = np.where(ok)[0]; s = slot[ok]
                pp = rng.uniform(30, 60, len(mask))
                pt[ar, s] = pp[ok]; eta[ar, s] = rng.uniform(-3, 3, len(mask))[ok]
                phi[ar, s] = rng.uniform(-np.pi, np.pi, len(mask))[ok]; m[ar, s] = 0.15 * pp[ok]; mask[ar, s] = True
        elif k == "ang":
            eta += x * rng.standard_normal(eta.shape); phi += x * rng.standard_normal(phi.shape)
        else:
            raise ValueError(atom)
    if thr is not None:
        mask &= pt >= thr
    return pt * mask, eta * mask, phi * mask, m * mask, mask
