"""Which further variables separate signal from the QCD that survives the 700 GeV selection?

Events: r14A pairing, A < 0.1, m_avg in [0.92, 1.04] M, P(signal) >= PMIN, offline H_T > 450 GeV. Signal: held-out
700 GeV h5 (c16-19, inclusive + boosted); QCD: the targeted tail h5 files (in-job r14A filter, H_T > 900 GeV).
For each candidate variable: weighted QCD survival at 90% / 70% signal efficiency for the better cut direction.
usage: featstudy.py onnx sig.h5[,..] qcd_glob nqcd [PMIN]"""
import glob, json, sys
import h5py
import numpy as np
import onnxruntime as ort

onnx, sigs, qglob, nq = sys.argv[1], sys.argv[2].split(","), sys.argv[3], int(sys.argv[4]); PMIN = float(sys.argv[5]) if len(sys.argv) > 5 else 0.99
M, LO, HI = 700.0, 0.92, 1.04
HTMIN = float(sys.argv[6]) if len(sys.argv) > 6 else 450.0
sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
G = np.array([list(x[1]) + list(x[2]) for x in json.load(open(onnx[:-5] + ".json"))["assignments"]])


def p4(pt, eta, phi, m):
    return np.stack([np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], -1)


def load(path):
    f = h5py.File(path, "r"); S = f["INPUTS/Source"]
    pt, eta, phi, m, mask = (S[k][:] for k in ("pt", "eta", "phi", "mass", "MASK"))
    w = f["EventVars/normweight"][:] * float(f.attrs["sigma_mb"]) / float(f.attrs["weight_sum"])
    mask = mask.astype(bool); keep = mask.sum(1) >= 6
    pt, eta, phi, m, mask, w = (x[keep] for x in (pt, eta, phi, m, mask, w))
    o = np.argsort(np.where(mask, -pt, np.inf), axis=1, kind="stable")
    pt, eta, phi, m, mask = (np.take_along_axis(x, o, 1) for x in (pt, eta, phi, m, mask))
    pad = lambda x: np.pad(x[:, :12], ((0, 0), (0, max(0, 12 - x.shape[1]))))
    P = p4(pad(pt), pad(eta), pad(phi), pad(m)) * pad(mask)[..., None]
    if len(P) == 0:
        return None
    lg, ps, _, ms, _ = sess.run(None, {"four_momenta": P.astype(np.float32)})
    b = lg.argmax(1); g = G[b]; n = np.arange(len(b))
    m1, m2 = ms[n, b, 0], ms[n, b, 1]; ma = 0.5 * (m1 + m2); A = np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)
    valid = mask[n[:, None], g].all(1)
    ht = (pt * mask).sum(1)
    sel = valid & (A < 0.1) & (ma > LO * M) & (ma < HI * M) & (ps >= PMIN) & (ht > HTMIN)
    if not sel.any():
        return None
    P7 = P[:, :7]; t1 = P7[n[:, None], g[:, :3]].sum(1); t2 = P7[n[:, None], g[:, 3:]].sum(1)
    def kin(v):
        pt_ = np.hypot(v[:, 1], v[:, 2]); y = 0.5 * np.log(np.maximum(v[:, 0] + v[:, 3], 1e-9) / np.maximum(v[:, 0] - v[:, 3], 1e-9))
        return pt_, y, np.arctan2(v[:, 2], v[:, 1])
    pt1, y1, f1 = kin(t1); pt2, y2, f2 = kin(t2)
    jm = np.take_along_axis(m[:, :12], np.zeros((len(m), 0), int), 1) if False else None
    six = np.take_along_axis(np.pad(m, ((0, 0), (0, max(0, 12 - m.shape[1]))))[:, :7], g, 1) / np.maximum(np.take_along_axis(np.pad(pt, ((0, 0), (0, max(0, 12 - pt.shape[1]))))[:, :7], g, 1), 1e-9)
    used = np.zeros((len(b), pt.shape[1]), bool); used[n[:, None], g] = True
    extra_pt = np.where(mask & ~used, pt, 0)
    v = {"|dy| triplets": np.abs(y1 - y2), "cos theta* = |tanh(dy/2)|": np.abs(np.tanh(0.5 * (y1 - y2))),
         "pT balance |pT1-pT2|/sum": np.abs(pt1 - pt2) / np.maximum(pt1 + pt2, 1e-9),
         "dphi triplets": np.abs(np.angle(np.exp(1j * (f1 - f2)))),
         "n jets": mask.sum(1).astype(float), "sum pT of unused jets / M": extra_pt.sum(1) / M,
         "leading unused jet pT / M": extra_pt.max(1) / M,
         "mean jet m/pT (6 chosen)": six.mean(1), "max jet m/pT (6 chosen)": six.max(1),
         "pair pT / M": np.hypot(t1[:, 1] + t2[:, 1], t1[:, 2] + t2[:, 2]) / M, "H_T / M": ht / M,
         "1 - P": 1 - ps}
    return {k: x[sel] for k, x in v.items()}, w[sel]


def stack(paths):
    parts = [r for r in (load(p) for p in paths) if r is not None]
    return {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}, np.concatenate([p[1] for p in parts])


sv, sw = stack(sigs)
qfiles = sorted(glob.glob(qglob))[:nq]
qv, qw = stack(qfiles); qw = qw / len(qfiles)
sw = np.ones_like(sw)  # signal: count events
print(f"signal events {len(sw)}, QCD MC events {len(qw)} (neff {qw.sum() ** 2 / (qw ** 2).sum():.0f}) after P >= {PMIN}, window [{LO}, {HI}] M")
for k in sv:
    out = []
    for eff in (0.9, 0.7):
        best = None
        for d in (1, -1):  # keep values below (d=1) or above (d=-1) a threshold
            t = np.quantile(d * sv[k], eff)
            surv = (qw[d * qv[k] <= t]).sum() / qw.sum()
            if best is None or surv < best[0]:
                best = (surv, d, t * d)
        out.append(f"sig {eff:.0%}: QCD {best[0]:.2f} ({'<' if best[1] == 1 else '>'} {best[2]:.3g})")
    print(f"{k:32s} " + "   ".join(out))
