"""F6: re-score h5 files with the r14A ONNX keeping the interpretation-level output, the mass head and extra features.

usage: rescore.py <kind: qcd|sig> <out.npz> <rich_pmin> <h5 files or a directory>
Per-event scalars are kept for events with >= 6 jets (as score_lowmass.py) and, for QCD, H_T > 400 GeV and either the
argmax interpretation valid with A < 0.1 or P(signal) >= rich_pmin with at least one symmetric interpretation; signal
keeps every event. For "rich" events (P >= rich_pmin; all signal events) the symmetric interpretations (valid, A < 0.1)
are stored, sorted by decreasing gated logit and padded to K = 24 (S_* arrays, row = rich_idx).
Weights: w = normweight; fb_per_w = sigma_mb * 1e12 / weight_sum of the file; fid = file number in the sample
(c index); file_* arrays describe every readable file (also files with no kept event).
"""
import glob, json, os, re, sys, time
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_lowmass_copy import variables, mm  # noqa: E402  (identical definitions to the scored npz files)

ONNX = f"{HERE}/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx"
ASG = json.load(open(ONNX[:-5] + ".json"))["assignments"]
G = np.array([list(x[1]) + list(x[2]) for x in ASG]); ISR = np.array([x[0] for x in ASG])
K = 24
_sess = None


def sess():
    global _sess
    if _sess is None:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        _sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    return _sess


def rap(v):
    return 0.5 * np.log(np.maximum(v[..., 0] + v[..., 3], 1e-9) / np.maximum(v[..., 0] - v[..., 3], 1e-9))


def one(f, kind, rich_pmin):
    with h5py.File(f, "r") as h:
        src = h["INPUTS/Source"]
        pt, eta, phi, m = (src[x][:] for x in ("pt", "eta", "phi", "mass"))
        mask = src["MASK"][:].astype(bool)
        w = h["EventVars/normweight"][:]
        at = {k: float(h.attrs[k]) for k in ("sigma_mb", "weight_sum", "n_tried", "mass")}
    cidx = int(re.search(r"_c(\d+)", os.path.basename(f)).group(1))
    order = np.argsort(np.where(mask, -pt, np.inf), axis=1, kind="stable")
    take = lambda x: np.take_along_axis(x, order, 1)
    pt, eta, phi, m, mask = map(take, (pt, eta, phi, m, mask))
    keep6 = mask.sum(1) >= 6
    pt, eta, phi, m, mask, w = (x[keep6] for x in (pt, eta, phi, m, mask, w))
    n = len(pt)
    meta = {"fid": cidx, "n_tried": at["n_tried"], "fb_per_w": at["sigma_mb"] * 1e12 / at["weight_sum"], "n": n, "mass": at["mass"]}
    if n == 0:
        return meta, None, None
    pad = max(0, 12 - pt.shape[1])
    pt12, eta12, phi12, m12, mask12 = (np.pad(x[:, :12], ((0, 0), (0, pad))) for x in (pt, eta, phi, m, mask))
    px, py, pz = pt12 * np.cos(phi12), pt12 * np.sin(phi12), pt12 * np.sinh(eta12)
    p4_12 = np.stack([np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m12 ** 2), px, py, pz], -1) * mask12[..., None]
    s = sess(); names = [o.name for o in s.get_outputs()]
    outs = {k: [] for k in names}
    for i in range(0, n, 256):
        o = s.run(None, {"four_momenta": p4_12[i:i + 256].astype(np.float32)})
        for k, x in zip(names, o):
            outs[k].append(x)
    outs = {k: np.concatenate(x) for k, x in outs.items()}
    lg, sl, ps, mest = outs["logits"], outs["signal_logits"], outs["p_signal"].reshape(n), outs["m_est"].reshape(n)
    ht = (pt * mask).sum(1)
    p4, pt7, mask7 = p4_12[:, :7].astype(np.float64), pt12[:, :7], mask12[:, :7]
    ok = mask7[:, G].all(-1)
    v3a, v3b = p4[:, G[:, :3]].sum(2), p4[:, G[:, 3:]].sum(2)  # (n, 70, 4) triplet four-vectors
    m1, m2 = mm(v3a), mm(v3b)
    ma = 0.5 * (m1 + m2); A = np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)
    sym = ok & (A < 0.1)
    best = lg.argmax(1); ar = np.arange(n)
    if kind == "qcd":
        keep = (ht > 400) & (sym[ar, best] | ((ps >= rich_pmin) & sym.any(1)))
    else:
        keep = np.ones(n, bool)
    meta["n_kept"] = int(keep.sum())
    if not keep.any():
        return meta, None, None
    (pt, eta, phi, m, mask, w, pt12, eta12, m12, mask12, lg, sl, ps, mest, ht, p4, pt7, mask7, ok, v3a, v3b, m1, m2, ma, A, sym, best) = (
        x[keep] for x in (pt, eta, phi, m, mask, w, pt12, eta12, m12, mask12, lg, sl, ps, mest, ht, p4, pt7, mask7, ok, v3a, v3b, m1, m2, ma, A, sym, best))
    n = len(pt); ar = np.arange(n)
    best_sig = np.where(ok, sl, -np.inf).argmax(1)
    pts1, pts2 = pt7[:, G[:, :3]].sum(2), pt7[:, G[:, 3:]].sum(2)
    am = np.where(ok, np.abs(m1 - m2), np.inf).argmin(1)
    E = {"ht": ht, "w": w, "psig": ps, "m_est": mest, "best": best.astype(np.int16), "best_sig": best_sig.astype(np.int16),
         "am": am.astype(np.int16), "n_jets": mask.sum(1).astype(np.int16), "pt6_lead": pt7[:, 5],
         "fid": np.full(n, cidx, np.int32), "fb_per_w": np.full(n, meta["fb_per_w"]), "pmass": np.full(n, at["mass"], np.float32)}
    for name, b in (("pn", best), ("am", am)):
        for k, x in variables(p4, pt7, G, b).items():
            E[f"{name}_{k}"] = x
        E[f"{name}_valid"] = ok[ar, b]
    ta, tb = v3a[ar, best], v3b[ar, best]
    six = ta + tb
    E["dy"] = np.abs(rap(ta) - rap(tb)); E["y6"] = np.abs(rap(six))
    pta, ptb = np.hypot(ta[:, 1], ta[:, 2]), np.hypot(tb[:, 1], tb[:, 2])
    E["tpt_max"] = np.maximum(pta, ptb); E["tpt_min"] = np.minimum(pta, ptb)
    E["dphi"] = np.abs(np.angle(np.exp(1j * (np.arctan2(ta[:, 2], ta[:, 1]) - np.arctan2(tb[:, 2], tb[:, 1])))))
    E["m6"] = mm(six); E["pt6sys"] = np.hypot(six[:, 1], six[:, 2])
    j6 = G[best]
    E["pt_unused"] = ht - pt7[ar[:, None], j6].sum(1)  # scalar pT sum of all jets outside the six
    isr = ISR[best]
    E["pt_isr"] = pt7[ar, isr] * mask7[ar, isr]
    E["pt8"] = pt12[:, 7] * mask12[:, 7]
    E["eta_max6"] = np.abs(eta12[ar[:, None], j6]).max(1)
    E["mj_sum6"] = m12[ar[:, None], j6].sum(1)
    E["pt1"] = pt7[:, 0]; E["pt2"] = pt7[:, 1]
    lgm = np.where(ok, lg, -np.inf)
    lp = lgm - lgm.max(1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(1, keepdims=True)); pr = np.exp(lp)
    E["ent"] = -(pr * np.where(pr > 0, lp, 0)).sum(1)
    srt = np.sort(lgm, 1)
    E["margin"] = srt[:, -1] - srt[:, -2]; E["pbest"] = pr.max(1); E["lg_best"] = srt[:, -1]
    E["n_sym"] = sym.sum(1).astype(np.int16); E["p_sym"] = (pr * sym).sum(1)
    mb = ma[ar, best]
    near = np.abs(ma / np.maximum(mb, 1e-9)[:, None] - 1) < 0.1
    E["p_near"] = (pr * (sym & near)).sum(1)  # probability of the symmetric interpretations within 10% of the chosen m_avg
    top = np.argsort(-lgm, 1)[:, :5]
    mt = ma[ar[:, None], top] / np.maximum(mb, 1e-9)[:, None]
    E["mspread3"] = mt[:, :3].max(1) - mt[:, :3].min(1); E["mspread5"] = mt.max(1) - mt.min(1)
    E["agree"] = best == best_sig
    sls = np.where(ok, sl, -np.inf); ss = np.sort(sls, 1)
    E["sl_margin"] = ss[:, -1] - ss[:, -2]; E["sl_best"] = ss[:, -1]
    lps = sls - sls.max(1, keepdims=True); lps = lps - np.log(np.exp(lps).sum(1, keepdims=True)); prs = np.exp(lps)
    E["sl_ent"] = -(prs * np.where(prs > 0, lps, 0)).sum(1)
    E["sl_ma_sig"] = ma[ar, best_sig]; E["sl_A_sig"] = A[ar, best_sig]
    E = {k: (x.astype(np.float32) if x.dtype == np.float64 and k not in ("w", "fb_per_w") else x) for k, x in E.items()}
    r = np.where(ps >= rich_pmin)[0] if kind == "qcd" else ar
    key = np.where(sym[r], lgm[r], -np.inf)
    o = np.argsort(-key, 1)[:, :K]
    rr = np.arange(len(r))[:, None]
    good = np.isfinite(key[rr, o])
    d1, d2 = (pts1 - m1)[r][rr, o], (pts2 - m2)[r][rr, o]
    mas = ma[r][rr, o]
    S = {"S_lp": np.where(good, lp[r][rr, o], -np.inf).astype(np.float16), "S_lps": np.where(good, lps[r][rr, o], -np.inf).astype(np.float16),
         "S_sl": np.where(good, sl[r][rr, o], -np.inf).astype(np.float16), "S_ma": np.where(good, mas, 0).astype(np.float32),
         "S_A": np.where(good, A[r][rr, o], 1).astype(np.float16),
         "S_dmin": np.where(good, np.minimum(d1, d2) / np.maximum(mas, 1e-9), 0).astype(np.float16),
         "S_dmax": np.where(good, np.maximum(d1, d2) / np.maximum(mas, 1e-9), 0).astype(np.float16),
         "S_idx": np.where(good, o, -1).astype(np.int8), "rich_local": r}
    return meta, E, S


def main():
    kind, out, rich_pmin = sys.argv[1:4]
    files = []
    for a in sys.argv[4:]:
        files += sorted(glob.glob(f"{a}/**/*.h5", recursive=True)) if os.path.isdir(a) else [a]
    t0 = time.time(); metas, Es, Ss, bad, off = [], [], [], [], 0
    for k, f in enumerate(files):
        try:
            meta, E, S = one(f, kind, float(rich_pmin))
        except Exception as e:  # noqa: BLE001
            bad.append((f, repr(e))); continue
        metas.append(meta)
        if E is not None:
            S["rich_idx"] = S.pop("rich_local") + off
            off += len(E["w"]); Es.append(E); Ss.append(S)
        if k % 20 == 0:
            print(k, len(files), f"{time.time() - t0:.0f}s", flush=True)
    d = {}
    if Es:
        d = {k: np.concatenate([e[k] for e in Es]) for k in Es[0]}
        d.update({k: np.concatenate([i[k] for i in Ss]) for k in Ss[0]})
    for k in ("fid", "n_tried", "fb_per_w", "n", "n_kept", "mass"):
        d[f"file_{k}"] = np.array([m.get(k, 0) for m in metas])
    d["n_bad"] = np.array(len(bad))
    np.savez_compressed(out, **d)
    print(f"{len(files)} files, {len(bad)} unreadable, {off} events kept of {sum(m['n'] for m in metas)}, {len(d.get('rich_idx', []))} rich, {time.time() - t0:.0f}s", bad[:3], flush=True)


if __name__ == "__main__":
    main()
