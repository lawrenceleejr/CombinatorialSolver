"""F4 audit: label-free FSR recovery prototype. Start from a 3+3 grouping (r14A's choice, or the ancestry-best 3+3 in 7)
and attach every other jet among the leading N to the group holding the nearest jet if that dR < Rc.
Signal: fraction of all produced events with A<0.1 in window; QCD: weighted fraction (held-out qcdht / qcdvlm).
Also: the correct per-size group masses under ancestry majority (fixes the size table) and attach purity.
usage: attach.py"""
import glob, json, os
import h5py, numpy as np
import onnxruntime as ort
import f4lib as L
ONNX = "/home/llee/CombinatorialSolver/models/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx"
ASG = json.load(open(ONNX.replace(".onnx", ".json")))["assignments"]
G = np.array([list(x[1]) + list(x[2]) for x in ASG])
RCS = (0.0, 0.6, 0.8, 1.0, 1.2)
so = ort.SessionOptions(); so.intra_op_num_threads = 4
S = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])


def r14a(p4):
    b, ps = [], []
    for i in range(0, len(p4), 4096):
        o = S.run(["logits", "p_signal"], {"four_momenta": p4[i:i + 4096].astype(np.float32)})
        b.append(o[0].argmax(1)); ps.append(o[1])
    return np.concatenate(b), np.concatenate(ps)


def etaphi(p4):
    pt = np.hypot(p4[..., 1], p4[..., 2])
    return np.arcsinh(p4[..., 3] / np.maximum(pt, 1e-9)), np.arctan2(p4[..., 2], p4[..., 1])


def attach(p4, valid, grp, N, rc):
    """grp: (n,6) jet indices (first 3 = group 1). Returns m1, m2, membership (n, N) in {0,1,2}."""
    n = len(p4); r = np.arange(n)[:, None]
    mem = np.full((n, N), 2)
    mem[r, grp[:, :3]] = 0; mem[r, grp[:, 3:]] = 1
    eta, phi = etaphi(p4[:, :N])
    if rc > 0:
        de = eta[:, :, None] - eta[:, None, :]
        dp = (phi[:, :, None] - phi[:, None, :] + np.pi) % (2 * np.pi) - np.pi
        dr = np.sqrt(de**2 + dp**2)
        cand = (mem == 2) & valid[:, :N]
        d0 = np.where((mem == 0)[:, None, :], dr, 9).min(-1)
        d1 = np.where((mem == 1)[:, None, :], dr, 9).min(-1)
        new = np.where(d0 <= d1, 0, 1)
        mem = np.where(cand & (np.minimum(d0, d1) < rc), new, mem)
    g = [(p4[:, :N] * (mem == k)[..., None]).sum(1) for k in (0, 1)]
    return L.mass(g[0]), L.mass(g[1]), mem


def frac(m1, m2, ok, M, w=None):
    tot = m1 + m2; A = np.abs(m1 - m2) / np.maximum(tot, 1e-9); x = 0.5 * tot / M
    w = np.ones(len(m1)) if w is None else w
    return {k: float((w * (ok & (A < .1) & (x > lo) & (x < hi))).sum()) for k, (lo, hi) in L.WIN.items()}


out = {"sig": {}, "bkg": {}, "sizes": {}}
for M in (400, 700, 1000, 1500, 2000):
    d = L.load_signal(f"samples/sigan_m{M}_c*.h5", max_files=6)
    n = len(d["ANC/n"]); jc = L.jet_collection(d); ok = d["EventVars/pass_sel"]
    b, ps = r14a(jc["sp4"])
    m1, m2, n1, n2 = L.flex_assign(jc, 10)
    tot = m1 + m2; A = np.abs(m1 - m2) / np.maximum(tot, 1e-9); x = 0.5 * tot / M
    inw = (n1 > 0) & (n2 > 0) & (A < .1) & (x > .85) & (x < 1.1)
    out["sizes"][M] = {f"{a}+{c}": [float(k.mean()), float(inw[k].mean()) if k.any() else 0.0]
                       for a in range(1, 7) for c in range(a, 7)
                       for k in [((n1 == a) & (n2 == c)) | ((n1 == c) & (n2 == a))] if k.mean() > 0.005}
    lab = np.where(jc["valid"][:, :10], jc["lpt"][:, :10].argmax(-1), 2)
    # ancestry-best 3+3 in 7 as a grouping
    T = L.ordered_33(7); lpt = jc["lpt"][:, :7]
    sc = lpt[:, T[:, :3], 0].sum(-1) + lpt[:, T[:, 3:], 1].sum(-1)
    sc = np.where(jc["valid"][:, :7][:, T].all(-1), sc, -1); gb = T[sc.argmax(1)]
    for start, grp, okk in (("r14A", G[b], ok), ("r14A_p0.9", G[b], ok & (ps > 0.9)), ("anc33", gb, ok)):
        for N in (7, 10, 12):
            for rc in RCS:
                if rc == 0 and N != 7: continue
                a1, a2, mem = attach(jc["sp4"], jc["valid"], grp, N, rc)
                r = frac(a1, a2, okk, M); r = {k: v / n for k, v in r.items()}
                # purity of attached jets (signal truth): attached to the group whose gluino owns the majority
                att = (mem[:, 6:] < 2) if False else None
                lb = np.where(jc["valid"][:, :N], jc["lpt"][:, :N].argmax(-1), 2)
                core = np.zeros_like(mem, bool); core[np.arange(n)[:, None], grp] = True
                g1lab = np.array([np.bincount(lb[i, grp[i, :3]], minlength=3)[:2].argmax() for i in range(n)])
                added = (~core) & (mem < 2)
                corr = added & (lb == np.where(mem == 0, g1lab[:, None], 1 - g1lab[:, None]))
                r["att_per_evt"] = float(added.sum() / n); r["att_correct"] = float(corr.sum() / max(added.sum(), 1))
                out["sig"][f"{M}/{start}/N{N}/rc{rc}"] = r
    print("sig", M, flush=True)

for tag, files, masses in (("qcdht", [f"/scratch/llee/pairnet/prod_ht/qcdht_c{i}.h5" for i in range(1400, 1440)], (700, 1000, 1500, 2000)),
                           ("qcdvlm", [f"/scratch/llee/pairnet/prod_lowmass/qcdvlm_c{i}.h5" for i in range(1050, 1062)], (400, 700))):
    xs, ws = [], []
    for fn in files:
        with h5py.File(fn) as f:
            xs.append(np.stack([f["INPUTS/Source/" + k][:, :L.K] for k in ("pt", "eta", "phi", "mass")], -1))
            ws.append(f["EventVars/normweight"][:] * f.attrs["sigma_mb"] / f.attrs["weight_sum"])
    x = np.concatenate(xs); w = np.concatenate(ws); sw = w.sum(); valid = x[..., 0] > 0
    p4 = L.p4_from_ptetaphim(x) * valid[..., None]
    b, ps = r14a(p4); okb = valid.sum(1) >= 6
    for start, okk in (("r14A", okb), ("r14A_p0.9", okb & (ps > 0.9))):
        for N in (7, 10, 12):
            for rc in RCS:
                if rc == 0 and N != 7: continue
                a1, a2, _ = attach(p4, valid, G[b], N, rc)
                for M in masses:
                    r = frac(a1, a2, okk, M, w)
                    out["bkg"][f"{tag}/{M}/{start}/N{N}/rc{rc}"] = {k: v / sw for k, v in r.items()}
    out["bkg"][f"{tag}/n"] = [len(w), float(sw**2 / (w**2).sum())]
    print("bkg", tag, flush=True)
json.dump(out, open("out/attach.json", "w"), indent=1)
