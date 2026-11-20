"""F4 audit: signal-in-peak ceilings per grouping scheme / jet definition, loss ladder, r14A on the same events.
usage: ceil.py <pattern with {M}> <outdir> [masses...]"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

import f4lib as L

ONNX = "/home/llee/CombinatorialSolver/models/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx"


def r14a(jc):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 3
    s = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    x = jc["sp4"].astype(np.float32)
    lg, ps, ms = [], [], []
    for i in range(0, len(x), 4096):
        o = s.run(["logits", "p_signal", "masses"], {"four_momenta": x[i:i + 4096]})
        b = o[0].argmax(1)
        lg.append(b)
        ps.append(o[1])
        ms.append(o[2][np.arange(len(b)), b])
    return np.concatenate(lg), np.concatenate(ps), np.concatenate(ms)


def run(args):
    pattern, outdir, M = args
    d = L.load_signal(pattern.format(M=M), max_files=int(os.environ.get("F4_MAXFILES", "0")) or None)
    n = len(d["ANC/n"])
    assert n == d["n_tried"], (n, d["n_tried"])
    R = {"mass": M, "n": n, "pass_sel": float(d["EventVars/pass_sel"].mean()), "schemes": {}, "ladder": {}, "sizes": {}}
    wins = {w: (lo * M, hi * M) for w, (lo, hi) in L.WIN.items()}

    def met(m1, m2, ok):
        return L.metrics(np.asarray(m1, float), np.asarray(m2, float), ok, M, n)

    defs = {"std": dict(ptmin=30.0, etamax=3.0), "pt20": dict(ptmin=20.0, etamax=3.0),
            "eta5": dict(ptmin=30.0, etamax=5.0), "pt20eta5": dict(ptmin=20.0, etamax=5.0)}
    jcs = {k: L.jet_collection(d, **v) for k, v in defs.items()}
    # the standard collection must reproduce the stored network input
    src = np.stack([d["INPUTS/Source/" + k][:, :L.K] for k in ("pt", "eta", "phi", "mass")], -1)
    R["std_matches_source"] = float(np.abs(jcs["std"]["ptetaphim"] - src).max())
    jcs["R08"] = L.recluster(jcs["std"], 0.8, 30.0)
    jcs["R08pt20"] = L.recluster(jcs["pt20"], 0.8, 30.0)
    jcs["R12"] = L.recluster(jcs["std"], 1.2, 30.0)

    for name, jc in jcs.items():
        fat = name.startswith("R")
        R["schemes"][name] = S = {}
        R["schemes"][name + "/njets"] = [float((jc["valid"].sum(1) >= k).mean()) for k in range(1, 11)]
        for N in ((4, 6, 8, 10) if fat else (7, 8, 9, 10, 12)):
            m1, m2, n1, n2 = L.flex_assign(jc, N)
            both = (n1 >= 1) & (n2 >= 1)
            S[f"flex{N}"] = met(m1, m2, both)
            S[f"flex{N}_s25"] = met(m1, m2, (n1 >= 2) & (n1 <= 5) & (n2 >= 2) & (n2 <= 5))
            S[f"flex{N}_s24"] = met(m1, m2, (n1 >= 2) & (n1 <= 4) & (n2 >= 2) & (n2 <= 4))
            S[f"flex{N}_s15"] = met(m1, m2, both & (n1 <= 5) & (n2 <= 5))
            if not fat and N <= 10:
                m1, m2, ok, cap = L.best33_assign(jc, N)
                S[f"b33_{N}"] = met(m1, m2, ok)
            if name in ("std",) or (fat and name == "R08"):
                tm1, tm2, _, _ = L.flex_assign(jc, N, key="tp4")
                S[f"flex{N}_truejets"] = met(tm1, tm2, both)
            if N == 10 or (fat and N == 6):
                sz = {}
                tot = m1 + m2
                A = np.abs(m1 - m2) / np.maximum(tot, 1e-9)
                inw = both & (A < 0.1) & (0.5 * tot > wins["w85"][0]) & (0.5 * tot < wins["w85"][1])
                for a in range(0, 7):
                    for b in range(a, 7):
                        k = ((n1 == a) & (n2 == b)) | ((n1 == b) & (n2 == a))
                        if k.sum() > 0:
                            sz[f"{a}+{b}"] = [float(k.mean()), float(inw[k].mean())]
                R["sizes"][name] = sz
        # exhaustive: mass oracle (any grouping in the window) and the minimum-asymmetry proxy
        ex = [("33", N, 3, 3, None) for N in (7, 8, 9, 10)] + [("s25", N, 2, 5, None) for N in (7, 8, 9, 10)] + \
             [("s24", N, 2, 4, None) for N in (7, 8, 10)]
        if fat:
            ex = [("s13", N, 1, 3, None) for N in (4, 6)] + [("s14", N, 1, 4, None) for N in (6, 8)]
        if name not in ("std", "R08"):
            ex = [e for e in ex if e[1] in (7, 10, 6)]
        for tag, N, a, b, sizes in ex:
            r = L.exhaustive(jc["p4"], jc["valid"], N, a, b, wins, sizes)
            o = {"any_" + w: float(r["any_" + w].mean()) for w in wins}
            mm = r["minA_m"].astype(float)
            sym = r["has"] & (r["minA_A"] < 0.1)
            for w, (lo, hi) in wins.items():
                o["minA_" + w] = float((sym & (mm > lo) & (mm < hi)).mean())
            o["npairs"] = int(len(L.pair_list(N, a, b, sizes)[0]))
            o["nsym_mean"] = float(r["nsym"].mean())
            S[f"ex_{tag}_{N}"] = o

    # r14A on the same events (leading 12 standard jets; events with < 6 jets are not selected)
    std = jcs["std"]
    b, ps, ms = r14a(std)
    ok = d["EventVars/pass_sel"]
    R["schemes"]["std"]["r14A"] = met(ms[:, 0], ms[:, 1], ok)
    for cut in (0.5, 0.9):
        R["schemes"]["std"][f"r14A_p{cut}"] = met(ms[:, 0], ms[:, 1], ok & (ps > cut))
    np.savez_compressed(f"{outdir}/r14a_m{M}.npz", best=b, p_signal=ps, masses=ms, ok=ok)
    # is the r14A grouping the ancestry-best 3+3? does it agree with the majority labels?
    asg = json.load(open(ONNX.replace(".onnx", ".json")))["assignments"]
    G = np.array([list(x[1]) + list(x[2]) for x in asg])[b]
    lab = np.where(std["valid"][:, :7], std["lpt"][:, :7].argmax(-1), 2)
    r = np.arange(n)[:, None]
    l1, l2 = lab[r, G[:, :3]], lab[r, G[:, 3:]]
    pure = ((l1 == 0).all(1) & (l2 == 1).all(1)) | ((l1 == 1).all(1) & (l2 == 0).all(1))
    R["r14A_pure33"] = float((pure & ok).mean())
    tot = ms.sum(1)
    A = np.abs(ms[:, 0] - ms[:, 1]) / np.maximum(tot, 1e-9)
    inw = ok & (A < 0.1) & (0.5 * tot > wins["w85"][0]) & (0.5 * tot < wins["w85"][1])
    R["r14A_inwin_pure33"] = float((inw & pure).mean())
    R["r14A_inwin_impure"] = float((inw & ~pure).mean())

    # ---------------- loss ladder (labelled parton momenta -> jets), N = 7 and 10
    g = d["truth/gluino_p4"].astype(float)
    dp = d["truth/desc_p4"].astype(float)
    allok = np.ones(n, bool)
    lad = R["ladder"]
    lad["0 gluino (Breit-Wigner)"] = met(L.mass(g[:, 0]), L.mass(g[:, 1]), allok)
    lad["1 all descendant partons"] = met(L.mass(dp[:, 0]), L.mass(dp[:, 1]), allok)
    lp4, tr, sm = d["ANC/label_p4"].astype(float), d["ANC/true"], d["ANC/smeared"]
    ex = np.arange(tr.shape[1])[None] < d["ANC/n"][:, None]

    def lsum(mask):
        s = (lp4 * mask[..., None, None]).sum(1)
        return L.mass(s[:, 0]), L.mass(s[:, 1])

    lad["2 partons in any jet (pT>10, |eta|<4.9)"] = met(*lsum(ex), allok)
    lad["3a + jet |eta| < 3"] = met(*lsum(ex & (np.abs(tr[..., 1]) < 3)), allok)
    lad["3b + jet pT(true) > 30 (all eta)"] = met(*lsum(ex & (tr[..., 0] > 30)), allok)
    lad["4 + jet pT(true) > 30 and |eta| < 3"] = met(*lsum(ex & (np.abs(tr[..., 1]) < 3) & (tr[..., 0] > 30)), allok)
    for N in (20, 10, 7):
        v = std["valid"] if N <= L.K else None
        if N == 20:
            sel = ex & (sm[..., 0] > 30) & (np.abs(sm[..., 1]) < 3) & (tr[..., 0] >= 24)
            lad["5 partons in all selected jets (smeared pT > 30)"] = met(*lsum(sel), allok)
            continue
        vN = std["valid"][:, :N]
        l4 = std["lp4"][:, :N].astype(float)
        s = (l4 * vN[..., None, None]).sum(1)
        lad[f"6 partons in the leading {N} jets"] = met(L.mass(s[:, 0]), L.mass(s[:, 1]), allok)
        lab = np.where(vN, std["lpt"][:, :N].argmax(-1), 2)
        own = [(l4[:, :, k] * (lab == k)[..., None]).sum(1) for k in (0, 1)]
        lad[f"7 own partons in majority jets, leading {N}"] = met(L.mass(own[0]), L.mass(own[1]), allok)
        allp = [(l4.sum(2) * (lab == k)[..., None]).sum(1) for k in (0, 1)]
        lad[f"8 all partons in majority jets (contamination), leading {N}"] = met(L.mass(allp[0]), L.mass(allp[1]), allok)
        m1, m2, n1, n2 = L.flex_assign(std, N, key="tp4")
        lad[f"9 hadron-level jets, majority, leading {N}"] = met(m1, m2, (n1 > 0) & (n2 > 0))
        m1, m2, n1, n2 = L.flex_assign(std, N)
        lad[f"10 smeared jets, majority, leading {N}"] = met(m1, m2, (n1 > 0) & (n2 > 0))
        m1, m2, ok33, _ = L.best33_assign(std, N)
        lad[f"11 smeared jets, ancestry-best 3+3, leading {N}"] = met(m1, m2, ok33)
    lad["12 r14A"] = R["schemes"]["std"]["r14A"]

    # ---------------- quark-level classification (decay quarks before FSR)
    q = d["truth/quarks_before_fsr"].astype(float)  # (n, 6, 3) pt, eta, phi
    dphi = np.abs((q[:, :, None, 2] - q[:, None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
    dr = np.sqrt((q[:, :, None, 1] - q[:, None, :, 1]) ** 2 + dphi**2) + 9 * np.eye(6)[None]
    acc, hard, iso = (np.abs(q[..., 1]) < 3).all(1), (q[..., 0] > 30).all(1), dr.min((1, 2)) > 0.4
    ji = d["truth/jet_index"]
    reco7 = (ji >= 0).all(1) & (ji < 7).all(1) & np.array([len(set(r)) == 6 for r in ji])
    R["quarks"] = {"all |eta|<3": float(acc.mean()), "+ all pT>30": float((acc & hard).mean()),
                   "+ no pair dR<0.4": float((acc & hard & iso).mean()),
                   "+ no pair dR<0.8": float((acc & hard & (dr.min((1, 2)) > 0.8)).mean()),
                   "six quarks on distinct jets among leading 7 (current truth)": float(reco7.mean()),
                   "same, among all stored jets": float(((ji >= 0).all(1) & np.array([len(set(r)) == 6 for r in ji])).mean()),
                   "fail: a quark |eta|>3": float((~acc).mean()), "fail: a quark pT<30 (in acc)": float((acc & ~hard).mean()),
                   "fail: merged pair dR<0.4 (in acc, hard)": float((acc & hard & ~iso).mean())}
    # label purity of the leading jets
    fr = std["lpt"][:, :10] / np.maximum(std["lpt"][:, :10].sum(-1, keepdims=True), 1e-9)
    v = std["valid"][:, :10]
    R["purity"] = {"mean label fractions (g1, g2, other), leading 10": [float((fr[..., k] * v).sum() / v.sum()) for k in range(3)],
                   "jets with majority fraction < 0.8": float(((fr.max(-1) < 0.8) & v).sum() / v.sum()),
                   "jets with majority = other": float(((fr.argmax(-1) == 2) & v).sum() / v.sum()),
                   "per rank majority=other": [float(((fr[:, k].argmax(-1) == 2) & v[:, k]).sum() / max(v[:, k].sum(), 1)) for k in range(10)]}
    json.dump(R, open(f"{outdir}/ceil_m{M}.json", "w"), indent=1)
    return M


if __name__ == "__main__":
    pattern, outdir = sys.argv[1:3]
    masses = [int(x) for x in sys.argv[3:]] or [400, 700, 1000, 1500, 2000]
    os.makedirs(outdir, exist_ok=True)
    with Pool(min(5, len(masses))) as p:
        for M in p.imap_unordered(run, [(pattern, outdir, M) for M in masses]):
            print("done", M, flush=True)
