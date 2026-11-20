"""Item 1: code-level tests of r14A (ONNX and PyTorch): invariances, padding, ordering, ties, NaN safety, precision, batch."""
import sys, os, json
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.expanduser("~/CombinatorialSolver")); sys.path.insert(0, os.path.expanduser("~/CombinatorialSolver/scripts"))
from lib import *
from export_pairnet_onnx import PairNetONNX
from src.pairnet import load_checkpoint
net, _ = load_checkpoint(os.path.expanduser("~/CombinatorialSolver/models/r14A/pairnet_r14A_model.pt"))
wr32 = PairNetONNX(net).eval()
import copy
wr64 = copy.deepcopy(wr32).double()
sess = session(2)
OUTS = ["logits", "p_signal", "signal_logits", "masses", "m_est"]
R = {}


def onnx(p4, bs=1024):
    return run(sess, p4, bs=bs, outs=OUTS)


_float = torch.Tensor.float


def pt_run(wr, p4, dtype):
    # the model calls mask.float(); for the float64 reference make that a double
    torch.Tensor.float = (lambda self: self.double()) if dtype == torch.float64 else _float
    with torch.no_grad():
        o = [[] for _ in OUTS]
        for i in range(0, len(p4), 512):
            r = wr(torch.tensor(p4[i:i + 512], dtype=dtype))
            for a, b in zip(o, r):
                a.append(b.numpy())
    return [np.concatenate(a) for a in o]


def logit(p):
    p = p.astype(np.float64); return np.log(p) - np.log1p(-p)


def cmp(name, a, b, extra=""):
    """a, b: output lists. Report gate-logit, p, argmax differences."""
    la, lb = logit(a[1]), logit(b[1]); ok = np.isfinite(la) & np.isfinite(lb)
    d = np.abs(la - lb)[ok]
    r = {"n": int(len(la)), "nan": int((~np.isfinite(a[1])).sum() + (~np.isfinite(b[1])).sum() + (~np.isfinite(a[0])).any(1).sum() + (~np.isfinite(b[0])).any(1).sum()),
         "max_dlogit": float(d.max()), "med_dlogit": float(np.median(d)), "p99_dlogit": float(np.quantile(d, 0.99)),
         "max_dp": float(np.abs(a[1].astype(float) - b[1]).max()), "argmax_diff": float((a[0].argmax(1) != b[0].argmax(1)).mean()),
         "rel_d_mest": float(np.nanmax(np.abs(a[4] / b[4] - 1)))}
    for t in (0.99, 0.997, 0.9993):
        r[f"flip@{t}"] = float(((a[1] >= t) != (b[1] >= t)).mean()); r[f"pass@{t}"] = float((a[1] >= t).mean())
    R[name] = r
    print(f"{name:34s} n={r['n']} nan={r['nan']} dlogit max {r['max_dlogit']:.2e} p99 {r['p99_dlogit']:.2e} med {r['med_dlogit']:.2e}  argmax diff {r['argmax_diff']:.2e}"
          f"  flips@.997 {r['flip@0.997']:.2e} (pass {r['pass@0.997']:.3f})  d m_est {r['rel_d_mest']:.1e} {extra}", flush=True)
    return r


def p4of(pt, eta, phi, m, mask):
    return sortpad(pt, eta, phi, m, mask)[2]


# ---------------- events: 3000 signal 700 (incl+boost) + 3000 tail QCD (P_nom >= 0.9) + 2000 unfiltered QCD
ev = []
for f, n in (("sig700_0", 3000), ("TvlmT_0", 3000), ("Uvlm_0", 2000), ("sig2000_0", 1500), ("ThtT_0", 1500)):
    c = np.load(f"cache/chunks/{f}.npz"); ev.append([c[k][:n].astype(np.float64) if k != "mask" else c[k][:n] for k in ("pt", "eta", "phi", "m", "mask")])
raw = [np.concatenate([e[i] for e in ev]) for i in range(5)]
pt, eta, phi, m, mask = raw
N = len(pt); rng = np.random.default_rng(3)
p4 = p4of(*raw)
base = onnx(p4)
print("events", N, " P>=0.997:", (base[1] >= 0.997).sum(), " max p", base[1].max(), " max gate logit", logit(base[1]).max())
R["max_p"] = float(base[1].max())

print("\n== ONNX vs PyTorch")
t32 = pt_run(wr32, p4, torch.float32); t64 = pt_run(wr64, p4, torch.float64)
cmp("onnx vs torch32", base, t32); cmp("onnx vs torch64", base, t64); cmp("torch32 vs torch64", t32, t64)
tail = base[1] >= 0.99
cmp("onnx vs torch64, P>=0.99", [x[tail] for x in base], [x[tail] for x in t64])
print("   pairing margin (top1-top2 logit) where argmax differs onnx/torch64:", np.sort(base[0], 1)[base[0].argmax(1) != t64[0].argmax(1)][:, -1:].ravel()[:5],
      (np.sort(base[0], 1)[:, -1] - np.sort(base[0], 1)[:, -2])[base[0].argmax(1) != t64[0].argmax(1)][:8])
gap = np.sort(base[0], 1)[:, -1] - np.sort(base[0], 1)[:, -2]
R["gap_lt_0.01"] = float((gap < 0.01).mean()); R["gap_lt_0.1"] = float((gap < 0.1).mean())
print("   fraction of events with top-2 pairing logit gap < 0.01: %.4f, < 0.1: %.4f" % (R["gap_lt_0.01"], R["gap_lt_0.1"]))

print("\n== batch size / composition (ONNX)")
cmp("bs=1 vs bs=1024 (first 400)", onnx(p4[:400], bs=1), [x[:400] for x in base])
perm = rng.permutation(N); o = onnx(p4[perm], bs=37); cmp("shuffled, bs=37", [x[np.argsort(perm)] for x in o], base)

print("\n== input precision (E computed in float32 by the user vs float64 then cast)")
a = [x.astype(np.float32) for x in sortpad(*raw)[:1]]
pts, masks, _ = sortpad(*raw)
o12 = np.argsort(np.where(mask, -pt, np.inf), axis=1, kind="stable")[:, :12]
e32, f32, m32 = (np.take_along_axis(x, o12, 1).astype(np.float32) for x in (eta, phi, m)); p32 = pts[:, :12].astype(np.float32)
px, py, pz = p32 * np.cos(f32), p32 * np.sin(f32), p32 * np.sinh(e32)
p4f = np.stack([np.sqrt(px * px + py * py + pz * pz + m32 * m32), px, py, pz], -1)
assert p4f.dtype == np.float32
cmp("float32-built four-vectors", onnx(p4f), base)
# what jet mass does the network actually see (float32 E^2 - p^2)?
mrec = np.sqrt(np.maximum(p4f[..., 0].astype(np.float32) ** 2 - (p4f[..., 1:] ** 2).sum(-1), 0)); ok = masks[:, :12]
rel = (mrec[ok] / np.maximum(m32[ok], 1e-6) - 1)
R["m_float32_rel_err_q68"] = float(np.quantile(np.abs(rel), 0.68)); R["m_float32_rel_err_q99"] = float(np.quantile(np.abs(rel), 0.99)); R["m_float32_zero_frac"] = float((mrec[ok] == 0).mean())
print("   jet mass recovered inside the graph from float32 (E,p): |rel err| 68%% %.2e, 99%% %.2e, max %.2e; m -> 0 for %.2e of jets" % (
    R["m_float32_rel_err_q68"], R["m_float32_rel_err_q99"], np.abs(rel).max(), R["m_float32_zero_frac"]))

print("\n== exact symmetries")
for c in (1e-3, 0.5, 2.0, 1e3):
    o = onnx(p4 * c); r = cmp(f"scale x{c:g}", o, base)
    print("      masses scale: max rel dev %.1e" % np.nanmax(np.abs(o[3] / (c * np.maximum(base[3], 1e-9)) - 1)[base[3] > 1]))
a = rng.uniform(-np.pi, np.pi, (N, 1))
cmp("phi rotation (random)", onnx(p4of(pt, eta, phi + a, m, mask)), base)
cmp("eta flip", onnx(p4of(pt, -eta, phi, m, mask)), base)
cmp("phi reflection (phi -> -phi)", onnx(p4of(pt, eta, -phi, m, mask)), base, "(parity: a true symmetry, not built in)")
cmp("z boost by eta 0.5 (not a symmetry)", onnx(p4of(pt, eta + 0.5, phi, m, mask)), base, "(longitudinal boost invariance is NOT built in)")

print("\n== ordering")
q = p4.copy()
for i in range(N):
    q[i, 7:] = q[i, 7 + rng.permutation(5)]
cmp("permute jets 8-12", onnx(q), base)
q = p4.copy(); pm = np.stack([rng.permutation(7) for _ in range(N)]); q[:, :7] = np.take_along_axis(p4[:, :7], pm[:, :, None], 1)
o = onnx(q)
r = cmp("permute jets 1-7 (unsorted input)", o, base)
# pairing equivariance: map chosen jets back
ar = np.arange(N); sel_new = np.sort(np.take_along_axis(pm, G[o[0].argmax(1)][:, :3], 1), 1), np.sort(np.take_along_axis(pm, G[o[0].argmax(1)][:, 3:], 1), 1)
b0 = G[base[0].argmax(1)]; s0 = np.sort(b0[:, :3], 1), np.sort(b0[:, 3:], 1)
same = ((sel_new[0] == s0[0]).all(1) & (sel_new[1] == s0[1]).all(1)) | ((sel_new[0] == s0[1]).all(1) & (sel_new[1] == s0[0]).all(1))
R["perm7_same_pairing"] = float(same.mean()); print("      same physical pairing after permuting the leading 7: %.4f" % same.mean())
q = p4.copy(); q[:, [6, 7]] = q[:, [7, 6]]
cmp("swap jets 7 and 8", onnx(q), base, "(changes the candidate set: expected to differ)")

print("\n== padding / jet multiplicity")
q = p4.copy(); q[:, 7:] = 0; o7 = onnx(q); cmp("only 7 jets given (8-12 zeroed)", o7, base, "(context removed: expected to differ)")
has = masks[:, :12].sum(1) >= 9
q = p4.copy(); q[:, 8] = 0  # hole in slot 9 instead of compacted
qq = p4.copy(); qq[:, 8:11] = p4[:, 9:12]; qq[:, 11] = 0
cmp("zero row in the middle vs compacted", onnx(q), onnx(qq))
for nj in (6, 5, 3, 1, 0):
    q = p4.copy(); q[:, nj:] = 0; o = onnx(q)
    nanp = (~np.isfinite(o[1])).mean(); nanl = (~np.isfinite(o[0])).any(1).mean(); nanm = (~np.isfinite(o[4])).mean()
    b = o[0].argmax(1); valid = (q[:, :7, 0] > 0)[ar[:, None], G[b]].all(1)
    R[f"nj{nj}"] = {"nan_p": float(nanp), "nan_logits": float(nanl), "nan_mest": float(nanm), "valid_argmax": float(valid.mean()),
                    "p_mean": float(np.nanmean(o[1])), "p_max": float(np.nanmax(o[1])) if nanp < 1 else None, "frac_p_gt_0.5": float((o[1] > 0.5).mean())}
    print(f"   {nj} jets: NaN p {nanp:.3f} NaN logits {nanl:.3f} NaN m_est {nanm:.3f}; argmax pairing valid {valid.mean():.3f}; p mean {np.nanmean(o[1]) if nanp < 1 else float('nan'):.4f} "
          f"max {np.nanmax(o[1]) if nanp < 1 else float('nan'):.4f}; logits row0 max {o[0][0].max():.3g}")
t0 = pt_run(wr32, np.zeros((2, 12, 4)), torch.float32); print("   torch32, all-zero event: p", t0[1], "logits finite", np.isfinite(t0[0]).all())

print("\n== ties and degenerate inputs")
q = p4.copy(); q[:, 1] = q[:, 0]; o = onnx(q); print("   duplicate leading jet: NaN", (~np.isfinite(o[1])).sum(), "p mean", o[1].mean())
q = p4.copy(); q[:, 6] = q[:, 5]; o = onnx(q); print("   jets 6 = 7 identical: NaN", (~np.isfinite(o[1])).sum(), (~np.isfinite(o[0])).sum())
gg = np.sort(o[0], 1); print("      exact logit ties top1 = top2: %.4f (expected: pairings differing only by 6 <-> 7)" % (gg[:, -1] == gg[:, -2]).mean())
R["tie_exact_dup67"] = float((gg[:, -1] == gg[:, -2]).mean())
o = onnx(p4of(pt, eta, phi, 0 * m, mask)); r = cmp("all jets massless", o, base, "(physics change, see item 2)"); print("      NaN:", (~np.isfinite(o[1])).sum(), (~np.isfinite(o[0])).sum(), (~np.isfinite(o[4])).sum())
# massless but E rounded BELOW |p| (E^2 - p^2 < 0)
q = p4of(pt, eta, phi, 0 * m, mask).astype(np.float32); q[..., 0] = np.nextafter(q[..., 0], np.float32(0)); o2 = onnx(q); cmp("massless, E < |p| by 1 ulp", o2, o)
q = p4.copy(); q[:, 11] = [1e-3, 1e-3, 0, 0]; o = onnx(q); print("   a 1 MeV 12th jet: NaN", (~np.isfinite(o[1])).sum()); cmp("tiny-pT 12th jet vs none", o, onnx(np.concatenate([p4[:, :11], np.zeros((N, 1, 4))], 1)))
q = p4.copy(); q[:, 11] = [1e-30, 1e-30, 0, 0]; o = onnx(q); print("   a 1e-30 GeV 12th jet (pt>0 but below clamps): NaN", (~np.isfinite(o[1])).sum())
q = p4.copy(); q[:, 0, 2] = 0; q[:, 0, 1] = np.abs(q[:, 0, 1]) + 1; o = onnx(q); print("   leading jet exactly on the x axis: NaN", (~np.isfinite(o[1])).sum())
q = p4of(pt, np.where(np.arange(pt.shape[1]) == 3, 6.0, eta), phi, m, mask); o = onnx(q); print("   one jet at eta = 6: NaN", (~np.isfinite(o[1])).sum(), (~np.isfinite(o[4])).sum())
q = p4.copy(); q[:, 2] = np.nan; o = onnx(q); print("   NaN jet in input: NaN outputs", (~np.isfinite(o[1])).mean())
q = p4.copy(); q[:, 5, 1:3] *= -0.0; q[:, 4, 1:3] = 0; q[:, 4, 3] = 100; q[:, 4, 0] = 100; o = onnx(q); print("   jet with pT = 0 but pz != 0 (treated as padding): NaN", (~np.isfinite(o[1])).sum())
json.dump(R, open("out/item1.json", "w"), indent=1, default=float)
