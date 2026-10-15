"""
Hemisphere-mixing closure test for a trained PairNet.

  python scripts/closure.py --model runs/X/model.pt --qcd "data/pythia/qcd_s9.h5" \
      "data/pythia/qcd_s10.h5" --out runs/X/closure.npz

Builds a 3-jet-hemisphere library from 5-jet QCD events, mixes fake 6-jet events,
and evaluates the network on (a) mixed events, (b) real exactly-6-jet QCD and
(c) real >=6-jet QCD.  Reports shape agreement of m_avg before and after a cut on
the network's signal probability, plus the cut efficiency in each sample.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hemimix import build_library, mix, mix_replace  # noqa: E402
from src.pairdata import load_events, to_tensors  # noqa: E402
from src.pairnet import PairNet  # noqa: E402


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    sizes = tuple((int(x[0]), int(x[1])) for x in a["sizes"].split(",")) if a.get("sizes") else None
    net = PairNet(
        n_jets=a["n_jets"], k=a["k"], d_model=a["d_model"], nhead=a["heads"],
        num_layers=a["layers"], dim_ff=2 * a["d_model"], dropout=0.0,
        use_pair_bias=not a["no_pair_bias"], use_group_kin=not a["no_group_kin"],
        use_null=not a["no_null"], sizes=sizes,
    )
    net.load_state_dict(ck["state"])
    net.eval()
    return net, a


def pad_to(d, n):
    out = {}
    for k in ("pt", "eta", "phi", "m", "mask"):
        v = d[k][:, :n]
        if v.shape[1] < n:
            v = np.pad(v, ((0, 0), (0, n - v.shape[1])))
        out[k] = v
    out["mask"] = out["mask"].astype(bool)
    return out


@torch.no_grad()
def run(net, d, bs=1024):
    d = pad_to(d, net.n_jets)
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()}
    t = {k: (v.float() if v.dtype != torch.bool else v) for k, v in t.items()}
    A = net.n_assign
    ps, mv = [], []
    for i in range(0, len(t["pt"]), bs):
        o = net(*(t[k][i : i + bs] for k in ("pt", "eta", "phi", "m", "mask")))
        lg = o["logits"]
        best = lg[:, :A].argmax(1)
        ar = torch.arange(len(best))
        p = lg.softmax(1)
        ps.append(1 - p[:, A] if lg.shape[1] > A else p[:, :A].max(1).values)
        mv.append(0.5 * (o["m1"][ar, best] + o["m2"][ar, best]) * o["ht"])
    return torch.cat(ps).numpy(), torch.cat(mv).numpy()


def chi2_shape(a, wa, b, wb, bins):
    ha, _ = np.histogram(a, bins, weights=wa)
    hb, _ = np.histogram(b, bins, weights=wb)
    va, _ = np.histogram(a, bins, weights=wa**2)
    vb, _ = np.histogram(b, bins, weights=wb**2)
    na, nb = ha.sum(), hb.sum()
    ha, hb, va, vb = ha / na, hb / nb, va / na**2, vb / nb**2
    ok = (va + vb) > 0
    c2 = ((ha - hb) ** 2 / np.where(ok, va + vb, 1))[ok].sum()
    return float(c2), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--qcd", nargs="+", required=True)
    ap.add_argument("--psig-cut", type=float, default=None,
                    help="cut on P(sig); default: value keeping 10%% of real 6-jet QCD")
    ap.add_argument("--unweighted", action="store_true")
    ap.add_argument("--mix-mode", choices=["replace", "pair"], default="replace",
                    help="replace: swap the 2-jet hemisphere of a 5-jet event for a compatible "
                         "3-jet one (analysis method); pair: two random compatible 3-jet hemispheres")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, a = load_model(args.model)
    cr = load_events(args.qcd, 5, 3, False, None, min_jets=5, max_jets=5)
    sr6 = load_events(args.qcd, 6, 3, False, None, min_jets=6, max_jets=6)
    srall = load_events(args.qcd, a["n_jets"], 3, False, None, min_jets=6)
    lib = build_library(cr["pt"], cr["eta"], cr["phi"], cr["m"], cr["mask"], cr["w"])
    if args.mix_mode == "replace":
        mx = mix_replace(cr["pt"], cr["eta"], cr["phi"], cr["m"], cr["mask"], cr["w"], seed=1)
    else:
        mx = mix(lib, seed=1)
    for d in (cr, sr6, srall, mx):
        if args.unweighted:
            d["w"] = np.ones(len(d["pt"]))
    print(f"5-jet CR {len(cr['pt'])}  hemispheres {len(lib['pt'])}  mixed {len(mx['pt'])}  "
          f"real 6j {len(sr6['pt'])}  real >=6j {len(srall['pt'])}")

    res = {}
    out = {}
    for name, d in (("mixed", mx), ("real6", sr6), ("real6plus", srall)):
        ps, mv = run(net, d)
        out[f"{name}_psig"], out[f"{name}_mavg"], out[f"{name}_w"] = ps, mv, d["w"]
    cut = args.psig_cut
    if cut is None:
        r = out["real6_psig"]
        cut = float(np.quantile(r, 0.9))
    bins = np.linspace(0, 2000, 41)
    for ref in ("real6", "real6plus"):
        c2, nd = chi2_shape(out["mixed_mavg"], out["mixed_w"], out[f"{ref}_mavg"], out[f"{ref}_w"], bins)
        res[f"mavg_chi2ndf_nocut_vs_{ref}"] = c2 / max(nd, 1)
        sm = out["mixed_psig"] >= cut
        sr = out[f"{ref}_psig"] >= cut
        c2, nd = chi2_shape(out["mixed_mavg"][sm], out["mixed_w"][sm], out[f"{ref}_mavg"][sr],
                            out[f"{ref}_w"][sr], bins)
        res[f"mavg_chi2ndf_cut_vs_{ref}"] = c2 / max(nd, 1)
        eff = lambda s, w: float((w * s).sum() / w.sum())
        res[f"cut_eff_mixed"] = eff(sm, out["mixed_w"])
        res[f"cut_eff_{ref}"] = eff(sr, out[f"{ref}_w"])
    res["psig_cut"] = cut
    print(json.dumps(res, indent=1))
    np.savez_compressed(args.out, **out, psig_cut=cut)
    json.dump(res, open(args.out.replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
