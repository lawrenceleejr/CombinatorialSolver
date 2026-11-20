"""
Peak-level comparison on a common test set: signal in the m_avg peak, QCD in the peak, and QCD sculpting.

For each model (scores npz: logit, mavg [GeV], asym, is_signal, pmass, w) and each signal mass M:
  * window W(M) = [0.85, 1.1] M ("wide") and [0.92, 1.04] M ("narrow"), always with A < 0.1;
  * no P cut: signal fraction in W and weighted QCD in W;
  * P cut scanned (on the gate logit): best S/sqrt(B) with at least MIN_NB raw QCD events left, and the signal
    kept at the QCD level r14A has at its own best point (S at equal B, the like-for-like number).
Sculpting, from all QCD with A < 0.1: at global P cuts keeping 1 % and 0.1 % of it, the QCD efficiency per
m_avg bin divided by the average, its log-log slope (smooth reshaping) and the largest residual from a
smooth quadratic in log-log (a local bump would show up here), in units of its statistical error.

  python peak_eval.py --ref scores/r14A.npz name=scores.npz ... --out peak.json
"""

import argparse
import json

import numpy as np

MIN_NB = 20
WINDOWS = {"wide": (0.85, 1.1), "narrow": (0.92, 1.04)}
BINS = np.array([200, 300, 400, 500, 600, 700, 850, 1000, 1200, 1450, 1750, 2100, 2500])


def load(path):
    d = np.load(path)
    if "logit" in d.files:
        return {k: d[k] for k in ("logit", "mavg", "asym", "is_signal", "pmass", "w")}
    p = np.clip(d["psig"], 1e-7, 1 - 1e-7)  # train_pairnet eval.npz
    return {"logit": np.log(p / (1 - p)), "mavg": d["mavg"], "asym": d["asym"], "is_signal": d["is_signal"],
            "pmass": d["pmass"], "w": d["w"]}


def scan(d, M, lo, hi):
    s, q = d["is_signal"].astype(bool), ~d["is_signal"].astype(bool)
    ms = s & (np.abs(d["pmass"] - M) < 1)
    rel = d["mavg"] / M
    inw = (d["asym"] < 0.1) & (rel > lo) & (rel < hi)
    ns = ms.sum()
    qs = q & inw
    lq, wq = d["logit"][qs], d["w"][qs]
    o = np.argsort(-lq)
    cB, cN = np.cumsum(wq[o]), np.arange(1, len(o) + 1)
    sl = np.sort(d["logit"][ms & inw])[::-1]
    # signal passing each QCD threshold
    cS = np.searchsorted(-sl, -lq[o], side="right") / ns
    ok = cN >= MIN_NB
    fom = np.where(ok, cS / np.sqrt(np.maximum(cB, 1e-30)), 0)
    k = int(fom.argmax())
    return {"S0": float((ms & inw).sum() / ns), "B0": float(wq.sum()), "best": {"S": float(cS[k]), "B": float(cB[k]),
            "nB": int(cN[k]), "fom": float(fom[k])}, "_curve": (cB, cS, cN)}


def s_at_fixed_b(d, ref, M, lo, hi, fracs=(0.1, 0.01), nboot=40, seed=1):
    """Signal fraction at weighted QCD = f x (r14A's QCD in the window, no P cut), ratio to r14A, with a paired
    Poisson bootstrap (both models are scored on the same events)."""
    rng = np.random.default_rng(seed)
    n = len(d["w"])
    out = {}
    def one(dd, ww, B):
        s, q = dd["is_signal"].astype(bool), ~dd["is_signal"].astype(bool)
        ms = s & (np.abs(dd["pmass"] - M) < 1)
        rel = dd["mavg"] / M
        inw = (dd["asym"] < 0.1) & (rel > lo) & (rel < hi)
        qs = q & inw
        lq, wq = dd["logit"][qs], (dd["w"] * ww)[qs]
        o = np.argsort(-lq); cB = np.cumsum(wq[o])
        k = np.searchsorted(cB, B, side="right") - 1
        thr = lq[o][k] if k >= 0 else np.inf
        return ((ms & inw & (dd["logit"] >= thr)) * ww).sum() / max((ms * ww).sum(), 1e-30), k + 1
    for f in fracs:
        vals = []
        for b in range(nboot + 1):
            ww = np.ones(n) if b == 0 else rng.poisson(1.0, n).astype(float)
            r0 = ref["w"] * ww
            q = ~ref["is_signal"].astype(bool); rel = ref["mavg"] / M
            B = f * (r0 * (q & (ref["asym"] < 0.1) & (rel > lo) & (rel < hi))).sum()
            sm, nb = one(d, ww, B); sr, _ = one(ref, ww, B)
            vals.append((sm, sr, nb))
        v = np.array(vals)
        rat = v[1:, 0] / np.maximum(v[1:, 1], 1e-30)
        out[f"{f:g}"] = {"S": float(v[0, 0]), "S_ref": float(v[0, 1]), "nB": int(v[0, 2]),
                         "ratio": float(v[0, 0] / max(v[0, 1], 1e-30)), "ratio_err": float(np.subtract(*np.percentile(rat[np.isfinite(rat) & (v[1:, 1] > 0)], [84, 16])) / 2)}
    return out


def s_at_b(curve, B):
    cB, cS, cN = curve
    k = np.searchsorted(cB, B, side="right") - 1
    return float(cS[k]) if k >= 0 else 0.0, int(cN[k]) if k >= 0 else 0


def sculpt(d):
    q = ~d["is_signal"].astype(bool) & (d["asym"] < 0.1)
    m, w, l = d["mavg"][q], d["w"][q], d["logit"][q]
    h0, _ = np.histogram(m, BINS, weights=w)
    o = np.argsort(-l); cw = np.cumsum(w[o]) / w.sum()
    out = {}
    x = np.log(np.sqrt(BINS[:-1] * BINS[1:]))
    for eff in (0.01, 0.001):
        thr = l[o][min(np.searchsorted(cw, eff), len(o) - 1)]
        c = l >= thr
        h, _ = np.histogram(m[c], BINS, weights=w[c])
        n, _ = np.histogram(m[c], BINS)
        r = h / np.maximum(h0, 1e-30) / eff
        good = n >= 5
        y = np.log(np.maximum(r, 1e-6))
        err = 1 / np.sqrt(np.maximum(n, 1))
        if good.sum() >= 3:
            A = np.vstack([x[good], np.ones(good.sum())] + ([x[good] ** 2] if good.sum() >= 5 else [])).T
            coef, *_ = np.linalg.lstsq(A / err[good, None], y[good] / err[good], rcond=None)
            res = (y[good] - A @ coef) / err[good]
            out[f"{eff:g}"] = {"eff_ratio": [round(float(v), 3) for v in r], "loglog_slope": float(np.polyfit(x[good], y[good], 1)[0]),
                               "max_abs_resid_sigma": float(np.abs(res).max()), "raw_events": n.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("runs", nargs="+", help="name=scores.npz (or a train_pairnet eval.npz)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = load(a.ref)
    runs = {"r14A": ref} | {r.split("=")[0]: load(r.split("=")[1]) for r in a.runs}
    masses = sorted(set(np.round(ref["pmass"][ref["is_signal"].astype(bool)]).astype(int)))
    res = {n: {"sculpt": sculpt(d)} for n, d in runs.items()}
    for wn, (lo, hi) in WINDOWS.items():
        print(f"\n=== window {wn} [{lo}, {hi}] M, A < 0.1.  S = signal fraction, B = weighted QCD (arb. units)")
        print(f"{'model':22s} " + " ".join(f"{M:>48d}" for M in masses))
        for n, d in runs.items():
            cells = []
            for M in masses:
                r = scan(d, M, lo, hi)
                rr = scan(ref, M, lo, hi)
                sb, nb = s_at_b(r["_curve"], rr["best"]["B"])
                r["S_at_refB"], r["nB_at_refB"] = sb, nb
                r["fom_ratio"] = r["best"]["fom"] / max(rr["best"]["fom"], 1e-30)
                r["S_at_refB_ratio"] = sb / max(rr["best"]["S"], 1e-30)
                r["B0_ratio"] = r["B0"] / max(rr["B0"], 1e-30)
                r.pop("_curve")
                r["fixedB"] = s_at_fixed_b(d, ref, M, lo, hi)
                res[n].setdefault(wn, {})[str(M)] = r
                fb = r["fixedB"]
                cells.append(f"S0 {r['S0']:.3f} B0x{r['B0_ratio']:4.2f} S@10% x{fb['0.1']['ratio']:4.2f}±{fb['0.1']['ratio_err']:.2f}"
                             f" S@1% x{fb['0.01']['ratio']:4.2f}±{fb['0.01']['ratio_err']:.2f} (nB {fb['0.01']['nB']})")
            print(f"{n:22s} " + " | ".join(cells))
    print("\n=== QCD sculpting (all QCD with A < 0.1): efficiency vs m_avg at global cuts keeping 1 % / 0.1 %")
    for n in runs:
        sc = res[n]["sculpt"]
        print(f"{n:22s} " + "  ".join(f"[{e}] slope {v['loglog_slope']:+.2f} max-resid {v['max_abs_resid_sigma']:.1f}σ"
                                         for e, v in sc.items()))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
