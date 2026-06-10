"""
Data-driven QCD background estimate via the chi-sideband transfer.

Physics
-------
The bump-hunt observable is the average candidate mass m_avg = (m1+m2)/2 of the
two NN-chosen triplets.  The rapidity separation y* = |y1-y2|/2 (chi = e^{2y*})
splits the data into three regions:

    SR (low y*)  : central — where pair-produced signal lives; bump hunt here.
    VR (mid y*)  : validation — transfer derivation and closure tests.
    CR (high y*) : forward — QCD-enriched template source.

QCD's t-channel angular shape in chi is approximately independent of the mass
scale (QCD scale invariance), so the m_avg shape measured in the CR transfers
to the SR up to a smooth residual drift (PDF / acceptance / reconstruction
effects).  That drift is modelled by a low-order polynomial transfer factor
R(m_avg) = [target region]/[CR], fit in signal-free regions:

    - R_VR  is fit over the full m_avg range in the VR  → closure test;
    - R_SR  is fit in the SR *mass sidebands* only (the blind window around the
      probed mass is excluded) → the SR background prediction.

The residual non-closure in the VR is carried as the background systematic.

Outputs (consumed by ``src.bump_hunt``)
---------------------------------------
``templates.npz`` with the m_avg binning, the predicted SR background template
with statistical + transfer-fit uncertainties, the (blinded) observed SR data,
and the SR signal template.  Also ``closure.json`` (metrics), a multi-panel
``background_estimate.pdf``, and the score-sculpting stability panel.

All inputs come from the ``observables.npz`` written by ``src.analysis``; in
the mock setup the QCD sample doubles as "data" (optionally with injected
signal), and the signal sample provides the contamination report and the SR
signal template.

NOTE: the toy QCD validates the *pipeline*, not the *method* — real validation
of the chi-factorisation needs realistic QCD MC or data sidebands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cut_analysis import (
    DEFAULT_SR_QUANTILE,
    DEFAULT_VR_QUANTILE,
    Observables,
    region_boundaries,
    region_masks,
)

REGIONS = ("SR", "VR", "CR")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_observables(npz_path: str, tag: str) -> Observables:
    """Rebuild an :class:`Observables` from the npz written by ``src.analysis``."""
    d = np.load(npz_path)
    kw = {}
    for k in ("avg_mass", "mass_asym", "delta_phi", "avg_boost",
              "dalitz_edge", "dalitz_corner", "y_star", "chi"):
        key = f"{tag}_{k}"
        if key in d:
            kw[k] = d[key]
    if f"{tag}_weight" in d:
        kw["weight"] = d[f"{tag}_weight"]
    if f"{tag}_score" in d:
        kw["score"] = d[f"{tag}_score"]
    if "y_star" not in kw:
        raise ValueError(
            f"{npz_path} has no {tag}_y_star — re-run src.analysis (it now "
            f"computes y*/chi)."
        )
    return Observables(**kw)


def concat_observables(a: Observables, b: Observables, b_scale: float = 1.0) -> Observables:
    """Concatenate two samples (e.g. background + injected signal scaled by mu)."""
    def _cat(x, y):
        if x is None or y is None:
            return None
        return np.concatenate([np.asarray(x), np.asarray(y)])

    return Observables(
        avg_mass=_cat(a.avg_mass, b.avg_mass),
        mass_asym=_cat(a.mass_asym, b.mass_asym),
        delta_phi=_cat(a.delta_phi, b.delta_phi),
        avg_boost=_cat(a.avg_boost, b.avg_boost),
        dalitz_edge=_cat(a.dalitz_edge, b.dalitz_edge),
        dalitz_corner=_cat(a.dalitz_corner, b.dalitz_corner),
        weight=np.concatenate([a.weights, b.weights * b_scale]),
        y_star=_cat(a.y_star, b.y_star),
        chi=_cat(a.chi, b.chi),
        score=_cat(a.score, b.score),
    )


# --------------------------------------------------------------------------- #
# Histogramming + transfer fit
# --------------------------------------------------------------------------- #
def whist(x: np.ndarray, w: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted histogram and its per-bin variance (sum of w^2)."""
    h, _ = np.histogram(x, bins=edges, weights=w)
    s2, _ = np.histogram(x, bins=edges, weights=w * w)
    return h, s2


def fit_transfer_factor(
    num: np.ndarray, num_var: np.ndarray,
    den: np.ndarray, den_var: np.ndarray,
    centers: np.ndarray,
    degree: int = 1,
    use_bins: np.ndarray | None = None,
) -> dict:
    """Weighted polynomial fit of the per-bin ratio R = num/den.

    Bins with empty denominator (or excluded by ``use_bins``) are skipped.
    Returns the coefficients, covariance, and an evaluator with a 1-sigma band.
    """
    good = (den > 0) & (num >= 0)
    if use_bins is not None:
        good &= use_bins
    n_good = int(good.sum())
    if n_good < degree + 1:
        raise ValueError(
            f"Only {n_good} usable bins for a degree-{degree} transfer fit; "
            f"coarsen the binning or lower the degree."
        )
    r = num[good] / den[good]
    # Ratio error propagation (independent Poisson-ish numerator/denominator).
    r_var = np.where(
        num[good] > 0,
        r ** 2 * (num_var[good] / np.clip(num[good], 1e-12, None) ** 2
                  + den_var[good] / den[good] ** 2),
        den_var[good] / den[good] ** 2,
    )
    r_sig = np.sqrt(np.clip(r_var, 1e-12, None))

    # Centre/scale the mass axis for numerical stability of the polynomial.
    x0, xs = centers.mean(), max(centers.std(), 1e-9)
    z = (centers - x0) / xs
    coeffs, cov = np.polyfit(z[good], r, degree, w=1.0 / r_sig, cov="unscaled")

    def evaluate(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        zz = (c - x0) / xs
        val = np.polyval(coeffs, zz)
        # Jacobian wrt coefficients: [z^d, ..., z, 1]
        J = np.vander(zz, degree + 1)
        var = np.einsum("ij,jk,ik->i", J, cov, J)
        return val, np.sqrt(np.clip(var, 0.0, None))

    # Fit quality on the used bins.
    r_fit, _ = evaluate(centers[good])
    chi2 = float((((r - r_fit) / r_sig) ** 2).sum())
    ndf = max(n_good - (degree + 1), 1)

    return {
        "coeffs": coeffs, "cov": cov, "x0": x0, "xs": xs,
        "evaluate": evaluate, "chi2": chi2, "ndf": ndf,
        "r": r, "r_sig": r_sig, "good": good,
    }


def closure_metrics(obs_h: np.ndarray, obs_var: np.ndarray,
                    pred_h: np.ndarray, pred_var: np.ndarray) -> dict:
    """Predicted-vs-observed comparison: per-bin pulls, chi2/ndf, max |pull|."""
    tot_var = obs_var + pred_var
    good = tot_var > 0
    pulls = np.zeros_like(obs_h)
    pulls[good] = (obs_h[good] - pred_h[good]) / np.sqrt(tot_var[good])
    chi2 = float((pulls[good] ** 2).sum())
    ndf = max(int(good.sum()), 1)
    return {
        "chi2": chi2, "ndf": ndf, "chi2_ndf": chi2 / ndf,
        "max_abs_pull": float(np.abs(pulls).max()) if len(pulls) else 0.0,
        "pulls": pulls,
    }


# --------------------------------------------------------------------------- #
# Main estimate
# --------------------------------------------------------------------------- #
def run_background_estimate(
    observables_npz: str,
    output_dir: str = "results/bkg_estimate",
    sr_quantile: float = DEFAULT_SR_QUANTILE,
    vr_quantile: float = DEFAULT_VR_QUANTILE,
    sr_cuts: dict | None = None,
    score_min: float | None = None,
    score_quantile: float | None = None,
    n_bins: int = 25,
    mass_range: tuple[float, float] | None = None,
    blind_window: tuple[float, float] | None = None,
    poly_degree: int = 1,
    inject_mu: float = 0.0,
) -> dict:
    """Run the chi-sideband background estimate + closure suite.

    The ``bkg`` arrays in *observables_npz* play the role of data (the toy
    stand-in); ``--inject-mu`` optionally adds the signal sample scaled by mu to
    the pseudo-data so the downstream bump hunt has something to find.  The
    signal arrays provide the SR signal template and the contamination report.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sig = load_observables(observables_npz, "sig")
    bkg = load_observables(observables_npz, "bkg")
    data = bkg if inject_mu <= 0 else concat_observables(bkg, sig, b_scale=inject_mu)

    if score_min is not None and data.score is None:
        print("  Warning: --score-min requested but no NN score in the inputs; "
              "ignoring the score cut.")
        score_min = None
    # A quantile-based score cut ("keep the top X% most signal-like data") is
    # robust to the score's calibration, which depends on the training class
    # balance; an absolute threshold is not.
    if score_quantile is not None:
        if data.score is None:
            print("  Warning: --score-quantile requested but no NN score in the "
                  "inputs; ignoring.")
        else:
            score_min = float(np.percentile(data.score, 100.0 * score_quantile))
            print(f"Score cut from data quantile {score_quantile:.2f}: "
                  f"score >= {score_min:.4f}")

    # Region boundaries from the *data* y* (background-like sample) so the
    # region populations are controlled; same boundaries for every sample.
    b1, b2 = region_boundaries(data.y_star, data.weights,
                               sr_quantile=sr_quantile, vr_quantile=vr_quantile)
    print(f"y* boundaries: SR < {b1:.3f} <= VR < {b2:.3f} <= CR "
          f"(quantiles {sr_quantile:.2f}/{vr_quantile:.2f})")

    masks_data = region_masks(data, (b1, b2), sr_cuts=sr_cuts, score_min=score_min)
    masks_sig = region_masks(sig, (b1, b2), sr_cuts=sr_cuts, score_min=score_min)

    # Common m_avg binning.
    sel_all = masks_data["SR"] | masks_data["VR"] | masks_data["CR"]
    if not sel_all.any():
        raise ValueError(
            "No data events survive the region selection — the SR cuts and/or "
            "the --score-min threshold removed everything. Loosen the score cut "
            "(check the score distribution in observables.npz; an untrained "
            "score head clusters near a constant) or the kinematic cuts."
        )
    if mass_range is None:
        m = data.avg_mass[sel_all]
        mass_range = (float(np.percentile(m, 0.5)), float(np.percentile(m, 99.5)))
    edges = np.linspace(*mass_range, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hists = {}
    for region in REGIONS:
        sel = masks_data[region]
        hists[region] = whist(data.avg_mass[sel], data.weights[sel], edges)
    h_sig_sr, v_sig_sr = whist(sig.avg_mass[masks_sig["SR"]],
                               sig.weights[masks_sig["SR"]], edges)

    yields = {r: float(hists[r][0].sum()) for r in REGIONS}
    print(f"Data yields: SR={yields['SR']:.1f}  VR={yields['VR']:.1f}  CR={yields['CR']:.1f}")

    # --- Closure: predict the VR from the CR -------------------------------
    h_cr, v_cr = hists["CR"]
    h_vr, v_vr = hists["VR"]
    h_sr, v_sr = hists["SR"]

    fit_vr = fit_transfer_factor(h_vr, v_vr, h_cr, v_cr, centers, degree=poly_degree)
    r_vr, r_vr_sig = fit_vr["evaluate"](centers)
    pred_vr = h_cr * r_vr
    pred_vr_var = v_cr * r_vr ** 2 + (h_cr * r_vr_sig) ** 2
    closure_vr = closure_metrics(h_vr, v_vr, pred_vr, pred_vr_var)
    print(f"VR closure: chi2/ndf = {closure_vr['chi2']:.1f}/{closure_vr['ndf']} "
          f"= {closure_vr['chi2_ndf']:.2f}, max |pull| = {closure_vr['max_abs_pull']:.2f}")
    if closure_vr["chi2_ndf"] > 2.0:
        print(f"  -> VR closure is poor: the m_avg-vs-y* drift is not captured "
              f"by a degree-{poly_degree} transfer factor. Try "
              f"--poly-degree {poly_degree + 1} (use the lowest degree that "
              f"closes; the residual is your systematic either way).")

    # Non-closure systematic: yield-weighted rms of the VR relative residuals,
    # with the expected statistical scatter subtracted in quadrature so pure
    # Poisson fluctuation is not double-counted as a systematic.  Weighting by
    # the predicted yield keeps near-empty tail bins (whose relative residuals
    # are huge but irrelevant) from dominating the number.
    good_nc = pred_vr > 0
    if good_nc.any():
        rel_resid = (h_vr[good_nc] - pred_vr[good_nc]) / pred_vr[good_nc]
        rel_stat = np.sqrt(v_vr[good_nc] + pred_vr_var[good_nc]) / pred_vr[good_nc]
        w_nc = pred_vr[good_nc]
        rms2 = float(np.sum(w_nc * rel_resid ** 2) / w_nc.sum())
        stat2 = float(np.sum(w_nc * rel_stat ** 2) / w_nc.sum())
        nonclosure_rel = float(np.sqrt(max(rms2 - stat2, 0.0)))
    else:
        nonclosure_rel = 0.0
    print(f"Non-closure systematic (stat-subtracted, yield-weighted rms of VR "
          f"residuals): {100*nonclosure_rel:.1f}%")

    # --- SR prediction: transfer fit in the SR mass sidebands --------------
    use_bins = np.ones(n_bins, dtype=bool)
    if blind_window is not None:
        lo, hi = blind_window
        use_bins &= ~((centers >= lo) & (centers <= hi))
        print(f"Blind window: m_avg in [{lo:.0f}, {hi:.0f}] excluded from the SR fit")
    fit_sr = fit_transfer_factor(h_sr, v_sr, h_cr, v_cr, centers,
                                 degree=poly_degree, use_bins=use_bins)
    r_sr, r_sr_sig = fit_sr["evaluate"](centers)
    pred_sr = np.clip(h_cr * r_sr, 0.0, None)
    pred_sr_stat = np.sqrt(np.clip(v_cr, 0.0, None)) * np.abs(r_sr)   # CR statistics
    pred_sr_fit = h_cr * r_sr_sig                                     # transfer-fit unc

    # CR-empty bins: zero observed CR events does NOT mean zero background —
    # it means the prediction is bounded by the Poisson fluctuation of an
    # empty bin transferred to the SR.  Encode "0 (+1.84) CR events x R":
    # a small nominal with the 68% CL upper (Garwood) as the uncertainty, so
    # a couple of SR events in the sparse tail cannot fake a large excess.
    cr_empty = (h_cr <= 0)
    if cr_empty.any():
        r_pos = np.clip(r_sr, 0.0, None)
        pred_sr[cr_empty] = 0.3 * r_pos[cr_empty]
        pred_sr_stat[cr_empty] = 1.84 * np.clip(r_pos[cr_empty], 1e-3, None)

    pred_sr_nonclosure = pred_sr * nonclosure_rel                     # VR non-closure

    # Sideband closure in the SR (the unblinded bins).
    closure_sr_sideband = closure_metrics(
        h_sr[use_bins], v_sr[use_bins], pred_sr[use_bins],
        (pred_sr_stat ** 2 + pred_sr_fit ** 2)[use_bins],
    )
    print(f"SR sideband closure: chi2/ndf = {closure_sr_sideband['chi2_ndf']:.2f}, "
          f"max |pull| = {closure_sr_sideband['max_abs_pull']:.2f}")

    # --- Signal contamination in CR / VR ------------------------------------
    sig_tot = float(sig.weights[masks_sig["SR"] | masks_sig["VR"] | masks_sig["CR"]].sum())
    contamination = {}
    for region in ("VR", "CR"):
        s_in = float(sig.weights[masks_sig[region]].sum())
        contamination[region] = {
            "sig_fraction": s_in / sig_tot if sig_tot else 0.0,
            "sig_over_data": s_in / yields[region] if yields[region] else 0.0,
        }
        print(f"Signal contamination {region}: {100*contamination[region]['sig_fraction']:.1f}% "
              f"of selected signal; sig/data = {contamination[region]['sig_over_data']:.3f} "
              f"(at mu=1 normalisation)")

    # --- Persist templates for src.bump_hunt --------------------------------
    np.savez(
        out / "templates.npz",
        edges=edges, centers=centers,
        boundaries=np.array([b1, b2]),
        data_sr=h_sr, data_sr_var=v_sr,
        data_vr=h_vr, data_vr_var=v_vr,
        data_cr=h_cr, data_cr_var=v_cr,
        bkg_sr=pred_sr,
        bkg_sr_stat=pred_sr_stat,
        bkg_sr_fit=pred_sr_fit,
        bkg_sr_nonclosure=pred_sr_nonclosure,
        sig_sr=h_sig_sr, sig_sr_var=v_sig_sr,
        blind_window=np.array(blind_window if blind_window else (np.nan, np.nan)),
    )

    metrics = {
        "boundaries": [b1, b2],
        "yields": yields,
        "vr_closure": {k: v for k, v in closure_vr.items() if k != "pulls"},
        "sr_sideband_closure": {k: v for k, v in closure_sr_sideband.items() if k != "pulls"},
        "nonclosure_rel": nonclosure_rel,
        "transfer_fit_vr": {"chi2": fit_vr["chi2"], "ndf": fit_vr["ndf"],
                            "coeffs": fit_vr["coeffs"].tolist()},
        "transfer_fit_sr": {"chi2": fit_sr["chi2"], "ndf": fit_sr["ndf"],
                            "coeffs": fit_sr["coeffs"].tolist()},
        "contamination": contamination,
        "inject_mu": inject_mu,
        "note": ("Toy QCD validates the pipeline, not the method: the chi "
                 "factorisation must be re-validated on realistic QCD MC or "
                 "data sidebands."),
    }
    with open(out / "closure.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Plots ---------------------------------------------------------------
    fig_path = _make_estimate_figure(
        out / "background_estimate.pdf",
        data=data, sig=sig, masks_data=masks_data,
        boundaries=(b1, b2), edges=edges, centers=centers,
        hists=hists, pred_vr=pred_vr, pred_vr_var=pred_vr_var,
        pred_sr=pred_sr,
        pred_sr_err=np.sqrt(pred_sr_stat**2 + pred_sr_fit**2 + pred_sr_nonclosure**2),
        closure_vr=closure_vr, fit_vr=fit_vr, blind_window=blind_window,
        score_min=score_min,
    )

    print(f"\nResults written to {out}/")
    print("  templates.npz            : binned templates + uncertainties (input to src.bump_hunt)")
    print("  closure.json             : closure metrics, contamination, fit quality")
    if fig_path:
        print("  background_estimate.pdf  : regions, transfer fit, closure, sculpting")
    print("\nNOTE: toy QCD closes by construction — re-validate the chi "
          "factorisation on realistic QCD MC / data sidebands.")
    return metrics


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _make_estimate_figure(out_path, *, data, sig, masks_data, boundaries,
                          edges, centers, hists, pred_vr, pred_vr_var,
                          pred_sr, pred_sr_err, closure_vr, fit_vr,
                          blind_window, score_min):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plot_style import HIST_COLORS, init_style, rgba, style_axis
        init_style(plt)
    except ImportError:
        print("  Warning: matplotlib not available; skipping estimate figure.")
        return None

    b1, b2 = boundaries
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # (0,0) Region layout: y* vs m_avg for data, signal contours overlaid.
    ax = axes[0, 0]
    ax.hexbin(data.avg_mass, data.y_star, gridsize=40, cmap="Greys", mincnt=1)
    ax.scatter(sig.avg_mass[::5], sig.y_star[::5], s=2.5,
               color=rgba(HIST_COLORS["signal"], 0.35), label="signal (MC)")
    for b, name in ((b1, "SR | VR"), (b2, "VR | CR")):
        ax.axhline(b, color=HIST_COLORS["qcd"], linewidth=1.4, linestyle="--")
        ax.text(ax.get_xlim()[1], b, f" {name}", va="center", fontsize=9,
                color=HIST_COLORS["qcd"])
    ax.set_xlabel("Average candidate mass [GeV]")
    ax.set_ylabel(r"$y^*$")
    ax.set_title("Analysis regions (grey = data, green = signal)", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    style_axis(ax)

    # (0,1) CR/VR/SR m_avg shapes (area-normalised) — the factorisation test.
    ax = axes[0, 1]
    for region, color in (("CR", HIST_COLORS["qcd"]),
                          ("VR", HIST_COLORS["accent2"]),
                          ("SR", HIST_COLORS["accent"])):
        h, _ = hists[region]
        tot = h.sum()
        if tot > 0:
            ax.stairs(h / tot, edges, color=color, linewidth=1.6, label=region)
    ax.set_xlabel("Average candidate mass [GeV]")
    ax.set_ylabel("Normalised events")
    ax.set_title("m$_{avg}$ shape by region (factorisation check)", loc="left")
    ax.legend(frameon=False)
    style_axis(ax)

    # (1,0) VR closure: observed vs predicted + pulls inset.
    ax = axes[1, 0]
    h_vr, v_vr = hists["VR"]
    ax.errorbar(centers, h_vr, yerr=np.sqrt(np.clip(v_vr, 0, None)), fmt="o",
                markersize=3.5, color="#2A2A2A", label="VR observed")
    ax.stairs(pred_vr, edges, color=HIST_COLORS["qcd"], linewidth=1.6,
              label="CR x R(m) prediction")
    ax.fill_between(centers, pred_vr - np.sqrt(pred_vr_var),
                    pred_vr + np.sqrt(pred_vr_var), step="mid",
                    facecolor=rgba(HIST_COLORS["qcd"], 0.2))
    ax.set_xlabel("Average candidate mass [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(
        f"VR closure  ($\\chi^2$/ndf = {closure_vr['chi2_ndf']:.2f}, "
        f"max |pull| = {closure_vr['max_abs_pull']:.1f})", loc="left")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)

    # (1,1) SR prediction with the blind window, plus sculpting stability when
    # a score is available: normalised data m_avg shape vs score cut.
    ax = axes[1, 1]
    if data.score is not None:
        cuts_to_try = [0.0, 0.3, 0.5, 0.7, 0.9]
        cmap = plt.get_cmap("viridis")
        base_sel = masks_data["SR"]
        for i, smin in enumerate(cuts_to_try):
            sel = base_sel & (data.score >= smin)
            h, _ = whist(data.avg_mass[sel], data.weights[sel], edges)
            if h.sum() > 0:
                ax.stairs(h / h.sum(), edges, color=cmap(i / max(len(cuts_to_try) - 1, 1)),
                          linewidth=1.4, label=f"score > {smin:.1f}")
        ax.set_title("SR data shape vs score cut (sculpting check)", loc="left")
        ax.set_ylabel("Normalised events")
        ax.legend(frameon=False, fontsize=9)
    else:
        h_sr, v_sr = hists["SR"]
        ax.errorbar(centers, h_sr, yerr=np.sqrt(np.clip(v_sr, 0, None)), fmt="o",
                    markersize=3.5, color="#2A2A2A", label="SR observed")
        ax.stairs(pred_sr, edges, color=HIST_COLORS["qcd"], linewidth=1.6,
                  label="predicted background")
        ax.fill_between(centers, pred_sr - pred_sr_err, pred_sr + pred_sr_err,
                        step="mid", facecolor=rgba(HIST_COLORS["qcd"], 0.2))
        if blind_window is not None:
            ax.axvspan(*blind_window, color="#DDDDDD", alpha=0.5, label="blind window")
        ax.set_title("SR prediction", loc="left")
        ax.set_ylabel("Events")
        ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("Average candidate mass [GeV]")
    style_axis(ax)

    fig.suptitle("Chi-sideband QCD background estimate", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(str(out_path))
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _sr_cuts_from_args(args) -> dict:
    return {
        "delta_phi": ("upper", args.dphi_min),
        "mass_asym": ("lower", args.asym_max),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Chi-sideband data-driven QCD background estimate + closure"
    )
    p.add_argument("--observables", type=str, required=True,
                   help="observables.npz from src.analysis")
    p.add_argument("--output", type=str, default="results/bkg_estimate")
    p.add_argument("--sr-quantile", type=float, default=DEFAULT_SR_QUANTILE)
    p.add_argument("--vr-quantile", type=float, default=DEFAULT_VR_QUANTILE)
    p.add_argument("--n-bins", type=int, default=25)
    p.add_argument("--mass-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--blind", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Blind m_avg window excluded from the SR transfer fit")
    p.add_argument("--poly-degree", type=int, default=1,
                   help="Degree of the polynomial transfer factor R(m_avg)")
    p.add_argument("--inject-mu", type=float, default=0.0,
                   help="Add the signal sample scaled by mu to the pseudo-data")
    # SR selection (applied identically to all three regions).
    p.add_argument("--dphi-min", type=float, default=2.5)
    p.add_argument("--asym-max", type=float, default=0.4)
    p.add_argument("--score-min", type=float, default=None,
                   help="NN signal-score cut (requires a trained score head)")
    p.add_argument("--score-quantile", type=float, default=None,
                   help="Cut at this quantile of the DATA score distribution "
                        "(e.g. 0.6 keeps the top 40%% most signal-like events); "
                        "robust to score calibration, overrides --score-min")
    args = p.parse_args()

    run_background_estimate(
        observables_npz=args.observables,
        output_dir=args.output,
        sr_quantile=args.sr_quantile,
        vr_quantile=args.vr_quantile,
        sr_cuts=_sr_cuts_from_args(args),
        score_min=args.score_min,
        score_quantile=args.score_quantile,
        n_bins=args.n_bins,
        mass_range=tuple(args.mass_range) if args.mass_range else None,
        blind_window=tuple(args.blind) if args.blind else None,
        poly_degree=args.poly_degree,
        inject_mu=args.inject_mu,
    )
