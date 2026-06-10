"""
pyhf bump hunt in the average candidate mass.

Consumes the binned templates written by ``src.background_estimate``
(``templates.npz``: SR data, chi-transfer background prediction with
stat/fit/non-closure uncertainties, and the SR signal template) and produces
the statistical results of the search:

  - local discovery p0 / significance scanned across m_avg, with an
    approximate trials-factor (look-elsewhere) corrected global p-value;
  - observed and expected 95% CLs upper limits on the signal strength mu
    (mu = 1 corresponds to the normalisation of the supplied signal sample);
  - a signal-injection test: pseudo-data = background + mu_inj x signal
    (Asimov), refit to verify the fitted mu_hat recovers the injection.

Model per mass point (single-channel HistFactory):

  signal      : SR signal template, POI ``mu`` (normfactor).
                At the MC mass point the MC template is used; elsewhere a
                Gaussian surrogate with the MC yield and relative resolution.
  background  : chi-transfer prediction with
                  - per-bin ``shapesys`` = stat (+) transfer-fit uncertainty,
                  - ``normsys``         = VR non-closure (correlated).

Example
-------
    python -m src.bump_hunt --templates results/bkg_estimate/templates.npz \
        --output results/bump_hunt_stats --inject-mu 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Template loading and signal surrogates
# --------------------------------------------------------------------------- #
def load_templates(path: str) -> dict:
    d = np.load(path)
    t = {k: d[k] for k in d.files}
    t["centers"] = np.asarray(t["centers"], dtype=float)
    t["edges"] = np.asarray(t["edges"], dtype=float)
    return t


def signal_peak_properties(centers: np.ndarray, sig: np.ndarray) -> tuple[float, float]:
    """(peak mean, width) of the MC signal template, from the bins within
    +-30% of the maximum bin (so combinatorial tails don't inflate the width)."""
    if sig.sum() <= 0:
        return float(centers.mean()), 0.1 * float(centers.mean())
    peak = centers[int(np.argmax(sig))]
    sel = np.abs(centers - peak) < 0.3 * peak
    w = sig[sel]
    c = centers[sel]
    mean = float(np.sum(w * c) / w.sum())
    var = float(np.sum(w * (c - mean) ** 2) / w.sum())
    return mean, max(np.sqrt(var), (centers[1] - centers[0]))


def make_signal_template(
    centers: np.ndarray, edges: np.ndarray,
    mass: float, yield_total: float, rel_width: float,
    mc_template: np.ndarray | None = None, mc_mass: float | None = None,
) -> np.ndarray:
    """Signal template at *mass*: the MC template when the scan point matches
    the MC mass (within half a resolution width), else a Gaussian surrogate
    with the same total yield and relative width."""
    width = max(rel_width * mass, (edges[1] - edges[0]) / 2.0)
    if mc_template is not None and mc_mass is not None and abs(mass - mc_mass) < 0.5 * width:
        return np.asarray(mc_template, dtype=float)
    from math import erf, sqrt
    cdf = lambda x: 0.5 * (1.0 + erf((x - mass) / (width * sqrt(2.0))))
    bin_frac = np.array([cdf(edges[i + 1]) - cdf(edges[i]) for i in range(len(centers))])
    return yield_total * bin_frac


# --------------------------------------------------------------------------- #
# pyhf model construction and fits
# --------------------------------------------------------------------------- #
def build_model(sig: np.ndarray, bkg: np.ndarray, bkg_shape_unc: np.ndarray,
                nonclosure_rel: float):
    """Single-channel HistFactory model.

    Only bins with a populated background prediction are kept: a bin with
    signal but no background model is meaningless for the fit (and bins with
    bkg ~ 0 destabilise the shapesys gamma constraints).  The per-bin shape
    uncertainty is floored at 1% of the background so the auxiliary Poisson
    constraint stays numerically sane.
    """
    import pyhf

    keep = (bkg > 0) | (sig > 0)
    sig_k = np.clip(sig[keep], 0.0, None)
    bkg_k = bkg[keep].copy()
    unc_k = np.maximum(bkg_shape_unc[keep], 0.01 * np.clip(bkg_k, 0.0, None))

    # Bins with signal but an empty background prediction (the CR ran out of
    # events there): the search is background-free in those bins.  Floor the
    # prediction at half the smallest populated-bin yield — roughly the
    # "one CR event transferred" scale — with a 100% uncertainty, so the
    # Poisson model stays defined without inventing background.
    empty = bkg_k <= 0
    if empty.any():
        floor = 0.5 * float(bkg_k[~empty].min()) if (~empty).any() else 0.5
        bkg_k[empty] = floor
        unc_k[empty] = floor

    bkg_modifiers = [
        {"name": "bkg_shape", "type": "shapesys", "data": unc_k.tolist()},
    ]
    if nonclosure_rel > 1e-4:
        bkg_modifiers.append({
            "name": "bkg_nonclosure", "type": "normsys",
            "data": {"hi": 1.0 + nonclosure_rel,
                     "lo": max(1.0 - nonclosure_rel, 0.01)},
        })

    spec = {
        "channels": [{
            "name": "SR",
            "samples": [
                {"name": "signal", "data": sig_k.tolist(),
                 "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]},
                {"name": "background", "data": bkg_k.tolist(),
                 "modifiers": bkg_modifiers},
            ],
        }],
    }
    model = pyhf.Model(spec, poi_name="mu")
    return model, keep


def _with_backend_fallback(fn):
    """Run *fn* under minuit, retrying with the scipy optimizer on a failed
    minimisation (each backend fails on different corners of these gamma-
    constrained likelihoods).  Returns None if both fail."""
    import pyhf

    for backend in ("minuit", None):
        try:
            if backend == "minuit":
                try:
                    pyhf.set_backend("numpy", "minuit")
                except Exception:
                    continue
            else:
                pyhf.set_backend("numpy")
            return fn()
        except pyhf.exceptions.FailedMinimization:
            continue
        except Exception:
            continue
    return None


def discovery_p0(model, data) -> tuple[float, float]:
    """Background-only discovery test: (p0, significance in sigma)."""
    import pyhf
    from scipy.stats import norm

    result = _with_backend_fallback(
        lambda: float(pyhf.infer.hypotest(0.0, data, model, test_stat="q0"))
    )
    if result is None:
        return float("nan"), float("nan")
    p0 = result
    z = float(norm.isf(min(max(p0, 1e-300), 1.0 - 1e-12)))
    return p0, z


def upper_limit_cls(model, data, mu_hint: float) -> tuple[float, np.ndarray]:
    """Observed and expected (median +-1/2 sigma) 95% CLs upper limits on mu.

    The scan window is grown adaptively until the observed limit is interior.
    """
    from pyhf.infer.intervals.upper_limits import upper_limit

    hi = max(mu_hint, 1e-6)
    obs, exp = float("nan"), np.full(5, np.nan)
    for _ in range(8):
        scan = np.linspace(0.0, hi, 41)
        result = _with_backend_fallback(lambda: upper_limit(data, model, scan, level=0.05))
        if result is None:
            return float("nan"), np.full(5, np.nan)
        obs, exp = result
        if float(obs) < 0.95 * hi:
            return float(obs), np.asarray(exp, dtype=float)
        hi *= 4.0
    return float(obs), np.asarray(exp, dtype=float)


def fit_mu(model, data) -> tuple[float, float]:
    """MLE mu_hat with a 1-sigma uncertainty from the profile likelihood
    (2 Delta lnL = 1), robust with the default scipy backend."""
    import pyhf
    from scipy.optimize import brentq

    poi_idx = model.config.poi_index
    fit_result = _with_backend_fallback(
        lambda: pyhf.infer.mle.fit(data, model, return_fitted_val=True)
    )
    if fit_result is None:
        return float("nan"), float("nan")
    bestfit, nll_min = fit_result
    mu_hat = float(bestfit[poi_idx])
    nll_min = float(nll_min)

    def dnll(mu):
        _, nll = pyhf.infer.mle.fixed_poi_fit(mu, data, model, return_fitted_val=True)
        return 2.0 * (float(nll) - nll_min) - 1.0

    poi_lo, poi_hi = model.config.suggested_bounds()[poi_idx]
    step = max(abs(mu_hat), 1e-3)
    try:
        hi = mu_hat + step
        while dnll(hi) < 0 and hi < poi_hi:
            hi = mu_hat + (hi - mu_hat) * 2.0
        err_up = brentq(dnll, mu_hat, min(hi, poi_hi), xtol=1e-4 * step) - mu_hat
    except Exception:
        err_up = float("nan")
    try:
        lo = max(mu_hat - step, poi_lo)
        while dnll(lo) < 0 and lo > poi_lo:
            lo = max(mu_hat - (mu_hat - lo) * 2.0, poi_lo)
        err_dn = mu_hat - brentq(dnll, max(lo, poi_lo), mu_hat, xtol=1e-4 * step)
    except Exception:
        err_dn = float("nan")
    err = float(np.nanmean([err_up, err_dn]))
    return mu_hat, err


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_bump_hunt(
    templates_path: str,
    output_dir: str = "results/bump_hunt_stats",
    scan_masses: np.ndarray | None = None,
    inject_mu: float = 0.0,
    skip_limits: bool = False,
) -> dict:
    import pyhf
    # Minuit is far more robust than the default SLSQP for the gamma-constrained
    # likelihoods this model produces (sparse tail bins with O(100%) shapesys).
    try:
        pyhf.set_backend("numpy", "minuit")
    except Exception:
        pyhf.set_backend("numpy")
        print("  Warning: iminuit unavailable; falling back to scipy optimizer "
              "(fits may be fragile).")

    t = load_templates(templates_path)
    centers, edges = t["centers"], t["edges"]
    bkg = np.asarray(t["bkg_sr"], dtype=float)
    bkg_shape_unc = np.sqrt(t["bkg_sr_stat"] ** 2 + t["bkg_sr_fit"] ** 2)
    # Correlated non-closure as a relative normsys.
    nc_rel = float(np.median(np.where(bkg > 0, t["bkg_sr_nonclosure"] / np.clip(bkg, 1e-9, None), 0.0)))
    sig_mc = np.asarray(t["sig_sr"], dtype=float)
    data_sr = np.asarray(t["data_sr"], dtype=float)

    mc_mass, mc_width = signal_peak_properties(centers, sig_mc)
    rel_width = mc_width / mc_mass
    sig_yield = float(sig_mc.sum())
    print(f"MC signal template: peak = {mc_mass:.0f} GeV, width = {mc_width:.0f} GeV "
          f"({100*rel_width:.1f}%), SR yield at mu=1: {sig_yield:.1f}")
    print(f"Background: {bkg.sum():.1f} events; non-closure normsys: {100*nc_rel:.1f}%")

    if scan_masses is None:
        lo = centers[2]
        hi = centers[-3]
        scan_masses = np.linspace(lo, hi, 9)

    # Observed data: pseudo-data = chi-transfer background prediction plus an
    # optional injected signal.  (With real, unblinded data this would be the
    # observed SR histogram, kept here as t['data_sr'].)
    if inject_mu > 0:
        observed = bkg + inject_mu * sig_mc
        print(f"Pseudo-data: background prediction + {inject_mu} x signal (Asimov injection)")
    else:
        observed = data_sr
        print("Observed data: SR histogram from templates.npz")

    results = []
    for mass in scan_masses:
        sig_m = make_signal_template(centers, edges, float(mass), sig_yield,
                                     rel_width, mc_template=sig_mc, mc_mass=mc_mass)
        model, keep = build_model(sig_m, bkg, bkg_shape_unc, nc_rel)
        data = list(np.asarray(observed)[keep]) + model.config.auxdata

        p0, z = discovery_p0(model, data)
        entry = {"mass": float(mass), "p0": p0, "z": z}

        if not skip_limits:
            naive = (2.0 + 1.64 * np.sqrt(max(bkg.sum(), 1.0))) / max(sig_yield, 1e-9)
            obs_lim, exp_lim = upper_limit_cls(model, data, mu_hint=5.0 * naive)
            entry["limit_obs"] = obs_lim
            entry["limit_exp"] = exp_lim.tolist()

        results.append(entry)
        lim_str = (f" | UL(mu) obs={entry['limit_obs']:.3g} "
                   f"exp={entry['limit_exp'][2]:.3g}" if not skip_limits else "")
        print(f"  m = {mass:7.1f} GeV : p0 = {p0:.3g} (Z = {z:5.2f} sigma){lim_str}")

    # Approximate look-elsewhere (trials-factor) correction for the largest
    # local significance: N_indep ~ scanned range / (2 x resolution).
    zs = np.array([r["z"] for r in results])
    p0s = np.array([r["p0"] for r in results])
    i_best = int(np.nanargmax(np.where(np.isfinite(zs), zs, -np.inf)))
    span = float(scan_masses[-1] - scan_masses[0])
    n_indep = max(span / (2.0 * rel_width * float(scan_masses[i_best])), 1.0)
    p_global = float(1.0 - (1.0 - p0s[i_best]) ** n_indep)
    from scipy.stats import norm
    z_global = float(norm.isf(min(max(p_global, 1e-300), 1.0 - 1e-12)))
    print(f"\nLargest excess: m = {results[i_best]['mass']:.0f} GeV, "
          f"local Z = {zs[i_best]:.2f}; trials factor ~{n_indep:.1f} -> "
          f"global p = {p_global:.3g} (Z = {z_global:.2f})")

    # Signal-injection recovery at the MC mass.
    injection = None
    if inject_mu > 0:
        sig_m = make_signal_template(centers, edges, mc_mass, sig_yield,
                                     rel_width, mc_template=sig_mc, mc_mass=mc_mass)
        model, keep = build_model(sig_m, bkg, bkg_shape_unc, nc_rel)
        data = list(np.asarray(observed)[keep]) + model.config.auxdata
        mu_hat, mu_err = fit_mu(model, data)
        pull = (mu_hat - inject_mu) / mu_err if mu_err and np.isfinite(mu_err) else float("nan")
        injection = {"mu_injected": inject_mu, "mu_hat": mu_hat,
                     "mu_err": mu_err, "pull": pull}
        print(f"Injection test @ {mc_mass:.0f} GeV: mu_hat = {mu_hat:.3f} +- {mu_err:.3f} "
              f"(injected {inject_mu}) -> pull = {pull:+.2f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "scan": results,
        "best": {"mass": results[i_best]["mass"], "z_local": float(zs[i_best]),
                 "p_global": p_global, "z_global": z_global,
                 "n_indep_windows": n_indep},
        "signal_mc": {"mass": mc_mass, "rel_width": rel_width, "sr_yield_mu1": sig_yield},
        "injection": injection,
        "note": "mu = 1 corresponds to the normalisation of the supplied signal sample",
    }
    with open(out / "bump_hunt.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig = _make_stats_figure(out / "bump_hunt_stats.pdf", results, centers, edges,
                             observed, bkg, bkg_shape_unc, t["bkg_sr_nonclosure"],
                             sig_mc, inject_mu, t.get("blind_window"))
    print(f"\nResults written to {out}/")
    print("  bump_hunt.json      : p0 scan, limits, injection test")
    if fig:
        print("  bump_hunt_stats.pdf : p0 scan, Brazil limits, SR spectrum")
    return summary


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def _make_stats_figure(out_path, results, centers, edges, observed, bkg,
                       bkg_shape_unc, bkg_nonclosure, sig_mc, inject_mu,
                       blind_window=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plot_style import HIST_COLORS, init_style, rgba, style_axis
        init_style(plt)
    except ImportError:
        return None
    from scipy.stats import norm

    masses = np.array([r["mass"] for r in results])
    p0s = np.array([r["p0"] for r in results])
    has_lims = "limit_obs" in results[0]

    ncols = 3 if has_lims else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 4.8))

    # p0 scan.
    ax = axes[0]
    ax.plot(masses, p0s, marker="o", markersize=4, color=HIST_COLORS["accent"],
            markeredgecolor="white", markeredgewidth=0.6)
    ax.set_yscale("log")
    for z in (1, 2, 3, 4, 5):
        p = norm.sf(z)
        if p > min(p0s.min() / 10.0, 1e-7):
            ax.axhline(p, color="#BBBBBB", linewidth=0.8, linestyle=":")
            ax.text(masses[-1], p, f" {z}$\\sigma$", va="center", fontsize=9, color="#888888")
    ax.set_xlabel("Probed mass [GeV]")
    ax.set_ylabel("Local $p_0$")
    ax.set_title("Discovery p-value scan", loc="left")
    style_axis(ax)

    # Brazil limit plot.
    if has_lims:
        ax = axes[1]
        exp = np.array([r["limit_exp"] for r in results])   # (n, 5)
        obs = np.array([r["limit_obs"] for r in results])
        ax.fill_between(masses, exp[:, 0], exp[:, 4], facecolor="#F4E04D", label=r"expected $\pm2\sigma$")
        ax.fill_between(masses, exp[:, 1], exp[:, 3], facecolor="#7BC96F", label=r"expected $\pm1\sigma$")
        ax.plot(masses, exp[:, 2], "--", color="#2A2A2A", linewidth=1.4, label="expected median")
        ax.plot(masses, obs, color="#2A2A2A", marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.6, linewidth=1.6, label="observed")
        ax.set_yscale("log")
        ax.set_xlabel("Probed mass [GeV]")
        ax.set_ylabel(r"95% CL$_s$ upper limit on $\mu$")
        ax.set_title("Upper limits", loc="left")
        ax.legend(frameon=False, fontsize=9)
        style_axis(ax)

    # SR spectrum: data vs background, signal overlaid.
    ax = axes[-1]
    width = edges[1] - edges[0]
    tot_unc = np.sqrt(bkg_shape_unc ** 2 + np.asarray(bkg_nonclosure) ** 2)
    ax.bar(centers, bkg, width=width, linewidth=0,
           facecolor=rgba(HIST_COLORS["qcd"], 0.4), label="chi-transfer bkg")
    ax.fill_between(centers, bkg - tot_unc, bkg + tot_unc, step="mid",
                    facecolor=rgba(HIST_COLORS["qcd"], 0.25), label="bkg uncertainty")
    ax.errorbar(centers, observed, yerr=np.sqrt(np.clip(observed, 0, None)), fmt="o",
                markersize=3.5, color="#2A2A2A",
                label="pseudo-data" if inject_mu > 0 else "data")
    if inject_mu > 0:
        ax.stairs(inject_mu * sig_mc, edges, color=HIST_COLORS["signal"], linewidth=1.6,
                  label=f"{inject_mu} x signal")
    if blind_window is not None and np.isfinite(np.asarray(blind_window)).all():
        ax.axvspan(*np.asarray(blind_window), color="#DDDDDD", alpha=0.4, label="blind window")
    ax.set_xlabel("Average candidate mass [GeV]")
    ax.set_ylabel("Events")
    ax.set_title("Signal region", loc="left")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax, grid_axis="y")

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="pyhf bump hunt on chi-transfer templates")
    p.add_argument("--templates", type=str, required=True,
                   help="templates.npz from src.background_estimate")
    p.add_argument("--output", type=str, default="results/bump_hunt_stats")
    p.add_argument("--scan", type=float, nargs=3, default=None,
                   metavar=("LO", "HI", "N"),
                   help="Mass scan: low, high, number of points")
    p.add_argument("--inject-mu", type=float, default=0.0,
                   help="Asimov signal injection strength for the recovery test")
    p.add_argument("--skip-limits", action="store_true",
                   help="Skip the CLs limit scan (faster)")
    args = p.parse_args()

    scan = None
    if args.scan is not None:
        scan = np.linspace(args.scan[0], args.scan[1], int(args.scan[2]))

    run_bump_hunt(
        templates_path=args.templates,
        output_dir=args.output,
        scan_masses=scan,
        inject_mu=args.inject_mu,
        skip_limits=args.skip_limits,
    )
