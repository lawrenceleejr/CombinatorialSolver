ASYM_CUT = 0.1  # stand-in for the analysis preselection on mass asymmetry


def summarize(ev):
    s = ev["is_signal"]
    tgt = ev["target"]
    rec = s & (tgt >= 0)
    r = {
        "n_sig": int(s.sum()),
        "n_bkg": int((~s).sum()),
        "frac_reconstructable": float(rec.sum() / max(s.sum(), 1)),
        "eff_on_reconstructable": float((ev["best"][rec] == tgt[rec]).mean()) if rec.any() else 0.0,
    }
    r["eff_on_all_signal"] = r["eff_on_reconstructable"] * r["frac_reconstructable"]
    if "m_est" in ev and s.any():
        rel = ev["m_est"][s] / np.maximum(ev["pmass"][s], 1)
        lo, hi = np.quantile(rel, [0.16, 0.84])
        r["m_est_rel_halfwidth68"] = float((hi - lo) / (hi + lo))
        relb = ev["mavg"][s] / np.maximum(ev["pmass"][s], 1)
        lo, hi = np.quantile(relb, [0.16, 0.84])
        r["mavg_rel_halfwidth68"] = float((hi - lo) / (hi + lo))
    pas = ev["asym"] < ASYM_CUT
    r["sig_pass_asym"] = float(pas[s].mean()) if s.any() else 0.0
    if s.any():
        rel = ev["mavg"][s] / np.maximum(ev["pmass"][s], 1)
        r["sig_pass_asym_in_window"] = float((pas[s] & (rel > 0.85) & (rel < 1.1)).mean())
    if (~s).any():
        w = ev["w"][~s]
        r["bkg_pass_asym"] = float((w * pas[~s]).sum() / w.sum())  # weighted
        r["bkg_mavg_median_passing"] = float(np.median(ev["mavg"][~s & pas])) if (pas & ~s).any() else 0.0
    if (~s).any() and s.any():
        from sklearn.metrics import roc_auc_score

        y = s.astype(int)
        r["auc"] = float(roc_auc_score(y, ev["psig"], sample_weight=None))
        thr = np.quantile(ev["psig"][s], 0.5)
        fb = (ev["psig"][~s] >= thr).mean()
        r["bkg_rej_at_50"] = float(1 / max(fb, 1e-6))
        r["sig_in_window_at_bkg"] = sig_at_bkg_curve(ev)
    return r


def sig_at_bkg_curve(ev, targets=(0.02, 0.05, 0.08)):
    """Signal passing the asymmetry cut with m_avg in [0.85, 1.1] M, read off at
    fixed weighted QCD pass rates by scanning the asymmetry cut."""
    s = ev["is_signal"]
    w = ev["w"][~s]
    rel = ev["mavg"] / np.maximum(ev["pmass"], 1)
    win = (rel > 0.85) & (rel < 1.1)
    cuts = np.linspace(0.002, 0.6, 300)
    q = np.array([(w * (ev["asym"][~s] < c)).sum() / w.sum() for c in cuts])
    sg = np.array([((ev["asym"][s] < c) & win[s]).mean() for c in cuts])
    return {f"{t:g}": float(np.interp(t, q, sg)) for t in targets}


