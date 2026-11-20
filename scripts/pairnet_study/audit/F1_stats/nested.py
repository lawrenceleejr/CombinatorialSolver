"""Items 1c, 2a, 2b(part), 3: published best-of vs nested choice, category correlation scan, eval/tune ratios,
QCD composition and weight concentration in the final selections, reach. Reads rerun_*.json."""
import json, sys
import numpy as np
import pipe
from pipe import s_up, mu_up_multi, mu_up_multi_corr
from item1_formula import asimov_multi, asimov_G, asimov_LN
D = pipe.D; L = 300.0
th = json.load(open("/scratch/llee/pairnet/xs_theory/pp13600_gluino_NNLO+NNLL.json"))["data"]
tm = np.array(sorted(int(float(k)) for k in th)); tx = np.array([th[str(m)]["xsec_pb"] * 1e3 for m in tm])
sig_th = lambda M: float(np.exp(np.interp(M, tm, np.log(tx))))
gm = lambda v: float(np.exp(np.mean(np.log(v))))
RHO = np.linspace(0, 1, 11)


def rho_scan(S, B, f):
    """Asimov limit (no 3-event floor) with per-category uncertainty f and correlation rho between categories"""
    return [asimov_multi(S, B, f * np.sqrt(r), f * np.sqrt(1 - r)) for r in RHO]


def reach(cur):
    M = np.array(sorted(m for m in cur if np.isfinite(cur[m]) and m >= 500)); y = np.array([cur[m] for m in M]); s = np.array([sig_th(m) for m in M])
    ex = np.where(y < s)[0]
    if len(ex) and ex[-1] + 1 < len(M):
        i = ex[-1]; d0, d1 = np.log(y[i] / s[i]), np.log(y[i + 1] / s[i + 1])
        return float(M[i] + (M[i + 1] - M[i]) * (-d0) / (d1 - d0))
    return float("nan")


def low(variant, f):
    d = json.load(open(f"{D}/rerun_{variant}.json")); out = {}
    for M in sorted(d, key=int):
        E = [(t, c, r) for t, x in d[M].items() for c, r in x.items() if not c.endswith("+cats")]
        C = [(t, c, r) for t, x in d[M].items() for c, r in x.items() if c.endswith("+cats")]
        o = {}
        t, c, r = min(E, key=lambda e: e[2]["limit_fb"]); o["pub_single"] = r["limit_fb"]; o["pub_single_sel"] = f"{c}@{t}"; pub = r
        o["fixed_Pall_450"] = d[M]["450"]["P+all"]["limit_fb"]; o["fixed_P_450"] = d[M]["450"]["P"]["limit_fb"]
        ns, sel = [], []
        for h in (0, 1):
            t, c, r = min(E, key=lambda e: e[2]["halves"][h]["tune"]["lim"]); ns.append(r["halves"][h]["lim"]); sel.append(f"{c}@{t}")
        o["nest_single"] = gm(ns) if np.all(np.isfinite(ns)) else float("inf"); o["nest_single_sel"] = sel
        # diagnostics of the published single selection
        H = pub["halves"]
        o["eval_over_tune_lim"] = [h["lim"] / h["tune"]["lim"] for h in H]
        o["B_eval_over_tune"] = [h["xsB_fb"] / h["tune"]["xsB_fb"] for h in H]
        o["S_eval_over_tune"] = [h["effS"] / h["tune"]["effS"] for h in H]
        o["neffB_eval"] = [h["neffB"] for h in H]; o["neffB_tune"] = [h["tune"]["neffB"] for h in H]; o["nB_eval"] = [h["nB"] for h in H]
        o["peak_eval"] = [h["qcd_peak_over_M"] for h in H]; o["peak_all"] = [h["qcd_peak_all"] for h in H]
        o["B_events"] = [h["xsB_fb"] * L for h in H]; o["effS"] = [h["effS"] for h in H]
        for k in ("comp", "top1", "top5", "dB_rel"):
            o[k] = [h.get(k) for h in H]
        # all (trig, conf) eval/tune ratios (for the global statistic)
        o["all_ratio"] = [(h["lim"] / h["tune"]["lim"], h["xsB_fb"] / h["tune"]["xsB_fb"]) for _, _, r in E for h in r["halves"] if np.isfinite(h["lim"]) and h["tune"]["lim"] > 0 and np.isfinite(h["tune"]["lim"])]
        if C:
            t, c, r = min(C, key=lambda e: e[2]["limit_fb"])
            o["pub_cats"] = r["limit_fb"]; o["pub_cats_sel"] = f"{c}@{t}"; o["pub_cats_corr"] = r["limit_corr_fb"]
            sc = [rho_scan(h["S_per_fb"], h["B"], f) for h in r["halves"]]
            o["pub_cats_rho"] = [gm([sc[0][i], sc[1][i]]) for i in range(len(RHO))]
            o["pub_cats_ncat"] = [len(h["bounds"]) for h in r["halves"]]; o["pub_cats_B"] = [h["B"] for h in r["halves"]]
            o["cats_eval_over_tune"] = [h["lim_u"] / h["tune_u"] for h in r["halves"]]
            nu, nc, nr, sel = [], [], [], []
            for h in (0, 1):
                t, c, r = min(C, key=lambda e: e[2]["halves"][h]["tune_u"]); hh = r["halves"][h]
                nu.append(hh["lim_u"]); nc.append(hh["lim_c"]); nr.append(rho_scan(hh["S_per_fb"], hh["B"], f)); sel.append(f"{c}@{t}")
            o["nest_cats"] = gm(nu); o["nest_cats_corr"] = gm(nc); o["nest_cats_rho"] = [gm([nr[0][i], nr[1][i]]) for i in range(len(RHO))]; o["nest_cats_sel"] = sel
            # nested, choice made with the correlated model on the tuning half
            nc2 = []
            for h in (0, 1):
                t, c, r = min(C, key=lambda e: e[2]["halves"][h]["tune_c"]); nc2.append(r["halves"][h]["lim_c"])
            o["nest_cats_corrchoice"] = gm(nc2)
        out[int(M)] = o
    return out


def high(key="f0.1", f=0.1):
    d = json.load(open(f"{D}/rerun_high.json")); out = {}
    for M in sorted(d, key=int):
        r = d[M][key]; H = r["halves"]; o = {"single": r["limit_fb"], "cats": r["count_cats_fb"], "cats_corr": r["count_cats_corr_fb"]}
        if not np.isfinite(r["limit_fb"]):
            out[int(M)] = o; continue
        o["eval_over_tune_lim"] = [h["lim"] / h["tune"]["lim"] for h in H]; o["B_eval_over_tune"] = [h["B"] / h["tune"]["B"] for h in H]
        o["S_eval_over_tune"] = [h["S_per_fb"] / h["tune"]["S_per_fb"] for h in H]
        o["neffB_eval"] = [h["neffB"] for h in H]; o["neffB_tune"] = [h["tune"]["neffB"] for h in H]; o["nB_eval"] = [h["nB"] for h in H]
        o["B_events"] = [h["B"] for h in H]; o["effS"] = [h["S_per_fb"] / L for h in H]; o["peak"] = [h["qcd_peak_over_M"] for h in H]
        for k in ("comp", "top1", "top5", "dB_rel", "B_ht_lt900_frac", "B_invalid_frac", "S_invalid_frac"):
            o[k] = [h.get(k) for h in H]
        if "cats_S" in H[0] and "cats_S" in H[1]:
            sc = [rho_scan(h["cats_S"], h["cats_B"], f) for h in H]
            o["cats_rho"] = [gm([sc[0][i], sc[1][i]]) for i in range(len(RHO))]; o["ncat"] = [len(h["cats_S"]) for h in H]
            o["cats_eval_over_tune"] = [h["count_cats_fb"] / h["cats_tune_u"] for h in H]
            # nested choice single vs categories is trivial (categories always at least as good on the tuning half)
        o["tune_single"] = [h["tune"]["lim"] for h in H]; o["eval_single"] = [h["lim"] for h in H]
        o["gauss_over_onoff"] = float(np.mean([asimov_G(h["B"], f) / s_up(h["B"], f) for h in H])); o["LN_over_onoff"] = float(np.mean([asimov_LN(h["B"], f) / s_up(h["B"], f) for h in H]))
        out[int(M)] = o
    return out


if __name__ == "__main__":
    res = {}
    for v, f in (("main", 0.1), ("peakboth", 0.1), ("t700_full", 0.1), ("t700_fullw", 0.1), ("t700_f5", 0.05), ("t700_f5w", 0.05)):
        try:
            res[v] = low(v, f)
        except FileNotFoundError:
            pass
    try:
        for k, f in (("f0.1", 0.1), ("f0.05", 0.05), ("f0.1_valid", 0.1), ("f0.1_valid_ht450", 0.1)):
            res["high_" + k] = high(k, f)
        res["high_nopeak"] = {int(M): {"single": r["f0.1_nopeak"]["limit_fb"], "cats": r["f0.1_nopeak"]["count_cats_fb"]} for M, r in json.load(open(f"{D}/rerun_high.json")).items()}
    except FileNotFoundError:
        pass
    json.dump(res, open(f"{D}/nested.json", "w"), indent=1, default=float)
    f2 = lambda v: "/".join(f"{x:.2f}" for x in v)
    for v in [k for k in res if not k.startswith("high")]:
        print(f"\n== low-mass pipeline, variant {v}: published best-of vs nested choice (limits in fb)")
        print(f"{'M':>5} {'pub single':>10} {'sel':>13} {'nested':>8} {'pub/nest':>8} {'Pall@450':>9} | {'pub cats':>8} {'nested':>8} {'pub/nest':>8} {'corr':>8} {'worst rho':>9} {'rho*':>4} {'nest worst':>10} | lim eval/tune  B eval/tune  neff eval  peak eval  peak all")
        for M, o in res[v].items():
            c = ""
            if "pub_cats" in o:
                i = int(np.argmax(o["pub_cats_rho"]))
                c = f"{o['pub_cats']:8.4g} {o['nest_cats']:8.4g} {o['pub_cats'] / o['nest_cats']:8.3f} {o['pub_cats_corr']:8.4g} {max(o['pub_cats_rho']):9.4g} {RHO[i]:4.1f} {max(o['nest_cats_rho']):10.4g}"
            print(f"{M:5d} {o['pub_single']:10.4g} {o['pub_single_sel']:>13} {o['nest_single']:8.4g} {o['pub_single'] / o['nest_single']:8.3f} {o['fixed_Pall_450']:9.4g} | {c:>60} | {f2(o['eval_over_tune_lim'])}  {f2(o['B_eval_over_tune'])}  {'/'.join(f'{x:.0f}' for x in o['neffB_eval'])}  {f2(o['peak_eval'])}  {f2(o['peak_all'])}")
        allr = np.array([x for o in res[v].values() for x in o["all_ratio"]])
        if len(allr):
            lr = np.log(allr)
            print(f"   all (mass, trigger, configuration, half) fits, n = {len(lr)}: geo-mean lim(eval)/lim(tune) = {np.exp(lr[:, 0].mean()):.3f} (x/ {np.exp(lr[:, 0].std() / np.sqrt(len(lr))):.3f}), geo-mean B(eval)/B(tune) = {np.exp(lr[:, 1].mean()):.3f} (x/ {np.exp(lr[:, 1].std() / np.sqrt(len(lr))):.3f}), fraction with B(eval) > B(tune): {(lr[:, 1] > 0).mean():.2f}")
        print("   composition of B in the published single selection (evaluation halves): sample B-fraction (events, n_eff); top-1 / top-5 weight fraction; dB/B (MC stat)")
        for M, o in res[v].items():
            for h in (0, 1):
                if o["comp"][h]:
                    print(f"   {M:5d} h{h}: " + ", ".join(f"{s} {x['B_frac']:.2f} ({x['n']}, {x['neff']:.1f})" for s, x in o["comp"][h].items()) + f"; top1 {o['top1'][h]:.2f} top5 {o['top5'][h]:.2f}; dB/B {o['dB_rel'][h]:.2f}; B = {o['B_events'][h]:.4g}")
    if "high_f0.1" in res:
        hv = res["high_f0.1"]
        print("\n== high-mass pipeline (fisher_high COUNT_F=0.1)")
        print(f"{'M':>5} {'single':>8} {'cats u':>8} {'cats c':>8} {'worst rho':>9} {'rho*':>4} {'+valid':>8} {'+valid,HT450':>12} {'no peak':>8} {'f=0.05':>8} {'cats5':>8} | lim e/t  B e/t  neff eval  top1  top5  dB/B  B(HT<900)  B invalid  G/onoff  LN/onoff  comp")
        for M, o in hv.items():
            if "comp" not in o:
                print(M, o); continue
            i = int(np.argmax(o["cats_rho"])) if "cats_rho" in o else 0
            cr = max(o["cats_rho"]) if "cats_rho" in o else float("nan")
            print(f"{M:5d} {o['single']:8.4g} {o['cats']:8.4g} {o['cats_corr']:8.4g} {cr:9.4g} {RHO[i]:4.1f} {res['high_f0.1_valid'][M]['single']:8.4g} {res['high_f0.1_valid_ht450'][M]['single']:12.4g} {res['high_nopeak'][M]['single']:8.4g} {res['high_f0.05'][M]['single']:8.4g} {res['high_f0.05'][M]['cats']:8.4g} | "
                  f"{f2(o['eval_over_tune_lim'])}  {f2(o['B_eval_over_tune'])}  {'/'.join(f'{x:.0f}' for x in o['neffB_eval'])}  {f2(o['top1'])}  {f2(o['top5'])}  {f2(o['dB_rel'])}  {f2(o['B_ht_lt900_frac'])}  {f2(o['B_invalid_frac'])}  {o['gauss_over_onoff']:.3f}  {o['LN_over_onoff']:.3f}  "
                  + " | ".join(", ".join(f"{s} {x['B_frac']:.2f} ({x['n']}, {x['neff']:.0f})" for s, x in c.items()) for c in o["comp"]))
        lr = np.log(np.array([[a, b] for o in hv.values() if "comp" in o for a, b in zip(o["eval_over_tune_lim"], o["B_eval_over_tune"])]))
        print(f"   high-mass fits n = {len(lr)}: geo-mean lim(eval)/lim(tune) = {np.exp(lr[:, 0].mean()):.3f} (x/ {np.exp(lr[:, 0].std() / np.sqrt(len(lr))):.3f}), B(eval)/B(tune) = {np.exp(lr[:, 1].mean()):.3f} (x/ {np.exp(lr[:, 1].std() / np.sqrt(len(lr))):.3f})")
    if "main" in res and "high_f0.1" in res:
        lo, hv = res["main"], res["high_f0.1"]
        cur = {}
        def put(name, lowf, highf, nested1000=None):
            c = {M: lowf(o) for M, o in lo.items() if M < 1000}
            c[1000] = nested1000 if nested1000 is not None else min(lowf(lo[1000]), highf(hv[1000]))
            c.update({M: highf(o) for M, o in hv.items() if M > 1000}); cur[name] = c
        # nested low-vs-high choice at 1000 GeV, per half on the tuning-half limits (same half index in both pipelines)
        n1 = gm([lo[1000]["nest_single"], hv[1000]["single"]]) if False else None
        put("published single", lambda o: o["pub_single"], lambda o: o["single"])
        put("published categories (uncorrelated)", lambda o: o.get("pub_cats", np.inf), lambda o: o["cats"])
        put("nested single", lambda o: o["nest_single"], lambda o: o["single"])
        put("nested categories (uncorrelated)", lambda o: o.get("nest_cats", np.inf), lambda o: o["cats"])
        put("nested categories (fully correlated)", lambda o: o.get("nest_cats_corr", np.inf), lambda o: o["cats_corr"])
        put("nested categories (worst rho)", lambda o: max(o["nest_cats_rho"]) if "nest_cats_rho" in o else np.inf, lambda o: max(o["cats_rho"]) if "cats_rho" in o else o["cats"])
        put("high: + valid pairing, H_T > 450", lambda o: o["nest_single"], lambda o: res["high_f0.1_valid_ht450"][[m for m in hv if hv[m] is o][0]]["single"])
        print("\n== combined curves (fb) and reach")
        Ms = sorted(cur["published single"])
        print(f"{'curve':<40}" + "".join(f"{m:>8d}" for m in Ms) + "   reach")
        for k, c in cur.items():
            print(f"{k:<40}" + "".join(f"{c[m]:8.3g}" for m in Ms) + f"   {reach(c):.0f}")
        json.dump({k: {"curve": c, "reach": reach(c)} for k, c in cur.items()}, open(f"{D}/curves.json", "w"), indent=1, default=float)
