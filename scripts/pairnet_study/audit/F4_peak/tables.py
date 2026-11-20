"""F4 audit: markdown tables from out/ceil_m*.json and out/bkg_*.json -> out/tables.md"""
import glob, json, sys
import numpy as np
od = sys.argv[1] if len(sys.argv) > 1 else "out"
R = {json.load(open(f))["mass"]: json.load(open(f)) for f in sorted(glob.glob(od + "/ceil_m*.json"))}
Ms = sorted(R)
o = []
SZ = json.load(open(od + "/attach.json"))["sizes"]  # ceil.py size table used the wrong masses; attach.py recomputes it
P = lambda x: f"{100 * x:.1f}"
o.append("events per mass: " + ", ".join(f"{M}: {R[M]['n']}" for M in Ms) + "\n")
o.append("pass standard selection (>= 6 jets pT > 30, |eta| < 3): " + ", ".join(f"{M}: {P(R[M]['pass_sel'])}%" for M in Ms) + "\n")

def table(title, rows, key="std"):
    o.append(f"\n### {title}\n\ncells: % of all produced events with A<0.1 in [0.85,1.1]M / in [0.92,1.04]M / 68% half-width of m_avg/M (A<0.1 candidates)\n")
    o.append("| scheme | " + " | ".join(f"M={M}" for M in Ms) + " |\n|---|" + "---|" * len(Ms))
    for lab, jd, k in rows:
        cells = []
        for M in Ms:
            v = R[M]["schemes"].get(jd, {}).get(k)
            if v is None: cells.append("–")
            elif "fA" in v: cells.append(f"{P(v['w85'])} / {P(v['w92'])} / {v['w68']:.3f}")
            else: cells.append(f"{P(v['any_w85'])} / {P(v['any_w92'])}")
        o.append(f"| {lab} | " + " | ".join(cells) + " |")

rows = [("r14A (argmax, no gate cut)", "std", "r14A"), ("r14A, P(sig) > 0.9", "std", "r14A_p0.9")]
rows += [(f"ancestry-best 3+3 in leading {N}", "std", f"b33_{N}") for N in (7, 8, 9, 10)]
rows += [(f"ancestry majority, free sizes, leading {N}", "std", f"flex{N}") for N in (7, 8, 9, 10, 12)]
rows += [(f"  same, sizes restricted to 2–5, leading {N}", "std", f"flex{N}_s25") for N in (7, 8, 10)]
rows += [(f"  same, sizes restricted to 2–4, leading {N}", "std", f"flex{N}_s24") for N in (7, 8, 10)]
table("Ceilings with the standard jets (R=0.4, pT>30, |eta|<3), ancestry-defined groupings", rows)
rows = []
for jd, lab in (("pt20", "pT>20"), ("eta5", "|eta|<4.9"), ("pt20eta5", "pT>20, |eta|<4.9")):
    rows += [(f"{lab}: best 3+3 in {N}", jd, f"b33_{N}") for N in (7, 10)] + [(f"{lab}: majority, leading {N}", jd, f"flex{N}") for N in (7, 10, 12)]
for jd, lab in (("R08", "R=0.8 reclustered (from pT>30 jets)"), ("R08pt20", "R=0.8 reclustered (from pT>20 jets)"), ("R12", "R=1.2 reclustered")):
    rows += [(f"{lab}: majority, leading {N}", jd, f"flex{N}") for N in (4, 6, 8)]
table("Changing the jets", rows)
rows = [(f"3+3 in {N} ({R[Ms[0]]['schemes']['std'][f'ex_33_{N}']['npairs']} groupings)", "std", f"ex_33_{N}") for N in (7, 8, 9, 10)]
rows += [(f"sizes 2–5 in {N} ({R[Ms[0]]['schemes']['std'][f'ex_s25_{N}']['npairs']})", "std", f"ex_s25_{N}") for N in (7, 8, 9, 10)]
rows += [(f"sizes 2–4 in {N} ({R[Ms[0]]['schemes']['std'][f'ex_s24_{N}']['npairs']})", "std", f"ex_s24_{N}") for N in (7, 8, 10)]
rows += [(f"R=0.8 fat jets, sizes {t[1]}–{t[2]} in {N}", "R08", f"ex_{t}_{N}") for t, N in (("s13", 6), ("s14", 6), ("s14", 8))]
o.append("\n### Mass oracle (NOT a usable target: any grouping of the scheme with A<0.1 in the window); cells: % in [0.85,1.1]M / [0.92,1.04]M\n")
o.append("| scheme | " + " | ".join(f"M={M}" for M in Ms) + " |\n|---|" + "---|" * len(Ms))
for lab, jd, k in rows:
    o.append(f"| {lab} | " + " | ".join(f"{P(R[M]['schemes'][jd][k]['any_w85'])} / {P(R[M]['schemes'][jd][k]['any_w92'])}" if k in R[M]['schemes'][jd] else "–" for M in Ms) + " |")
o.append("\n### Minimum-asymmetry classical proxy on signal; cells: % in [0.85,1.1]M / [0.92,1.04]M\n")
o.append("| scheme | " + " | ".join(f"M={M}" for M in Ms) + " |\n|---|" + "---|" * len(Ms))
for lab, jd, k in rows:
    o.append(f"| {lab} | " + " | ".join(f"{P(R[M]['schemes'][jd][k]['minA_w85'])} / {P(R[M]['schemes'][jd][k]['minA_w92'])}" if k in R[M]['schemes'][jd] else "–" for M in Ms) + " |")

o.append("\n### Loss ladder (standard jets); cells as in the first table\n")
o.append("| step | " + " | ".join(f"M={M}" for M in Ms) + " |\n|---|" + "---|" * len(Ms))
for k in R[Ms[0]]["ladder"]:
    o.append(f"| {k} | " + " | ".join(f"{P(R[M]['ladder'][k]['w85'])} / {P(R[M]['ladder'][k]['w92'])} / {R[M]['ladder'][k]['w68']:.3f}" for M in Ms) + " |")
o.append("\n### Decay-quark classification (before FSR), % of all produced events\n")
o.append("| | " + " | ".join(f"M={M}" for M in Ms) + " |\n|---|" + "---|" * len(Ms))
for k in R[Ms[0]]["quarks"]:
    o.append(f"| {k} | " + " | ".join(P(R[M]["quarks"][k]) for M in Ms) + " |")
for k in ("r14A_pure33", "r14A_inwin_pure33", "r14A_inwin_impure"):
    o.append(f"| {k} | " + " | ".join(P(R[M][k]) for M in Ms) + " |")
o.append("\n### Group sizes under ancestry majority (leading 10 standard jets): % of events / % of those in [0.85,1.1]M with A<0.1\n")
keys = ["3+3", "3+4", "4+4", "3+5", "4+5", "2+3", "2+4", "2+2", "2+5", "3+6"]
o.append("| sizes | " + " | ".join(f"M={M}" for M in Ms) + " |\n|---|" + "---|" * len(Ms))
for k in keys:
    o.append(f"| {k} | " + " | ".join(f"{P(SZ[str(M)].get(k, [0, 0])[0])} / {P(SZ[str(M)].get(k, [0, 0])[1])}" for M in Ms) + " |")
o.append("| other | " + " | ".join(P(1 - sum(SZ[str(M)].get(k, [0, 0])[0] for k in keys)) for M in Ms) + " |")
o.append("\n### Label purity of the leading 10 standard jets\n")
for M in Ms:
    p = R[M]["purity"]
    o.append(f"- M={M}: mean fractions g1/g2/other = " + "/".join(f"{x:.3f}" for x in p["mean label fractions (g1, g2, other), leading 10"]) +
             f"; majority<0.8: {P(p['jets with majority fraction < 0.8'])}%; majority=other: {P(p['jets with majority = other'])}%; by rank: " +
             " ".join(f"{100 * x:.0f}" for x in p["per rank majority=other"]))

# background
for fn in sorted(glob.glob(od + "/bkg_*.json")):
    B = json.load(open(fn)); t = B["tot"]
    if "n" not in t or "test" in fn: continue
    o.append(f"\n### Background cost on {B['tag']} ({B['nfiles']} files, {int(t['n'])} events, n_eff = {t['sw']**2 / t['sw2']:.0f})\n")
    wins = sorted({k.split("/")[2] for k in t if k.count("/") == 3}, key=lambda s: (int(s.split("_")[0]), s))
    for kind in ("any", "minA", "rand"):
        o.append(f"\n{kind}: weighted % of QCD events (ratio to 3+3-in-7) [raw events]\n")
        o.append("| scheme | " + " | ".join(wins) + " |\n|---|" + "---|" * len(wins))
        for sch in sorted({"/".join(k.split("/")[:2]) for k in t if k.count("/") == 3}, key=lambda s: (s.split("/")[0] != "std", s.split("/")[1].split("_")[0], int(s.split("_")[-1]))):
            cells = []
            for w in wins:
                v, ref = t[f"{sch}/{w}/{kind}"], t[f"std/33_7/{w}/{kind}"]
                cells.append(f"{100 * v[0] / t['sw']:.3g} ({v[0] / max(ref[0], 1e-300):.2f}) [{v[2]}]")
            o.append(f"| {sch} | " + " | ".join(cells) + " |")
open(od + "/tables.md", "w").write("\n".join(o) + "\n")
print("\n".join(o))
