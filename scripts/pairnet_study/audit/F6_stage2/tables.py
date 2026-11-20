"""Collect F6 job outputs (out/*.tar -> out/full/*.json) into markdown tables: out/tables.md."""
import json, glob, tarfile, os
os.makedirs("out/full", exist_ok=True)
for t in glob.glob("out/b34_*.tar") + glob.glob("out/cond_*.tar") + glob.glob("out/s2*.tar") + glob.glob("out/combo_*.tar"):
    with tarfile.open(t) as z:
        for m in z.getmembers():
            if m.name.endswith(".json"):
                m.name = os.path.basename(m.name); z.extract(m, "out/full")
def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        for M, v in json.load(open(f)).items():
            d.setdefault(M, {}).update(v)
    return dict(sorted(d.items(), key=lambda kv: int(kv[0])))
CO = load("out/full/combo_*.json")
B, C, S2 = load("out/full/b34_*.json"), load("out/full/cond_*.json"), load("out/full/s2*.json")
L = []
def cell(v, key="ratio"):
    if v is None: return "-"
    if not v.get("feasible", True): return "infeas"
    return f"{v[key]:.2f}±{v[key + '_err']:.2f}" if key + "_err" in v else f"{v[key]:.2f}"
def table(title, D, names, key="ratio", ref=None):
    for f in ("f0.1", "f0.05"):
        L.append(f"\n### {title} ({f})\n")
        L.append("| M | " + (f"{ref} limit fb | neffB(eval) | " if ref else "") + " | ".join(names) + " |")
        L.append("|---" * (len(names) + (3 if ref else 1)) + "|")
        for M, r in D.items():
            if f not in r: continue
            sel = r[f]["sel"]; row = [M]
            if ref: row += [f"{sel[ref]['limit_fb']:.4g}±{100 * sel[ref]['limit_relerr']:.0f}%", f"{sel[ref]['neffB']:.0f}"]
            row += [cell(sel.get(n), key) for n in names]
            L.append("| " + " | ".join(map(str, row)) + " |")
if B:
    table("Baselines and items 3/4, ratio vs P+all", B, ["P", "P+boost", "P+all+cats", "P+all|win", "P+all|mest", "P+all|mest|win", "P+all|both", "P+all|wavg|win"], ref="P+all")
    x4 = sorted({k for r in B.values() for k in r["f0.1"]["sel"] if k.startswith("P+all+") and k != "P+all+cats"})
    table("Item 4 (P+all + X), ratio vs P+all", B, x4)
    p4 = sorted({k for r in B.values() for k in r["f0.1"]["sel"] if k.startswith("P|") or (k.startswith("P+") and not k.startswith(("P+all", "P+boost")))})
    table("Items 3/4 on top of P only, ratio vs P", B, p4, key="ratio_vs_P", ref="P")
if S2:
    names = [k for k in S2[next(iter(S2))]["f0.1"]["sel"] if not k.startswith("P")]
    table("Item 1 stage 2, ratio vs P+all", S2, ["P", "P+all+cats"] + names, ref="P+all")
    table("Item 1 stage 2 (dr2, lin2), ratio vs P", S2, ["dr2", "lin2"], key="ratio_vs_P", ref="P")
if C:
    names = list(C[next(iter(C))]["f0.1"]["sel"])
    table("Item 2 conditioned pairing, ratio vs arg:P", C, names, ref="arg:P")
    L.append("\n### Item 2 sculpting at the arg:P cut (f0.1)\n")
    for M, r in C.items():
        L.append(f"- {M}: " + "; ".join(f"{k}: effS {v['sig_eff']:.4f}, QCD {v['qcd_fb']:.3g} fb (n {v['qcd_n']}, neff {v['qcd_neff']:.0f})" for k, v in r["f0.1"]["sculpt"].items() if isinstance(v, dict)))
if CO:
    table("Item 5: window re-tuned + categories vs current recipe (P+all+cats)", CO, ["P+all", "P+all|win", "P+all|win+cats"], ref="P+all+cats")
open("out/tables.md", "w").write("\n".join(L) + "\n")
print(f"wrote out/tables.md ({len(L)} lines); masses b34 {list(B)} s2 {list(S2)} cond {list(C)}")
