#!/usr/bin/env python
"""Tar bundles of held-out h5 files for OSPool re-scoring. Writes bundles/<name>.tar and bundles/list.txt (name kind)."""
import glob, os, re, subprocess, sys
O = "/ospool/uc-shared/project/futurecolliders/llee/pairnet"
B = "/scratch/llee/pairnet/audit/F6_stage2/bundles"
def cnum(f): return int(re.search(r"_c(\d+)", os.path.basename(f)).group(1))
def chunks(l, n): return [l[i:i + n] for i in range(0, len(l), n)]
jobs = []
def bundle(tag, files, per, kind):
    files = sorted(files, key=cnum)
    for i, ch in enumerate(chunks(files, per)):
        name = f"{tag}_{i:03d}"
        tar = f"{B}/{name}.tar"
        if not os.path.exists(tar):
            lst = f"{B}/{name}.lst"
            open(lst, "w").write("\n".join(ch) + "\n")
            subprocess.check_call(["tar", "-cf", tar + ".tmp", "--transform", "s,.*/,in/,", "-T", lst])
            os.rename(tar + ".tmp", tar); os.remove(lst)
        jobs.append((name, kind))
    print(tag, len(files), "files", flush=True)
bundle("old", glob.glob("/scratch/llee/pairnet/data_r3/prod/qcd_c*[89].h5"), 50, "qcd")
bundle("ht", [f for f in glob.glob(f"{B}/../ht_h5/prod_ht/qcdht_c*.h5") if cnum(f) >= 1400], 100, "qcd")
bundle("vlm", [f for f in glob.glob(f"{O}/prod_lowmass_spill/qcdvlm_c*.h5") if cnum(f) >= 1100], 60, "qcd")
bundle("vlmT", [f for f in glob.glob(f"{O}/tail/qcdvlmT_c*_tail.h5") if cnum(f) >= 900], 150, "qcd")
bundle("htT", [f for f in glob.glob(f"{O}/tail/qcdhtT_c*_tail.h5") if cnum(f) >= 500], 150, "qcd")
bundle("q7T", glob.glob(f"{O}/t700/qcd700T_c*_tail.h5"), 300, "qcd")
bundle("q7U", glob.glob(f"{O}/t700/qcd700U_c*_tail.h5"), 300, "qcd")
for m in (200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000):
    bundle(f"siglm{m}", [f for k in ("incl", "boost") for c in (16, 17, 18, 19)
                         for f in glob.glob(f"/scratch/llee/pairnet/prod_lowmass/siglm_m{m}_{k}_c{c}.h5")], 8, "sig")
for m in range(400, 2401, 100):
    bundle(f"sig{m}", [f for c in (8, 9, 18, 19) for f in glob.glob(f"/scratch/llee/pairnet/data_r3/prod/sig_m{m}_c{c}.h5")], 4, "sig")
open(f"{B}/list.txt", "w").write("".join(f"{n} {k}\n" for n, k in jobs))
print(len(jobs), "bundles")
