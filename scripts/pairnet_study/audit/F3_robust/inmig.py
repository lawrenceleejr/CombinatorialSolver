"""Share of the varied background (fixed cuts) carried by tail-core events with nominal P in [0.9, 0.95) and [0.95, 0.97):
a large share signals that in-migration from P_nom < 0.9 (not in any sample) is likely non-negligible."""
import numpy as np
from wps import *
D = "/scratch/llee/pairnet/audit/F3_robust"
for name in ("400 P", "700 P", "700 P+all", "1000 P", "1500 P", "2000 P"):
    wp = WPS[name]; T = np.load(f"{D}/out/merged/{QCD[wp['M']][0]}.npz"); T = {k: T[k] for k in T.files}
    V = [str(v) for v in T["variants"]]; p0 = T["p0"]; s = []
    for i, v in enumerate(V):
        if v == "nom":
            continue
        m = aux(T, wp, i) & (T["psig"][:, i] >= wp["psig"]); w = T["w"] * m; B = w.sum()
        s.append(f"{v}:{w[p0 < 0.95].sum() / B:.2f}/{w[(p0 >= 0.95) & (p0 < 0.97)].sum() / B:.2f}" if B > 0 else f"{v}:-")
    print(name, " ".join(s))
