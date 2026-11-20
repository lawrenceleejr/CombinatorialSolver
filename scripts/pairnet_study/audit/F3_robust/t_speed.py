import time, glob, numpy as np, sys
from lib import *
sess = session(int(sys.argv[1]))
pt, eta, phi, m, mask, w, at = load_h5("/scratch/llee/pairnet/prod_lowmass/siglm_m700_incl_c16.h5")
pt, eta, phi, m, mask = widen(pt, eta, phi, m, mask)
t = time.time(); o = evaluate(sess, pt, eta, phi, m, mask); dt = time.time() - t
print("threads", sys.argv[1], "events/s", len(pt) / dt, "psig max", o["psig"].max(), "valid", o["valid"].mean())
sel = o["valid"] & (o["asym"] < 0.1) & (o["mavg"] > 595) & (o["mavg"] < 770)
print("win frac", sel.mean(), "P>0.997", (sel & (o["psig"] >= 0.99705)).mean(), "ntried", at["n_tried"])
o2 = evaluate(sess, *perturb("pu:2", pt, eta, phi, m, mask))
print("pu2 nj diff", (o2["nj"].astype(int) - o["nj"]).mean())
