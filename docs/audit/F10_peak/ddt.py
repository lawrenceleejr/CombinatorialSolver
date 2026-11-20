"""Analysis-level decorrelation: replace the gate logit by its QCD tail quantile within the event's m_avg bin
(fitted on the even-index half of the QCD with A < 0.1), so that any cut keeps the same QCD fraction at every
m_avg.  Writes a scores npz with only the odd-index QCD (weights x2) + all signal, for an unbiased evaluation.
usage: ddt.py in_scores.npz out_scores.npz"""
import sys, numpy as np
d = dict(np.load(sys.argv[1]))
s = d["is_signal"].astype(bool); q = ~s
idx = np.arange(len(s)); even, odd = q & (idx % 2 == 0), q & (idx % 2 == 1)
edges = np.exp(np.linspace(np.log(150), np.log(3000), 15))
fit = even & (d["asym"] < 0.1)
lm = np.log(np.clip(d["mavg"], 150, 2999))
b_all = np.clip(np.digitize(d["mavg"], edges) - 1, 0, len(edges) - 2)
score = np.zeros(len(s))
for b in range(len(edges) - 1):
    sel = fit & (b_all == b)
    if sel.sum() < 50:  # merge sparse bins with the neighbour below
        sel = fit & (np.abs(b_all - b) <= 1)
    l, w = d["logit"][sel], d["w"][sel]
    o = np.argsort(l); cl = l[o]; tail = 1 - np.cumsum(w[o]) / w.sum() + w[o] / w.sum()  # P(logit' >= l)
    tgt = b_all == b
    k = np.searchsorted(cl, d["logit"][tgt], side="left")
    qv = np.where(k < len(cl), tail[np.minimum(k, len(cl) - 1)], 0.5 * w[o][-1] / w.sum())
    score[tgt] = -np.log(np.clip(qv, 1e-9, 1))
keep = s | odd
out = {k: v[keep] for k, v in d.items()}
out["logit"] = score[keep]
out["w"] = np.where(out["is_signal"], out["w"], 2 * out["w"])
np.savez_compressed(sys.argv[2], **out)
print("wrote", sys.argv[2], "QCD kept", int(odd.sum()))
