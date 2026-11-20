"""Working points audited by F3 (all with: valid pairing, A < 0.1, H_T > 450, m_avg/M in the window)."""
W, Nw = (0.85, 1.10), (0.92, 1.04)
WPS = {
    "400 P+all":       dict(M=400, win=W, psig=0.98833, dmax=0.1, dmin=0.1, d32=0.15, d63=0.6, ptlow=0.15, asym=0.08),
    "400 P":           dict(M=400, win=W, psig=0.98833),
    "700 P+all":       dict(M=700, win=W, psig=0.99630, dmax=0.1, dmin=-0.1, d32=0.15, d63=0.8, ptlow=0.05, asym=0.08),
    "700 P":           dict(M=700, win=W, psig=0.99705),
    "700 P+boost":     dict(M=700, win=W, psig=0.99705, dmax=0.2, dmin=-0.2),
    "700 P narrow":    dict(M=700, win=Nw, psig=0.99713),
    "700 P+boost nar": dict(M=700, win=Nw, psig=0.99711, dmax=0.0, dmin=0.0),
    "1000 P+boost":    dict(M=1000, win=W, psig=0.997445, dmax=0.1, dmin=0.1),
    "1000 P":          dict(M=1000, win=W, psig=0.997445),
    "1500 P":          dict(M=1500, win=W, psig=0.998078),
    "2000 P":          dict(M=2000, win=W, psig=0.99928),
}
QCD = {400: ("TvlmT", "Uvlm", (280, 1150)), 700: ("TvlmT", "Uvlm", (280, 1150)), 1000: ("TvlmT", "Uvlm", (280, 1150)),
       1500: ("ThtT", "Uht", (1100, 2600)), 2000: ("ThtT", "Uht", (1100, 2600))}
TRIG = 450.0


def aux(d, wp, i=None):
    """Everything except the P cut. d: dict of arrays (N, V) or (N,); i: variant index or None."""
    g = (lambda k: d[k][:, i]) if i is not None else (lambda k: d[k])
    M = wp["M"]; x = g("mavg") / M
    ok = g("valid") & (g("asym") < wp.get("asym", 0.1)) & (x >= wp["win"][0]) & (x < wp["win"][1]) & (g("ht") > TRIG)
    for k, s in (("dmax", 1), ("dmin", 1), ("ptlow", 1)):
        if k in wp:
            ok &= g(k) / M > wp[k]
    for k in ("d32", "d63"):
        if k in wp:
            ok &= g(k) < wp[k]
    return ok
