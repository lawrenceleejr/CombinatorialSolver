"""F4 audit plots (one panel per PDF)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import f4lib as L
S = "/home/llee/.claude/skills/synced/a7952cd1-f33e-4b50-b57a-831b94afa47b_fca9c36b-51d5-47de-aed9-4cb04835fa9b/tufte-plots/assets"
plt.style.use(S + "/tufte.mplstyle")
Ms = [400, 700, 1000, 1500, 2000]
R = {M: json.load(open(f"out/ceil_m{M}.json")) for M in Ms}
for w, wl in (("w85", "[0.85, 1.1] M"), ("w92", "[0.92, 1.04] M")):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    rows = [("mass oracle, sizes 2-5 in 10", lambda r: r["schemes"]["std"]["ex_s25_10"]["any_" + w], "0.7", ":"),
            ("majority, pT>20, leading 12", lambda r: r["schemes"]["pt20"]["flex12"][w], "#1b6ca8", "-"),
            ("majority, leading 10", lambda r: r["schemes"]["std"]["flex10"][w], "#1b6ca8", "--"),
            ("best 3+3 in 7", lambda r: r["schemes"]["std"]["b33_7"][w], "0.3", "-"),
            ("r14A", lambda r: r["schemes"]["std"]["r14A"][w], "#c0392b", "-")]
    for lab, f, c, ls in rows:
        y = [100 * f(R[M]) for M in Ms]
        ax.plot(Ms, y, ls, color=c, lw=1.2, marker="o", ms=2.5)
        ax.text(2050, y[-1], lab, color=c, fontsize=7, va="center")
    ax.set_xlabel("gluino mass M [GeV]"); ax.set_ylabel(f"% of signal, $A<0.1$, m in {wl}")
    ax.set_xlim(350, 2000); ax.set_ylim(0, 100)
    fig.savefig(f"plots/ceiling_{w}.pdf", bbox_inches="tight"); plt.close(fig)
# m_avg/M distributions at 1 TeV: r14A vs ancestry majority (A<0.1)
M = 1000
d = L.load_signal(f"samples/sigan_m{M}_c*.h5", max_files=6); jc = L.jet_collection(d)
z = np.load(f"out/r14a_m{M}.npz"); ms = z["masses"]
fig, ax = plt.subplots(figsize=(4.6, 3.3)); bins = np.linspace(0.3, 1.6, 66)
def h(m1, m2, ok, c, lab, y):
    t = m1 + m2; A = np.abs(m1 - m2) / np.maximum(t, 1e-9); s = ok & (A < .1)
    hh, _ = np.histogram(0.5 * t[s] / M, bins); hh = hh / len(m1) / (bins[1] - bins[0])
    ax.step(bins[:-1], hh, where="post", color=c, lw=1); ax.text(1.32, y, lab, color=c, fontsize=7)
h(ms[:, 0], ms[:, 1], z["ok"], "#c0392b", "r14A", 2.6)
m1, m2, n1, n2 = L.flex_assign(jc, 10); h(m1, m2, (n1 > 0) & (n2 > 0), "#1b6ca8", "ancestry majority, 10 jets", 3.0)
m1, m2, ok, _ = L.best33_assign(jc, 7); h(m1, m2, ok, "0.3", "ancestry best 3+3 in 7", 2.2)
for x in (0.85, 1.1): ax.axvline(x, color="0.8", lw=0.6)
ax.set_xlabel(r"$m_{avg}/M$ ($A<0.1$), M = 1 TeV"); ax.set_ylabel("events / all produced / unit")
fig.savefig("plots/mavg_m1000.pdf", bbox_inches="tight"); plt.close(fig)
