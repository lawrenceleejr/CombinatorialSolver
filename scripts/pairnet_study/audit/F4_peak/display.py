"""F4 audit: eta-phi event displays of the ancestry labels (one PDF per event) + a text dump."""
import sys
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = "/home/llee/.claude/skills/synced/a7952cd1-f33e-4b50-b57a-831b94afa47b_fca9c36b-51d5-47de-aed9-4cb04835fa9b/tufte-plots/assets"
plt.style.use(S + "/tufte.mplstyle")
fn, events = sys.argv[1], [int(x) for x in sys.argv[2:]]
f = h5py.File(fn)
tr, lpt, na, std = f["ANC/true"][:], f["ANC/label_pt"][:], f["ANC/n"][:], f["ANC/std_index"][:]
q = f["truth/quarks_before_fsr"][:]
col = ["#1b6ca8", "#c0392b", "0.55"]
for e in events:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    print(f"event {e}: jets (true pT, eta, phi | fractions g1 g2 other | network slot)")
    for j in range(na[e]):
        pt, eta, phi = tr[e, j, :3]
        fr = lpt[e, j] / max(lpt[e, j].sum(), 1e-9)
        k = int(fr.argmax())
        print(f"  {pt:7.1f} {eta:5.2f} {phi:5.2f} | {fr[0]:.2f} {fr[1]:.2f} {fr[2]:.2f} | {std[e, j]}")
        ax.add_patch(plt.Circle((eta, phi), 0.4, fc=col[k], ec="none", alpha=0.25 + 0.6 * fr[k] * (pt > 30)))
        ax.text(eta, phi, f"{pt:.0f}", ha="center", va="center", fontsize=6)
    for g in range(2):
        ax.plot(q[e, 3 * g:3 * g + 3, 1], q[e, 3 * g:3 * g + 3, 2], "x", color=col[g], ms=5, mew=1.2)
    ax.set_xlim(-5, 5); ax.set_ylim(-3.6, 3.6); ax.set_aspect("equal")
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$\phi$")
    ax.text(-4.9, 3.75, "jets by ancestry majority: gluino 1", color=col[0], fontsize=7)
    ax.text(1.3, 3.75, "gluino 2", color=col[1], fontsize=7); ax.text(3.0, 3.75, "other", color="0.4", fontsize=7)
    ax.text(-4.9, 4.25, r"number: jet $p_T$ [GeV]; crosses: decay quarks; faint: $p_T<30$ GeV", fontsize=6)
    fig.savefig(f"plots/display_m{int(f.attrs['mass'])}_ev{e}.pdf", bbox_inches="tight"); plt.close(fig)
