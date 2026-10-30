"""Working points of the 10% mass-window selection: 1 - P(signal) threshold vs mass (top category), with a smooth
log-linear parameterisation for use between the tuned masses. One panel.
usage: wp_plot.py style out working_points.json"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out, wpj = sys.argv[1:4]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
wp = {int(k): v for k, v in json.load(open(wpj)).items()}
M = np.array([m for m in sorted(wp) if m >= 300 and wp[m]["cuts"] and wp[m]["cuts"].get("psig", -1) > 0.9])
y = np.array([1 - wp[m]["cuts"]["psig"] for m in M])
extra = np.array([wp[m]["sel"] != "P" for m in M])
c = np.polyfit(M, np.log10(y), 1)
fig, ax = plt.subplots(figsize=(4.4, 2.9))
ax.plot(M[extra], y[extra], "o", ms=4, color="#9b2c2c", label="tuned with extra cuts (H$_T$/M, $\\Delta$, Dalitz, $p_{T,6}$, A)")
ax.plot(M[~extra], y[~extra], "o", ms=4, color="0.3", label="tuned, P(signal) cut only")
mm = np.linspace(300, 2400, 50)
ax.plot(mm, 10 ** np.polyval(c, mm), color="#2b5d8a", lw=1.0, ls="--", label=f"1 $-$ P$_{{\\rm min}}$ = 10$^{{{c[1]:.2f} {c[0] * 1000:+.2f}\\,M/{{\\rm TeV}}}}$")
ax.set_yscale("log"); ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"1 $-$ P(signal) threshold")
ax.set_title("r14A working points for the 10% mass-window search (top category)", fontsize=6.5, loc="left")
ax.legend(fontsize=5.8, frameon=False, loc="lower left")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/wp_psig.pdf"); fig.savefig(f"{out}/wp_psig.png", dpi=170, bbox_inches="tight")
print("fit: log10(1-P) = %.3f %+.3f M/TeV" % (c[1], c[0] * 1000), " ".join(f"{m}:{1 - 10 ** np.polyval(c, m):.4f}" for m in (300, 500, 1000, 1500, 2000, 2400)))
