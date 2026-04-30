"""
Generate a PDF diagram of the JetAssignmentTransformer network architecture.

Illustrates the data flow, where physics features are computed, and how the
GroupTransformer can access that information – as discussed in arXiv:2202.03772.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_INPUT    = "#AED6F1"   # light blue
C_ENCODER  = "#A9DFBF"   # light green
C_GROUP    = "#F9E79F"   # light yellow – GroupTransformer
C_PHYSICS  = "#F1948A"   # light red   – physics features
C_SCORER   = "#D7BDE2"   # light purple – scorer MLPs
C_OUTPUT   = "#D5D8DC"   # light grey   – outputs
C_ISR      = "#FAD7A0"   # light orange – ISR head
C_ADV      = "#FDFEFE"   # white-ish    – adversary

ARROW_KW = dict(arrowstyle="-|>", color="#333333", lw=1.5,
                connectionstyle="arc3,rad=0.0",
                mutation_scale=14)
CURVED_KW = dict(arrowstyle="-|>", color="#E74C3C", lw=1.5,
                 connectionstyle="arc3,rad=0.35",
                 mutation_scale=14)

def box(ax, x, y, w, h, label, sublabel="", color="#FFFFFF",
        fontsize=8, sublabel_fontsize=6.5, radius=0.03):
    """Draw a rounded rectangle with centred label and optional sublabel."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=color, edgecolor="#555555", linewidth=1.2, zorder=3,
    )
    ax.add_patch(patch)
    if sublabel:
        ax.text(x, y + h * 0.13, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x, y - h * 0.25, sublabel, ha="center", va="center",
                fontsize=sublabel_fontsize, color="#555555", zorder=4,
                style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)
    return (x, y, w, h)

def arrow(ax, x0, y0, x1, y1, curved=False, label="", color="#333333", rad=0.0):
    style = f"arc3,rad={rad}"
    kw = dict(arrowstyle="-|>", color=color, lw=1.5,
              connectionstyle=style, mutation_scale=14, zorder=5)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=kw)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.04, my, label, fontsize=5.5, color=color, zorder=6,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.8))

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

# Column centres
XL   = 3.5    # left column  (main flow)
XR   = 10.5   # right column (grouping / scoring detail)
XM   = 7.0    # middle – links / physics

# ---------------------------------------------------------------------------
# ── INPUT ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
Y_INPUT = 16.8
box(ax, XL, Y_INPUT, 4.5, 0.7,
    "7 Jets  (E, px, py, pz)",
    sublabel="normalized by event HT",
    color=C_INPUT, fontsize=9)

# Input projection
Y_PROJ = 15.6
box(ax, XL, Y_PROJ, 4.5, 0.7,
    "Linear Projection  +  Positional Embedding",
    sublabel="(E,px,py,pz)  →  d_model = 256",
    color=C_INPUT)
arrow(ax, XL, Y_INPUT - 0.35, XL, Y_PROJ + 0.35)

# ---------------------------------------------------------------------------
# ── TRANSFORMER ENCODER ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
Y_ENC = 14.2
box(ax, XL, Y_ENC, 4.5, 0.9,
    "Transformer Encoder",
    sublabel="6 layers, 8 heads, d=256  ·  pT-hierarchy attention bias",
    color=C_ENCODER, fontsize=9)
arrow(ax, XL, Y_PROJ - 0.35, XL, Y_ENC + 0.45)

# ---------------------------------------------------------------------------
# ── ENUMERATE 70 ASSIGNMENTS ────────────────────────────────────────────────
# ---------------------------------------------------------------------------
Y_ENUM = 12.85
box(ax, XL, Y_ENUM, 4.5, 0.7,
    "Enumerate All 70 Assignments",
    sublabel="(ISR jet j) × (grouping k)  →  n_combos = 7 × 10",
    color=C_ENCODER)
arrow(ax, XL, Y_ENC - 0.45, XL, Y_ENUM + 0.35)

# Dashed split line into two streams
Y_SPLIT = 12.35
ax.plot([XL], [Y_SPLIT], marker=".", ms=8, color="#555", zorder=6)
# Left stream → GroupTransformer
arrow(ax, XL, Y_SPLIT, XL - 1.8, Y_SPLIT - 0.15)
# Right stream → Physics
arrow(ax, XL, Y_SPLIT, XR - 2.5, Y_SPLIT - 0.15)

# ---------------------------------------------------------------------------
# ── GROUP TRANSFORMER  (left branch) ────────────────────────────────────────
# ---------------------------------------------------------------------------
Y_GT = 11.2
box(ax, XL - 1.8, Y_GT, 4.0, 1.1,
    "GroupTransformer  (shared)",
    sublabel="2 layers, 4 heads\nSelf-attention over 3-jet embeddings per group\n→ mean-pool  →  (g1_emb, g2_emb)",
    color=C_GROUP, fontsize=8.5)
arrow(ax, XL - 1.8, Y_SPLIT - 0.15, XL - 1.8, Y_GT + 0.55)

# Symmetric combination
Y_SYM = 9.85
box(ax, XL - 1.8, Y_SYM, 4.0, 0.7,
    "Symmetric Combination",
    sublabel="sym_sum = g1+g2  ·  sym_prod = g1⊙g2  ·  sym_diff = |g1−g2|",
    color=C_GROUP)
arrow(ax, XL - 1.8, Y_GT - 0.55, XL - 1.8, Y_SYM + 0.35)

# ---------------------------------------------------------------------------
# ── PHYSICS FEATURES  (right branch) ────────────────────────────────────────
# ---------------------------------------------------------------------------
Y_PHYS = 11.2
box(ax, XR, Y_PHYS, 5.0, 1.8,
    "Physics Features  (per assignment)",
    sublabel=(
        "Computed from raw 4-momenta  ← LATE in network\n"
        "─────────────────────────────────────────────\n"
        "Inter-group (6):  m₁, m₂, m_sum, m_asym, m_ratio, ΔR\n"
        "Intra-group ×2 (9+9):\n"
        "  pT_max/pT_min, pT-CV, min-z_Lund, max-kT_Lund\n"
        "  ECF₂, ECF₃, D₂=ECF₃/ECF₂²\n"
        "  Dalitz max/min  →  total 24 features"
    ),
    color=C_PHYSICS, fontsize=7.5)
arrow(ax, XR - 2.5, Y_SPLIT - 0.15, XR, Y_PHYS + 0.9)

# LayerNorm on physics
Y_LN = 9.85
box(ax, XR, Y_LN, 3.5, 0.7,
    "LayerNorm  (physics_norm)",
    sublabel="stabilises mixed-scale physics features before scorer",
    color=C_PHYSICS)
arrow(ax, XR, Y_PHYS - 0.9, XR, Y_LN + 0.35)

# ---------------------------------------------------------------------------
# ── FEEDBACK:  physics → GroupTransformer  (the key innovation arrow) ───────
# ---------------------------------------------------------------------------
# Horizontal dashed arrow from physics box to GroupTransformer
ax.annotate(
    "",
    xy   =(XL - 1.8 + 2.0, Y_GT + 0.0),   # right edge of GT box
    xytext=(XR - 2.5, Y_PHYS - 0.2),
    arrowprops=dict(
        arrowstyle="-|>", color="#E74C3C", lw=2.0,
        connectionstyle="arc3,rad=-0.25",
        mutation_scale=16, linestyle="dashed",
    ),
    zorder=7,
)
ax.text(6.2, 10.5,
        "Physics info\nfed to GroupTransformer\n(proposed – arXiv:2202.03772)",
        fontsize=6.5, color="#C0392B", ha="center", va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FDEBD0", ec="#E74C3C",
                  alpha=0.95))

# ---------------------------------------------------------------------------
# ── CONCAT  → GROUPING SCORER ───────────────────────────────────────────────
# ---------------------------------------------------------------------------
Y_CAT = 8.7
box(ax, XM, Y_CAT, 6.5, 0.65,
    "Concatenate  [sym_sum | sym_prod | sym_diff | physics_normed]",
    sublabel="shape: (batch, n_combos, 3·d_model + 24)",
    color=C_SCORER)
# arrows from both branches into concat
arrow(ax, XL - 1.8, Y_SYM - 0.35, XL - 1.8, Y_CAT + 0.32)
ax.annotate("", xy=(XM - 3.25 + 0.3, Y_CAT + 0.0),
            xytext=(XM - 3.25 + 0.3, Y_CAT + 0.32),
            arrowprops=dict(arrowstyle="-", color="#555", lw=1))
arrow(ax, XR, Y_LN - 0.35, XR, Y_CAT + 0.32)
ax.annotate("", xy=(XM + 3.25 - 0.3, Y_CAT + 0.0),
            xytext=(XM + 3.25 - 0.3, Y_CAT + 0.32),
            arrowprops=dict(arrowstyle="-", color="#555", lw=1))

Y_GSCORER = 7.6
box(ax, XM, Y_GSCORER, 5.5, 0.75,
    "Grouping Scorer MLP",
    sublabel="Linear→GELU→Drop→Linear→GELU→Drop→Linear(1)\n→ grouping_logits (batch, 7, 10)",
    color=C_SCORER)
arrow(ax, XM, Y_CAT - 0.32, XM, Y_GSCORER + 0.375)

# ---------------------------------------------------------------------------
# ── ISR HEAD ────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
Y_GSUM = 6.5
box(ax, XL - 0.5, Y_GSUM, 4.5, 0.75,
    "Attention-Pooled Grouping Summary",
    sublabel="softmax(grp_logits) · combined  →  proj d_model\n→ grouping_summary (batch, 7, d_model)",
    color=C_GROUP)
arrow(ax, XM - 2.5, Y_GSCORER - 0.375, XL - 0.5, Y_GSUM + 0.375, rad=-0.2)

Y_ISR = 5.4
box(ax, XL - 0.5, Y_ISR, 4.5, 0.9,
    "ISR Head",
    sublabel=(
        "Input: [jet_emb | LOO_ctx | isr_physics | grp_summary]\n"
        "= d_model + d_model + 3 + d_model\n"
        "Linear→GELU→Drop×2 → isr_logits (batch, 7)"
    ),
    color=C_ISR, fontsize=8)
arrow(ax, XL - 0.5, Y_GSUM - 0.375, XL - 0.5, Y_ISR + 0.45)

# Jet embeddings feed ISR head from encoder
ax.annotate("",
    xy=(XL - 0.5 - 2.25 + 0.05, Y_ISR + 0.0),
    xytext=(XL, Y_ENC - 0.45),
    arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2,
                    connectionstyle="arc3,rad=0.35",
                    mutation_scale=12, linestyle="dotted"),
    zorder=5)
ax.text(1.5, 9.5, "jet_emb\n(encoder out)",
        fontsize=5.5, color="#666", ha="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# ---------------------------------------------------------------------------
# ── COMBINE LOGITS ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
Y_COMB = 4.1
box(ax, XM, Y_COMB, 6.0, 0.75,
    "Combine Logits",
    sublabel=(
        "log P(assign) = scale · isr_logits + grouping_logits\n"
        "→ flat_logits (batch, 70)  ←  canonical assignment ordering"
    ),
    color=C_SCORER)
arrow(ax, XL - 0.5, Y_ISR - 0.45, XM, Y_COMB + 0.375)
arrow(ax, XM, Y_GSCORER - 0.375, XM, Y_COMB + 0.375)

# ---------------------------------------------------------------------------
# ── ADVERSARIAL HEAD ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
Y_ADV = 6.2
box(ax, XR + 0.3, Y_ADV, 3.2, 1.1,
    "Adversarial Head",
    sublabel=(
        "Gradient Reversal Layer\nmean(jet_emb) →\nLinear→GELU→Linear(1)\n"
        "→ mass_pred  (decorrelate mass)"
    ),
    color=C_ADV, fontsize=7.5)
ax.annotate("",
    xy=(XR + 0.3 - 1.6, Y_ADV + 0.3),
    xytext=(XL + 0.5, Y_ENC - 0.45),
    arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2,
                    connectionstyle="arc3,rad=-0.3",
                    mutation_scale=12, linestyle="dotted"),
    zorder=5)

# ---------------------------------------------------------------------------
# ── OUTPUTS ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
Y_OUT = 2.8
out_labels = [
    ("logits\n(70)", 2.5),
    ("isr_logits\n(7)", 5.0),
    ("grp_logits\n(7×10)", 7.5),
    ("mass_asym\nmass_sum", 10.0),
    ("mass_pred\n(adversary)", 12.5),
]
for lbl, xc in out_labels:
    box(ax, xc, Y_OUT, 2.2, 0.65, lbl, color=C_OUTPUT, fontsize=7)
arrow(ax, XM, Y_COMB - 0.375, XM, Y_OUT + 0.325)

# mass_asym/mass_sum from physics
ax.annotate("",
    xy=(10.0, Y_OUT + 0.325),
    xytext=(XR, Y_LN - 0.35),
    arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2,
                    connectionstyle="arc3,rad=0.2",
                    mutation_scale=12, linestyle="dotted"),
    zorder=5)
# mass_pred from adversary
ax.annotate("",
    xy=(12.5, Y_OUT + 0.325),
    xytext=(XR + 0.3, Y_ADV - 0.55),
    arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2,
                    connectionstyle="arc3,rad=0.0",
                    mutation_scale=12),
    zorder=5)

# ---------------------------------------------------------------------------
# ── LEGEND ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
legend_items = [
    mpatches.Patch(color=C_INPUT,   label="Input / Projection"),
    mpatches.Patch(color=C_ENCODER, label="Transformer Encoder"),
    mpatches.Patch(color=C_GROUP,   label="GroupTransformer"),
    mpatches.Patch(color=C_PHYSICS, label="Physics Features  ← computed LATE"),
    mpatches.Patch(color=C_SCORER,  label="Scorer / Combination MLP"),
    mpatches.Patch(color=C_ISR,     label="ISR Head"),
    mpatches.Patch(color=C_ADV,     label="Adversarial Head"),
    mpatches.Patch(color=C_OUTPUT,  label="Network Outputs"),
    mpatches.Patch(facecolor="white", edgecolor="#E74C3C", linewidth=2,
                   label="Proposed: physics → GroupTransformer feed"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=7,
          framealpha=0.9, edgecolor="#999",
          bbox_to_anchor=(0.99, 0.01))

# ---------------------------------------------------------------------------
# ── TITLE & ANNOTATIONS ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
ax.set_title(
    "JetAssignmentTransformer  –  Network Architecture\n"
    "Physics features computed LATE (from raw 4-momenta) · GroupTransformer access point highlighted",
    fontsize=11, fontweight="bold", pad=12,
)

ax.text(
    0.01, 0.01,
    "Inspiration: arXiv:2202.03772  –  Late-stage physics observables with transformer feedback",
    transform=ax.transAxes, fontsize=6.5, color="#777",
    va="bottom",
)

# ---------------------------------------------------------------------------
# ── SAVE ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
out_path = "architecture_diagram.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=200, facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close(fig)
