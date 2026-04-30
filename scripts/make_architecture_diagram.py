"""
Professional architecture diagram for JetAssignmentTransformer.
Physics-conditioned GroupTransformer (arXiv:2202.03772-inspired).

Usage:
    python scripts/make_architecture_diagram.py [output_path]

Default output: architecture_diagram.pdf in the repository root.
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import numpy as np

# ── palette (dark-theme scientific) ─────────────────────────────────────────
BG          = "#0D1117"   # page background
PANEL_BG    = "#161B22"   # module panel background
BORDER      = "#30363D"   # panel border
ACCENT_BLUE = "#58A6FF"   # main accent
ACCENT_GRN  = "#3FB950"   # secondary accent
ACCENT_ORG  = "#F0883E"   # physics / warm accent
ACCENT_RED  = "#FF7B72"   # new feature / highlight
ACCENT_PUR  = "#BC8CFF"   # ISR / output
ACCENT_TEA  = "#39D353"   # outputs
TEXT_MAIN   = "#E6EDF3"
TEXT_DIM    = "#8B949E"
TEXT_TINY   = "#6E7681"
ARROW_CLR   = "#8B949E"
NEW_CLR     = "#FF7B72"   # the new physics-conditioning path

fig_w, fig_h = 22, 30
fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.set_facecolor(BG)
ax.axis("off")

# ── helpers ──────────────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, title, body="", accent=ACCENT_BLUE,
         title_size=10, body_size=7.5, alpha_face=0.12):
    """Rounded box with coloured left-edge accent strip."""
    x0, y0 = cx - w / 2, cy - h / 2
    # shadow
    shadow = FancyBboxPatch((x0 + 0.06, y0 - 0.06), w, h,
                            boxstyle="round,pad=0.15",
                            fc="#000000", ec="none", alpha=0.35, zorder=2)
    ax.add_patch(shadow)
    # main fill
    face = FancyBboxPatch((x0, y0), w, h,
                          boxstyle="round,pad=0.15",
                          fc=accent, ec=accent, alpha=alpha_face, zorder=3,
                          linewidth=0)
    ax.add_patch(face)
    # border
    border = FancyBboxPatch((x0, y0), w, h,
                            boxstyle="round,pad=0.15",
                            fc="none", ec=accent, alpha=0.7, zorder=4,
                            linewidth=1.4)
    ax.add_patch(border)
    # accent left strip
    strip = FancyBboxPatch((x0, y0 + h * 0.05), 0.12, h * 0.9,
                           boxstyle="round,pad=0.04",
                           fc=accent, ec="none", alpha=0.9, zorder=5)
    ax.add_patch(strip)
    # title
    ax.text(cx + 0.1, cy if not body else cy + h * 0.22,
            title, ha="center", va="center",
            fontsize=title_size, color=TEXT_MAIN, fontweight="bold",
            fontfamily="monospace", zorder=6)
    if body:
        ax.text(cx + 0.1, cy - h * 0.18,
                body, ha="center", va="center",
                fontsize=body_size, color=TEXT_DIM,
                fontfamily="monospace", linespacing=1.5, zorder=6)


def dim_tag(ax, cx, cy, txt, color=TEXT_TINY, size=6.5):
    """Small dimension annotation."""
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=size, color=color, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL_BG, ec=BORDER,
                      linewidth=0.8, alpha=0.92),
            zorder=10)


def arr(ax, x0, y0, x1, y1, color=ARROW_CLR, lw=1.5, rad=0.0,
        style="-|>", label="", label_side="right", lbl_size=6.5,
        dashed=False):
    ls = (0, (4, 3)) if dashed else "solid"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=style, color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=16, linestyle=ls),
                zorder=7)
    if label:
        mx = (x0 + x1) / 2 + (0.15 if label_side == "right" else -0.15)
        my = (y0 + y1) / 2
        ax.text(mx, my, label, fontsize=lbl_size, color=color,
                ha="left" if label_side == "right" else "right",
                va="center", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc=PANEL_BG, ec="none", alpha=0.8),
                zorder=9)


def section_bg(ax, x0, y0, w, h, label="", color=ACCENT_BLUE, alpha=0.04):
    """Subtle background panel for a logical section."""
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0.2",
                       fc=color, ec=color, alpha=alpha,
                       linewidth=0.8, zorder=1)
    ax.add_patch(p)
    if label:
        ax.text(x0 + 0.3, y0 + h - 0.35, label,
                fontsize=7, color=color, alpha=0.7,
                fontfamily="monospace", fontweight="bold", zorder=2)


# ═══════════════════════════════════════════════════════════════════════════
# TITLE BLOCK
# ═══════════════════════════════════════════════════════════════════════════
ax.text(fig_w / 2, 29.3,
        "JetAssignmentTransformer — Architecture",
        ha="center", va="center",
        fontsize=18, color=TEXT_MAIN, fontweight="bold",
        fontfamily="monospace")
ax.text(fig_w / 2, 28.8,
        "Physics-Conditioned GroupTransformer  ·  Inspired by arXiv:2202.03772",
        ha="center", va="center",
        fontsize=10, color=ACCENT_ORG, fontfamily="monospace")
ax.axhline(28.55, xmin=0.05, xmax=0.95, color=BORDER, linewidth=0.8)

# ═══════════════════════════════════════════════════════════════════════════
# LEFT / MAIN COLUMN  (x ≈ 6)
# RIGHT / DETAILS     (x ≈ 16)
# ═══════════════════════════════════════════════════════════════════════════
XL = 6.2    # left column centre
XR = 16.0   # right column centre
XM = 11.1   # merge / middle

# ── INPUT ────────────────────────────────────────────────────────────────────
section_bg(ax, 0.5, 27.1, fig_w - 1, 1.2, "INPUT", ACCENT_BLUE, 0.05)
rbox(ax, XL, 27.55, 8, 0.75,
     "7 Jets  ·  (E, px, py, pz)₄",
     body="normalized by event HT  ·  pT-sorted",
     accent=ACCENT_BLUE, title_size=11, body_size=8)

# ── PROJECTION ───────────────────────────────────────────────────────────────
Y_PROJ = 26.0
rbox(ax, XL, Y_PROJ, 8, 0.65,
     "Linear Projection  +  Positional Embedding",
     body="R⁴ → R^{d}    ·    d_model = 256",
     accent=ACCENT_BLUE)
arr(ax, XL, 27.18, XL, Y_PROJ + 0.33)
dim_tag(ax, XL + 4.3, Y_PROJ, "(B, 7, 256)")

# ── ENCODER ──────────────────────────────────────────────────────────────────
section_bg(ax, 0.5, 24.6, fig_w - 1, 1.25, "ENCODER", ACCENT_GRN, 0.05)
Y_ENC = 25.15
rbox(ax, XL, Y_ENC, 8, 0.75,
     "Transformer Encoder",
     body="6 layers  ·  8 heads  ·  d_ff = 1024  ·  pre-LN  ·  pT-hierarchy attn bias: w·log(pTᵢ/pTⱼ)",
     accent=ACCENT_GRN, title_size=11, body_size=7.8)
arr(ax, XL, Y_PROJ - 0.33, XL, Y_ENC + 0.38)
dim_tag(ax, XL + 4.3, Y_ENC, "(B, 7, 256)")

# ── ENUMERATE ────────────────────────────────────────────────────────────────
Y_ENUM = 23.55
rbox(ax, XL, Y_ENUM, 8, 0.65,
     "Enumerate All 70 Assignments",
     body="(ISR jet j)  ×  (grouping k ∈ C₆³/2)    →    n_combos = 7 × 10 = 70",
     accent=ACCENT_GRN)
arr(ax, XL, Y_ENC - 0.38, XL, Y_ENUM + 0.33)

# ── FORK: two branches  ───────────────────────────────────────────────────────
# branch point
BP_X, BP_Y = XL, Y_ENUM - 0.33 - 0.2
ax.plot(BP_X, BP_Y, "o", ms=6, color=ACCENT_GRN, zorder=8)
ax.text(BP_X - 0.3, BP_Y, "fork", fontsize=6.5, color=TEXT_DIM,
        ha="right", va="center", fontfamily="monospace")

XB_L = 5.0    # left branch (GroupTransformer)
XB_R = 14.5   # right branch (physics features)

# ═══════════════════════════════════════════════════════════════════════════
# LEFT BRANCH — GroupTransformer
# ═══════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.4, 14.8, 10.2, 8.5, "GROUP TRANSFORMER", ACCENT_ORG, 0.04)

# arrow down to GT area
arr(ax, BP_X, BP_Y, XB_L, 22.35, color=TEXT_DIM, rad=-0.15)

# ── RAW 4-MOM GATHER ─────────────────────────────────────────────────────────
Y_GATHER = 22.1
rbox(ax, XB_L, Y_GATHER, 7.5, 0.65,
     "Gather Per-Group Jet 4-Momenta  (raw)",
     body="gather(four_momenta, g1_idx)  →  g1_jets  ·  g2_jets    shape (B, 70, 3, 4)",
     accent=ACCENT_ORG, body_size=7.2)
dim_tag(ax, XB_L + 3.85, Y_GATHER, "(B,70,3,4)")

# ── INTRA PHYSICS ────────────────────────────────────────────────────────────
Y_INTRA = 20.85
rbox(ax, XB_L, Y_INTRA, 7.5, 1.0,
     "Intra-Group Physics  (LATE, from raw kinematics)",
     body=(
         "intra_group_features(g_jets)  →  9 observables per group\n"
         "  • max pT ratio  ·  pT coeff of variation\n"
         "  • min Lund z  ·  max Lund kT\n"
         "  • ECF₂  ·  ECF₃  ·  D₂ = ECF₃/ECF₂²\n"
         "  • Dalitz max/min pairwise mass ratio"
     ),
     accent=ACCENT_ORG, title_size=9.5, body_size=7, alpha_face=0.18)
arr(ax, XB_L, Y_GATHER - 0.33, XB_L, Y_INTRA + 0.5)
dim_tag(ax, XB_L + 3.85, Y_INTRA, "(B,70,9)")

# ── PHYSICS PROJ ─────────────────────────────────────────────────────────────
Y_PROJ2 = 19.45
rbox(ax, XB_L, Y_PROJ2, 7.5, 0.65,
     "group_physics_proj  ·  Linear(9→d) + GELU",
     body="projects intra physics to d_model  →  conditioning token per group",
     accent=NEW_CLR, title_size=9, body_size=7.5)
arr(ax, XB_L, Y_INTRA - 0.5, XB_L, Y_PROJ2 + 0.33, color=NEW_CLR)
dim_tag(ax, XB_L + 3.85, Y_PROJ2, "(B,70,1,256)")

# ── APPEND TOKEN ─────────────────────────────────────────────────────────────
Y_APPEND = 18.3
rbox(ax, XB_L, Y_APPEND, 7.5, 0.65,
     "Append Conditioning Token",
     body="[jet₁ · jet₂ · jet₃ · phys_cond]    seq_len = 3+1 = 4 tokens",
     accent=NEW_CLR, title_size=9.5, body_size=7.5)
arr(ax, XB_L, Y_PROJ2 - 0.33, XB_L, Y_APPEND + 0.33, color=NEW_CLR)
dim_tag(ax, XB_L + 3.85, Y_APPEND, "(B,70,4,256)")

# also bring down jet embeddings from encoder
arr(ax, BP_X, BP_Y, XB_L - 3.3, 18.3, color=ACCENT_GRN, rad=0.3,
    label="jet_emb", label_side="left")

# ── GROUP TRANSFORMER ────────────────────────────────────────────────────────
Y_GT = 16.95
rbox(ax, XB_L, Y_GT, 7.5, 1.1,
     "GroupTransformer  (shared, intra-group attn)",
     body=(
         "2 layers  ·  4 heads  ·  pre-LN  ·  d_ff = 2×d\n"
         "Self-attention over [jet₁, jet₂, jet₃, phys_cond]\n"
         "→ mean-pool over 4 tokens  →  group embedding"
     ),
     accent=ACCENT_ORG, title_size=10, body_size=7.8, alpha_face=0.2)
arr(ax, XB_L, Y_APPEND - 0.33, XB_L, Y_GT + 0.55, color=ACCENT_ORG)
dim_tag(ax, XB_L + 3.85, Y_GT, "(B,70,256)")

# highlight box around new feature
hl = FancyBboxPatch((XB_L - 3.9, 18.0 - 0.85), 7.9, 2.9,
                    boxstyle="round,pad=0.12",
                    fc="none", ec=NEW_CLR, alpha=0.6, linewidth=2,
                    linestyle="--", zorder=8)
ax.add_patch(hl)
ax.text(XB_L + 3.7, 18.0 - 0.72, "NEW", fontsize=7.5,
        color=NEW_CLR, fontweight="bold", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=NEW_CLR, linewidth=1),
        zorder=9)

# ── SYMMETRIC COMBINATION ────────────────────────────────────────────────────
Y_SYM = 15.65
rbox(ax, XB_L, Y_SYM, 7.5, 0.75,
     "Symmetric Group Combination",
     body="sym_sum = g1+g2  ·  sym_prod = g1⊙g2  ·  sym_diff = |g1−g2|",
     accent=ACCENT_ORG)
arr(ax, XB_L, Y_GT - 0.55, XB_L, Y_SYM + 0.38, color=ACCENT_ORG)
dim_tag(ax, XB_L + 3.85, Y_SYM, "3×(B,70,256)")

# ═══════════════════════════════════════════════════════════════════════════
# RIGHT BRANCH — Full Physics Features
# ═══════════════════════════════════════════════════════════════════════════
section_bg(ax, 10.8, 14.8, 10.8, 8.5, "FULL PHYSICS FEATURES  (24)", ACCENT_RED, 0.04)

arr(ax, BP_X, BP_Y, XB_R, 22.35, color=TEXT_DIM, rad=0.22)

Y_PHYS = 20.5
rbox(ax, XB_R, Y_PHYS, 8, 2.5,
     "Full Physics Features  (24 per assignment)",
     body=(
         "Computed from raw 4-momenta  —  LATE, not from latent space\n"
         "─────────────────────────────────────────────────────────\n"
         "Inter-group (6):\n"
         "  mass_sum · mass_asym · mass_ratio · m₁ · m₂ · ΔR\n"
         "\n"
         "Intra-group × 2 groups (9+9=18):\n"
         "  pT_max/pT_min · pT-CV · min Lund-z · max Lund-kT\n"
         "  ECF₂ · ECF₃ · D₂=ECF₃/ECF₂²\n"
         "  Dalitz max ratio · Dalitz min ratio"
     ),
     accent=ACCENT_RED, title_size=10, body_size=7.5, alpha_face=0.15)
arr(ax, XB_R, 22.35, XB_R, Y_PHYS + 1.25, color=TEXT_DIM)
dim_tag(ax, XB_R + 4.2, Y_PHYS, "(B,70,24)")

Y_LN = 17.85
rbox(ax, XB_R, Y_LN, 6.5, 0.65,
     "LayerNorm  (physics_norm)",
     body="stabilises mixed-scale features  (ratios · masses · angles)",
     accent=ACCENT_RED, body_size=7.5)
arr(ax, XB_R, Y_PHYS - 1.25, XB_R, Y_LN + 0.33, color=ACCENT_RED)
dim_tag(ax, XB_R + 3.35, Y_LN, "(B,70,24)")

# extract mass_sum / mass_asym before LN (dashed side arrow)
arr(ax, XB_R + 3.3, Y_PHYS - 0.5, XB_R + 5.2, Y_PHYS - 0.5,
    color=TEXT_DIM, lw=1.2, dashed=True,
    label="raw mass_sum\nmass_asym", label_side="right")

# ═══════════════════════════════════════════════════════════════════════════
# MERGE — CONCAT + SCORER
# ═══════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.4, 10.2, 21.2, 4.45, "GROUPING SCORER + ISR HEAD", ACCENT_PUR, 0.04)

Y_CAT = 14.65
rbox(ax, XM, Y_CAT, 12, 0.75,
     "Concatenate  [sym_sum | sym_prod | sym_diff | physics_normed]",
     body="shape: (B, 70, 3·d + 24)    ·    d = 256   →   (B, 70, 792)",
     accent=ACCENT_PUR, title_size=10)
# arrows from both branches
arr(ax, XB_L, Y_SYM - 0.38, XM - 5.5, Y_CAT + 0.18, color=ACCENT_ORG, rad=0.1)
arr(ax, XB_R, Y_LN - 0.33, XM + 5.5, Y_CAT + 0.18, color=ACCENT_RED, rad=-0.1)

Y_GSCORER = 13.35
rbox(ax, XM, Y_GSCORER, 10, 0.85,
     "Grouping Scorer MLP",
     body=(
         "Linear(792→512) → GELU → Dropout\n"
         "→ Linear(512→256) → GELU → Dropout → Linear(256→1)\n"
         "→ grouping_logits    (B, 7, 10)"
     ),
     accent=ACCENT_PUR, title_size=10, body_size=7.5)
arr(ax, XM, Y_CAT - 0.38, XM, Y_GSCORER + 0.43, color=ACCENT_PUR)

Y_GSUM = 11.95
rbox(ax, XM - 1.5, Y_GSUM, 8, 0.75,
     "Attention-Pooled Grouping Summary",
     body="softmax(grp_logits) · combined  →  grouping_summary_proj  →  (B, 7, d)",
     accent=ACCENT_PUR, body_size=7.5)
arr(ax, XM, Y_GSCORER - 0.43, XM - 1.5, Y_GSUM + 0.38,
    color=ACCENT_PUR, rad=-0.15)

Y_ISR = 10.6
rbox(ax, XM - 1.5, Y_ISR, 8, 0.9,
     "ISR Head",
     body=(
         "input: [jet_emb | LOO_ctx | isr_physics | grp_summary]\n"
         "Linear(3d+3→2d) → GELU → Drop → Linear → GELU → Drop → Linear(→1)\n"
         "→ isr_logits    (B, 7)"
     ),
     accent=ACCENT_PUR, title_size=10, body_size=7.5)
arr(ax, XM - 1.5, Y_GSUM - 0.38, XM - 1.5, Y_ISR + 0.45, color=ACCENT_PUR)

# jet_emb feed (dotted) into ISR head
arr(ax, XL, Y_ENC - 0.38, XM - 1.5 - 4.1, Y_ISR + 0.0,
    color=ACCENT_GRN, rad=0.4, lw=1.2, dashed=True, label="jet_emb",
    label_side="left", lbl_size=6)

# ═══════════════════════════════════════════════════════════════════════════
# COMBINE LOGITS
# ═══════════════════════════════════════════════════════════════════════════
Y_COMB = 9.15
rbox(ax, XM, Y_COMB, 10.5, 0.75,
     "Combine Logits",
     body=(
         "log P(assign) = scale · isr_logits + grouping_logits\n"
         "→ flat_logits    (B, 70)    canonical assignment ordering"
     ),
     accent=ACCENT_PUR, title_size=10)
arr(ax, XM - 1.5, Y_ISR - 0.45, XM, Y_COMB + 0.38, color=ACCENT_PUR, rad=0.1)
arr(ax, XM, Y_GSCORER - 0.43, XM + 0.2, Y_COMB + 0.38, color=ACCENT_PUR, rad=0.15)

# ═══════════════════════════════════════════════════════════════════════════
# ADVERSARIAL HEAD (right side)
# ═══════════════════════════════════════════════════════════════════════════
Y_ADV = 11.8
rbox(ax, 20.3, Y_ADV, 3.5, 1.2,
     "Adversarial\nHead",
     body=(
         "Gradient Reversal\n"
         "mean(jet_emb)\n"
         "→ Linear→GELU→Lin(1)\n"
         "→ mass_pred"
     ),
     accent=ACCENT_RED, title_size=8.5, body_size=7)
arr(ax, XL + 2, Y_ENC - 0.2, 20.3, Y_ADV + 0.6,
    color=ACCENT_RED, rad=-0.4, lw=1.2, dashed=True, label="GRL", lbl_size=6)

# ═══════════════════════════════════════════════════════════════════════════
# OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.4, 6.8, 21.2, 2.1, "OUTPUTS", ACCENT_TEA, 0.04)

out_specs = [
    ("logits\n(B, 70)",      3.3,  ACCENT_PUR),
    ("isr_logits\n(B, 7)",   7.0,  ACCENT_PUR),
    ("grp_logits\n(B,7,10)", 10.7, ACCENT_PUR),
    ("mass_asym\nmass_sum",  14.4, ACCENT_RED),
    ("mass_pred",            18.0, ACCENT_RED),
]
arr(ax, XM, Y_COMB - 0.38, XM, 8.7, color=ACCENT_TEA)

for lbl, xc, clr in out_specs:
    rbox(ax, xc, 7.65, 3.2, 0.75, lbl, accent=clr, title_size=8, alpha_face=0.2)

# arrows from combine to output boxes
for xc, _ in [(3.3, 0), (7.0, 0), (10.7, 0)]:
    arr(ax, XM, 8.37, xc, 8.03, color=ACCENT_PUR, lw=1.2)
arr(ax, XB_R + 5.2, Y_PHYS - 0.5, 14.4, 8.03, color=ACCENT_RED, lw=1.2, rad=0.3)
arr(ax, 20.3, Y_ADV - 0.6, 18.0, 8.03, color=ACCENT_RED, lw=1.2)

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOSSES (bottom)
# ═══════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.4, 4.5, 21.2, 2.05, "TRAINING LOSSES", ACCENT_ORG, 0.04)

losses = [
    ("CrossEntropy\n(assignment)", 2.7,  ACCENT_PUR),
    ("CrossEntropy\n(ISR)",        6.0,  ACCENT_PUR),
    ("λ_sym · E[mass_asym]",       9.4,  ACCENT_ORG),
    ("λ_qcd · (−E[H·m_asym])",    13.0,  ACCENT_ORG),
    ("λ_adv · MSE\n(adversary)",  16.5,  ACCENT_RED),
    ("KL distill\n(Phase 1→2)",   19.8,  ACCENT_RED),
]
for lbl, xc, clr in losses:
    rbox(ax, xc, 5.4, 3.0, 0.75, lbl, accent=clr, title_size=7.5, alpha_face=0.2)

# ═══════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=ACCENT_BLUE, alpha=0.7,  label="Input / Projection"),
    mpatches.Patch(facecolor=ACCENT_GRN,  alpha=0.7,  label="Transformer Encoder"),
    mpatches.Patch(facecolor=ACCENT_ORG,  alpha=0.7,  label="GroupTransformer"),
    mpatches.Patch(facecolor=ACCENT_RED,  alpha=0.7,  label="Physics Features (raw kinematics)"),
    mpatches.Patch(facecolor=NEW_CLR,     alpha=0.7,  label="★ NEW: Physics→GroupTransformer conditioning"),
    mpatches.Patch(facecolor=ACCENT_PUR,  alpha=0.7,  label="Scorer / ISR / Combination"),
    mpatches.Patch(facecolor=ACCENT_TEA,  alpha=0.7,  label="Outputs"),
    plt.Line2D([0], [0], color=ARROW_CLR, lw=1.5, linestyle="--",
               label="Dotted: auxiliary / skip connection"),
]
leg = ax.legend(handles=legend_items, loc="lower left",
                fontsize=8, framealpha=0.92, edgecolor=BORDER,
                facecolor=PANEL_BG,
                labelcolor=TEXT_DIM,
                bbox_to_anchor=(0.01, 0.005))
for txt in leg.get_texts():
    txt.set_fontfamily("monospace")

# ── footer ───────────────────────────────────────────────────────────────────
ax.text(fig_w / 2, 0.25,
        "lawrenceleejr/CombinatorialSolver  ·  GroupTransformer physics conditioning  ·  "
        "Inspired by PELICAN (arXiv:2202.03772)",
        ha="center", va="center", fontsize=7, color=TEXT_TINY,
        fontfamily="monospace")

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Generate architecture diagram PDF")
parser.add_argument(
    "output",
    nargs="?",
    default=os.path.join(_repo_root, "architecture_diagram.pdf"),
    help="Output PDF path (default: <repo_root>/architecture_diagram.pdf)",
)
args = parser.parse_args()

out = args.output
fig.savefig(out, bbox_inches="tight", dpi=200, facecolor=BG)
print(f"Saved: {out}  ({os.path.getsize(out)//1024} KB)")
plt.close(fig)
