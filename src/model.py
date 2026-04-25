"""
Pair-Symmetric Slot Transformer (PSST) for gluino-pair jet assignment.

Design goals (vs. the previous factored architecture that plateaued at ~42 %):

  1. ISR is part of the combinatorial problem, not a separate decision.
     A single joint scorer evaluates every (ISR, g1, g2) hypothesis.  The
     hard case — ISR is *not* the lowest-pT jet — is handled on the same
     footing as the easy case.

  2. No positional embedding tied to pT rank.  The jet slots are exchangeable;
     pT rank enters only as a *feature* so the model can use it without
     being locked into a shortcut.

  3. Pairwise attention bias with physics distances: log(pT_i/pT_j),
     log(ΔR_ij), log(m_ij/HT).  These are injected as additive biases in
     every self-attention layer so the backbone sees relational geometry
     directly (not via attention it has to re-learn from scalars).

  4. Triplet bank.  All C(num_jets, 3) = 35 triplets of jets are turned into
     physically meaningful 3-body tokens (ΔR_max, kT_softest, m_ijk/HT …),
     processed by a small Transformer, and attended to by each jet via a
     masked cross-attention that restricts jet j to the triplets it belongs
     to.  This gives the jet representation direct access to 3-body structure
     — the shape the gluino decays actually produce.

  5. Per-hypothesis intra-group self-attention.  For each of the 70 flat
     assignments, the signal jets (g1 ∪ g2) go through a 1-layer self-attention
     biased by intra-group ΔR; g1 and g2 are pooled separately and combined
     symmetrically (a+b, a*b, (a-b)²) so the scorer is invariant to the
     unordered-parents symmetry.

  6. Physics feature block at scoring time keeps the existing inter-group
     features (mass-sum/asymmetry/ratio, ΔR, masses) plus the Lund / ECF /
     Dalitz features per group, plus ISR-system features that couple the
     ISR candidate to the two signal 3-jets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .combinatorics import (
    build_assignment_tensors,
    build_neighbor_map,
    build_triplet_tensors,
)
from .utils import compute_invariant_mass


# --------------------------------------------------------------------------- #
# Gradient reversal (kept for the optional mass-adversarial head)
# --------------------------------------------------------------------------- #


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# --------------------------------------------------------------------------- #
# Small kinematic utilities
# --------------------------------------------------------------------------- #


def _wrap_dphi(dphi: torch.Tensor) -> torch.Tensor:
    return dphi - 2 * torch.pi * torch.round(dphi / (2 * torch.pi))


def _extract_kinematics(four_momenta: torch.Tensor):
    """Return (E, px, py, pz, pt, eta, phi, m) from a (..., J, 4) tensor."""
    E = four_momenta[..., 0]
    px = four_momenta[..., 1]
    py = four_momenta[..., 2]
    pz = four_momenta[..., 3]
    pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
    eta = torch.asinh(pz / pt)
    phi = torch.atan2(py, px)
    m2 = (E * E - px * px - py * py - pz * pz).clamp(min=0.0)
    m = torch.sqrt(m2 + 1e-8)
    return E, px, py, pz, pt, eta, phi, m


def _pair_invariant_mass(p_i: torch.Tensor, p_j: torch.Tensor) -> torch.Tensor:
    s = p_i + p_j
    m2 = (s[..., 0] ** 2 - s[..., 1] ** 2 - s[..., 2] ** 2 - s[..., 3] ** 2).clamp(min=1e-8)
    return torch.sqrt(m2)


# --------------------------------------------------------------------------- #
# Pairwise attention bias (injected into every self-attention layer)
# --------------------------------------------------------------------------- #


def compute_pair_bias_features(four_momenta: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Three physical pairwise features for attention bias, shape (B, J, J, 3).

    Channels:
        0: log(pT_i / pT_j)                    (pT hierarchy / recoil)
        1: log(ΔR_ij + ε)                      (angular distance)
        2: log(m_ij / HT + ε)                  (pair mass scale, HT-normalised)

    Invalid (zero-padded) jets get their pair entries zeroed; downstream the
    attention mask also hides them.
    """
    _, px, py, pz, pt, eta, phi, _ = _extract_kinematics(four_momenta)

    log_pt = torch.log(pt)
    log_pt_ratio = log_pt.unsqueeze(-1) - log_pt.unsqueeze(-2)              # (B, J, J)

    deta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
    dphi = _wrap_dphi(phi.unsqueeze(-1) - phi.unsqueeze(-2))
    dr = torch.sqrt(deta * deta + dphi * dphi + 1e-8)
    log_dr = torch.log(dr + 1e-6)

    ht = pt.sum(dim=-1, keepdim=True).clamp(min=1e-6)                       # (B, 1)
    E = four_momenta[..., 0]
    p_i = four_momenta.unsqueeze(-2)                                        # (B, J, 1, 4)
    p_j = four_momenta.unsqueeze(-3)                                        # (B, 1, J, 4)
    s = p_i + p_j
    m2_ij = (s[..., 0] ** 2 - s[..., 1] ** 2 - s[..., 2] ** 2 - s[..., 3] ** 2).clamp(min=1e-8)
    m_ij = torch.sqrt(m2_ij)
    log_m_ij = torch.log(m_ij / ht.unsqueeze(-1) + 1e-6)                    # (B, J, J)

    feats = torch.stack([log_pt_ratio, log_dr, log_m_ij], dim=-1)           # (B, J, J, 3)

    # Zero-out rows/cols corresponding to invalid (padded) jets so their bias
    # entries don't leak signal — they're also masked in attention but this
    # keeps pair features from producing spurious numerical gradients.
    v_i = valid_mask.unsqueeze(-1).unsqueeze(-1)                            # (B, J, 1, 1)
    v_j = valid_mask.unsqueeze(-2).unsqueeze(-1)                            # (B, 1, J, 1)
    feats = feats * v_i * v_j

    # Keep these unused locals available for downstream per-jet features.
    del px, py, pz, E
    return feats


# --------------------------------------------------------------------------- #
# Per-jet physics features (slot-equivariant: no positional identity)
# --------------------------------------------------------------------------- #


def compute_jet_features(four_momenta: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Per-jet features, shape (B, J, 10).

    The model's input projection sees these in addition to the raw 4-vector
    so that all kinematic invariants the scorer cares about are available
    from the first layer.  None of the features carries positional identity —
    they depend only on the jet's own kinematics and its neighbourhood.

    pT rank is deliberately NOT included as a feature: it gives the model a
    direct shortcut to "ISR == lowest-pT jet" that dominates training and
    locks the model out of the hard case (ISR is not the lowest-pT jet).
    Absolute pT is still available via log_pt; the model can learn rank
    structure from raw pT if it actually needs it.

    Channels:
        0  log pT
        1  η
        2  sin φ
        3  cos φ
        4  log(m / pT + ε)            (jet mass softness)
        5  valid mask                  (1 for real jet, 0 for zero-pad)
        6  log(min ΔR to other valid jets + ε)
        7  log(kT to nearest valid neighbour + ε)
        8  log(m_nn / HT + ε)          (nearest-neighbour pair mass)
        9  |Δφ(jet, -recoil)|          (recoil-alignment angle — has correlation
                                        with ISR but is not a 1:1 shortcut the
                                        way pT rank is)
    """
    _, _, _, _, pt, eta, phi, m = _extract_kinematics(four_momenta)
    B, J = pt.shape

    log_pt = torch.log(pt + 1e-6)
    log_m_over_pt = torch.log(m / pt + 1e-6)

    # Nearest-neighbour geometry, ignoring self and invalid jets.
    deta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
    dphi = _wrap_dphi(phi.unsqueeze(-1) - phi.unsqueeze(-2))
    dr = torch.sqrt(deta * deta + dphi * dphi + 1e-8)                       # (B, J, J)

    big = torch.full_like(dr, 1e3)
    eye = torch.eye(J, dtype=torch.bool, device=dr.device).unsqueeze(0).expand(B, J, J)
    other_valid = valid_mask.unsqueeze(-2).expand(B, J, J) & (~eye)         # (B, J, J)
    dr_other = torch.where(other_valid, dr, big)

    min_dr, nn_idx = dr_other.min(dim=-1)                                   # (B, J)
    log_min_dr = torch.log(min_dr.clamp(min=1e-6) + 1e-6)

    # Gather nearest-neighbour momentum for kT & m_nn
    pt_nn = torch.gather(pt, -1, nn_idx)                                    # (B, J)
    pt_soft = torch.minimum(pt, pt_nn)
    kt = pt_soft * min_dr.clamp(min=1e-6)
    log_kt = torch.log(kt + 1e-6)

    # m_nn: pair invariant mass to nearest neighbour
    nn_idx_4 = nn_idx.unsqueeze(-1).expand(B, J, 4)
    p_nn = torch.gather(four_momenta, 1, nn_idx_4)                          # (B, J, 4)
    m_nn = _pair_invariant_mass(four_momenta, p_nn)
    ht = pt.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    log_m_nn = torch.log(m_nn / ht + 1e-6)

    # Recoil angle — is this jet pointing opposite the sum of the other jets?
    # Strong cue for ISR in the *easy* direction, but a hard-case test too.
    total_px = four_momenta[..., 1].sum(dim=-1, keepdim=True)               # (B, 1)
    total_py = four_momenta[..., 2].sum(dim=-1, keepdim=True)
    other_px = total_px - four_momenta[..., 1]
    other_py = total_py - four_momenta[..., 2]
    phi_other = torch.atan2(other_py, other_px)
    dphi_recoil = torch.abs(_wrap_dphi(phi - (phi_other + torch.pi)))       # 0 if anti-aligned

    feats = torch.stack(
        [
            log_pt,
            eta,
            torch.sin(phi),
            torch.cos(phi),
            log_m_over_pt,
            valid_mask.float(),
            log_min_dr,
            log_kt,
            log_m_nn,
            dphi_recoil,
        ],
        dim=-1,
    )

    # Zero-out features for invalid jets (keep channel 5 = mask untouched).
    mask_f = valid_mask.float().unsqueeze(-1)
    feats_zeroed = feats * mask_f
    feats_zeroed[..., 5] = valid_mask.float()
    return feats_zeroed


# --------------------------------------------------------------------------- #
# Triplet features (physics per 3-jet candidate)
# --------------------------------------------------------------------------- #


def compute_triplet_features(
    four_momenta: torch.Tensor, triplet_indices: torch.Tensor
) -> torch.Tensor:
    """Per-triplet physics features, shape (B, T, 6).

    Channels:
        0  log(m_ijk / HT + ε)         (3-body mass)
        1  ΔR_max across the 3 pairs
        2  ΔR_min across the 3 pairs
        3  min z = min pT_soft/(pT_soft + pT_hard) across the 3 pairs
        4  log(pT_sum / HT + ε)
        5  log(E_sum / HT + ε)
    """
    B = four_momenta.shape[0]
    T = triplet_indices.shape[0]

    # Gather 4-vectors for each triplet: (B, T, 3, 4)
    tri = triplet_indices.unsqueeze(0).expand(B, T, 3)                      # (B, T, 3)
    tri_4 = tri.unsqueeze(-1).expand(B, T, 3, 4)
    jets = torch.gather(four_momenta.unsqueeze(1).expand(B, T, -1, -1), 2, tri_4)

    # 3-body 4-vector → mass
    p_sum = jets.sum(dim=2)                                                 # (B, T, 4)
    m2 = (p_sum[..., 0] ** 2 - p_sum[..., 1] ** 2 - p_sum[..., 2] ** 2 - p_sum[..., 3] ** 2).clamp(min=1e-8)
    m_ijk = torch.sqrt(m2)

    _, _, _, _, pt_ev, _, _, _ = _extract_kinematics(four_momenta)
    ht = pt_ev.sum(dim=-1, keepdim=True).clamp(min=1e-6)                    # (B, 1)

    _, _, _, _, pt_t, eta_t, phi_t, _ = _extract_kinematics(jets)           # (B, T, 3)
    E_t = jets[..., 0]

    pairs = [(0, 1), (0, 2), (1, 2)]
    dr_list, z_list = [], []
    for i, j in pairs:
        deta = eta_t[..., i] - eta_t[..., j]
        dphi = _wrap_dphi(phi_t[..., i] - phi_t[..., j])
        dr = torch.sqrt(deta * deta + dphi * dphi + 1e-8)
        dr_list.append(dr)

        pt_i, pt_j = pt_t[..., i], pt_t[..., j]
        pt_soft = torch.minimum(pt_i, pt_j)
        z = pt_soft / (pt_i + pt_j).clamp(min=1e-6)
        z_list.append(z)

    dr_stack = torch.stack(dr_list, dim=-1)                                 # (B, T, 3)
    z_stack = torch.stack(z_list, dim=-1)

    feats = torch.stack(
        [
            torch.log(m_ijk / ht + 1e-6),
            torch.log(dr_stack.max(dim=-1).values + 1e-6),
            torch.log(dr_stack.min(dim=-1).values + 1e-6),
            z_stack.min(dim=-1).values,
            torch.log(pt_t.sum(dim=-1) / ht + 1e-6),
            torch.log(E_t.sum(dim=-1) / ht + 1e-6),
        ],
        dim=-1,
    )
    return feats                                                            # (B, T, 6)


# --------------------------------------------------------------------------- #
# Per-group intra-group features (Lund / ECF / Dalitz) — used at scoring time
# --------------------------------------------------------------------------- #


def _intra_group_features(jets_4vec: torch.Tensor) -> torch.Tensor:
    """9 QCD-discriminating features per 3-jet group.

    Input (..., 3, 4).  Output (..., 9):
        [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2, dalitz_max, dalitz_min]
    """
    E = jets_4vec[..., 0].clamp(min=1e-8)
    px = jets_4vec[..., 1]
    py = jets_4vec[..., 2]
    pz = jets_4vec[..., 3]
    pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)

    pt_max = pt.max(dim=-1).values
    pt_min = pt.min(dim=-1).values.clamp(min=1e-8)
    pt_mean = pt.mean(dim=-1).clamp(min=1e-8)
    pt_std = torch.sqrt(torch.var(pt, dim=-1, unbiased=False).clamp(min=0))
    max_pt_ratio = pt_max / pt_min
    pt_cv = pt_std / pt_mean

    eta = torch.asinh(pz / pt)
    phi = torch.atan2(py, px)

    pairs = [(0, 1), (0, 2), (1, 2)]
    z_list, kt_list, dr_list = [], [], []
    for i, j in pairs:
        pt_i, pt_j = pt[..., i], pt[..., j]
        pt_soft = torch.min(pt_i, pt_j)
        z_ij = pt_soft / (pt_i + pt_j).clamp(min=1e-8)

        deta = eta[..., i] - eta[..., j]
        dphi = _wrap_dphi(phi[..., i] - phi[..., j])
        dr = torch.sqrt(deta * deta + dphi * dphi + 1e-8)

        z_list.append(z_ij)
        kt_list.append(pt_soft * dr)
        dr_list.append(dr)

    min_z = torch.stack(z_list, dim=-1).min(dim=-1).values
    max_kt = torch.stack(kt_list, dim=-1).max(dim=-1).values

    E_sum = E.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    z_E = E / E_sum

    ecf2 = torch.zeros_like(pt_max)
    for k, (i, j) in enumerate(pairs):
        ecf2 = ecf2 + z_E[..., i] * z_E[..., j] * dr_list[k]
    ecf3 = z_E[..., 0] * z_E[..., 1] * z_E[..., 2] * dr_list[0] * dr_list[1] * dr_list[2]
    d2 = ecf3 / ecf2.clamp(min=1e-4) ** 2

    p_group = jets_4vec.sum(dim=-2)
    m2_group = (
        p_group[..., 0] ** 2 - p_group[..., 1] ** 2
        - p_group[..., 2] ** 2 - p_group[..., 3] ** 2
    ).clamp(min=1e-8)
    m_group = torch.sqrt(m2_group).clamp(min=1e-8)

    dalitz_list = []
    for i, j in pairs:
        p_ij = jets_4vec[..., i, :] + jets_4vec[..., j, :]
        m2_ij = (p_ij[..., 0] ** 2 - p_ij[..., 1] ** 2 - p_ij[..., 2] ** 2 - p_ij[..., 3] ** 2).clamp(min=1e-8)
        dalitz_list.append(torch.sqrt(m2_ij) / m_group)

    dalitz_t = torch.stack(dalitz_list, dim=-1)
    dalitz_max = dalitz_t.max(dim=-1).values
    dalitz_min = dalitz_t.min(dim=-1).values

    return torch.stack(
        [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2, dalitz_max, dalitz_min],
        dim=-1,
    )


def _inter_group_features(g1_4vec: torch.Tensor, g2_4vec: torch.Tensor) -> torch.Tensor:
    """6 symmetric inter-group features: (mass_sum, mass_asym, mass_ratio, m1, m2, ΔR)."""

    def inv_mass(p):
        m2 = p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2
        return torch.sqrt(m2.clamp(min=1e-8))

    m1 = inv_mass(g1_4vec)
    m2 = inv_mass(g2_4vec)
    mass_sum = m1 + m2
    mass_asym = torch.abs(m1 - m2) / mass_sum.clamp(min=1e-8)
    mass_ratio = torch.min(m1, m2) / torch.max(m1, m2).clamp(min=1e-8)

    def eta_phi(p):
        px, py, pz = p[..., 1], p[..., 2], p[..., 3]
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        return torch.asinh(pz / pt), torch.atan2(py, px)

    eta1, phi1 = eta_phi(g1_4vec)
    eta2, phi2 = eta_phi(g2_4vec)
    dphi = _wrap_dphi(phi1 - phi2)
    delta_r = torch.sqrt((eta1 - eta2) ** 2 + dphi * dphi)

    return torch.stack([mass_sum, mass_asym, mass_ratio, m1, m2, delta_r], dim=-1)


def _isr_system_features(
    isr_4vec: torch.Tensor, g1_4vec: torch.Tensor, g2_4vec: torch.Tensor, ht: torch.Tensor
) -> torch.Tensor:
    """5 features coupling the ISR candidate to the two signal groups.

    ht: (B, 1) HT of the whole event.  For 6-jet mode isr_4vec is all zeros;
    the features remain well defined (they just evaluate to ~0).
    """
    px, py, pz = isr_4vec[..., 1], isr_4vec[..., 2], isr_4vec[..., 3]
    pt_isr = torch.sqrt(px * px + py * py + 1e-8)
    # ht is (B, 1); pt_isr is (B, N) — broadcast over assignments.
    log_pt_frac = torch.log(pt_isr / ht + 1e-6)

    eta_isr = torch.asinh(pz / pt_isr.clamp(min=1e-8))
    abs_eta_isr = torch.abs(eta_isr)

    # ΔR from ISR candidate to each group's summed 4-vector.
    def eta_phi(p):
        px2, py2, pz2 = p[..., 1], p[..., 2], p[..., 3]
        pt2 = torch.sqrt(px2 * px2 + py2 * py2).clamp(min=1e-8)
        return torch.asinh(pz2 / pt2), torch.atan2(py2, px2)

    phi_isr = torch.atan2(py, px)
    eta_g1, phi_g1 = eta_phi(g1_4vec)
    eta_g2, phi_g2 = eta_phi(g2_4vec)

    dphi1 = _wrap_dphi(phi_isr - phi_g1)
    dphi2 = _wrap_dphi(phi_isr - phi_g2)
    dr_g1 = torch.sqrt((eta_isr - eta_g1) ** 2 + dphi1 * dphi1 + 1e-8)
    dr_g2 = torch.sqrt((eta_isr - eta_g2) ** 2 + dphi2 * dphi2 + 1e-8)
    dr_min = torch.minimum(dr_g1, dr_g2)
    dr_max = torch.maximum(dr_g1, dr_g2)

    # "recoil alignment": is the ISR candidate back-to-back with g1+g2 summed?
    px_rest = g1_4vec[..., 1] + g2_4vec[..., 1]
    py_rest = g1_4vec[..., 2] + g2_4vec[..., 2]
    phi_rest = torch.atan2(py_rest, px_rest)
    dphi_recoil = torch.abs(_wrap_dphi(phi_isr - (phi_rest + torch.pi)))

    return torch.stack([log_pt_frac, abs_eta_isr, dr_min, dr_max, dphi_recoil], dim=-1)


# --------------------------------------------------------------------------- #
# Self-attention encoder layer with additive pairwise bias (applied each layer)
# --------------------------------------------------------------------------- #


class BiasedEncoderLayer(nn.Module):
    """Pre-norm self-attention + FFN where attention accepts an additive bias.

    The bias is (B, nhead, J, J) — one channel per head — and is recomputed
    fresh from the pairwise features once per forward, then reused across
    all layers (the features are intrinsic to the event geometry).
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, J, D); attn_bias: (B*nhead, J, J); key_padding_mask: (B, J) bool (True = pad).

        Merges the bool padding mask into the float attn_bias so
        MultiheadAttention sees a single float mask (avoids the deprecated
        "mismatched attn_mask + key_padding_mask" path).
        """
        h = self.ln1(x)
        merged_mask = attn_bias
        if key_padding_mask is not None:
            B, J = key_padding_mask.shape
            H = self.nhead
            kpm = (
                key_padding_mask.to(attn_bias.dtype)
                .masked_fill(key_padding_mask, float("-inf"))
                .view(B, 1, 1, J)
                .expand(B, H, J, J)
                .reshape(B * H, J, J)
            )
            merged_mask = attn_bias + kpm
        a, _ = self.attn(
            h, h, h,
            attn_mask=merged_mask,
            need_weights=False,
        )
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# --------------------------------------------------------------------------- #
# The main model — Pair-Symmetric Slot Transformer
# --------------------------------------------------------------------------- #


class JetAssignmentTransformer(nn.Module):
    """Pair-Symmetric Slot Transformer (PSST) for 6- and 7-jet gluino assignment.

    Outputs a flat logit per combinatorial assignment (70 for 7 jets, 10 for 6).
    The forward returns a dict compatible with ``evaluate.py`` and ``train.py``:

        {
          "logits":         (B, num_assignments),
          "mass_asym_flat": (B, num_assignments),    # mass asymmetry per hypothesis
          "mass_pred":      (B, 1),                  # adversarial head output
        }
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_jets: int = 7,
        input_dim: int = 4,     # kept in signature so evaluate.py's onnx export still works
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_jets = num_jets
        self.has_isr = num_jets >= 7
        self.n_jet_feats = 10
        # input_dim is retained for backward-compat with train.py/evaluate.py;
        # this architecture always reads 4-vectors plus its own derived features.
        self.raw_4vec_dim = 4

        # --- Combinatorics buffers ---
        at = build_assignment_tensors(num_jets)
        self.register_buffer("isr_indices_assign", at["isr_indices"])        # (N,) (-1 if 6-jet)
        self.register_buffer("group1_indices", at["group1_indices"])         # (N, 3)
        self.register_buffer("group2_indices", at["group2_indices"])         # (N, 3)
        self.num_assignments = at["num_assignments"]

        tt = build_triplet_tensors(num_jets)
        self.register_buffer("triplet_indices", tt["triplet_indices"])       # (T, 3)
        self.register_buffer("jet_in_triplet", tt["jet_in_triplet"])         # (J, T)
        self.num_triplets = tt["num_triplets"]

        nm = build_neighbor_map(num_jets)
        self.register_buffer("neighbor_idx", nm["neighbor_idx"])             # (N, n_nbrs)
        self.n_neighbors = nm["n_neighbors"]

        # --- Input projection: [4-vector, 11 jet features] → d_model ---
        self.input_proj = nn.Linear(self.raw_4vec_dim + self.n_jet_feats, d_model)
        # Slot embedding is *not* tied to pT rank — it's a learned per-slot bias
        # that simply helps residual streams differentiate, with no physical
        # meaning.  Permutation symmetry is recovered in training via φ-rotation
        # + η-flip augmentation and the fact that the scorer is pair-symmetric.
        self.slot_embedding = nn.Embedding(num_jets, d_model)

        # --- Pairwise bias head: 3 pair features → per-head scalar bias ---
        # One small MLP maps (log pT_ratio, log ΔR, log m_ij/HT) to (nhead,);
        # initialised near zero so the first-layer behaviour is pure attention.
        self.pair_bias_mlp = nn.Sequential(
            nn.Linear(3, 8),
            nn.GELU(),
            nn.Linear(8, nhead),
        )
        nn.init.zeros_(self.pair_bias_mlp[-1].weight)
        nn.init.zeros_(self.pair_bias_mlp[-1].bias)

        # --- Backbone: stacked biased self-attention layers ---
        self.layers = nn.ModuleList(
            [BiasedEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )

        # --- Triplet bank: 6 physics feats → d_model, 1 Transformer layer ---
        self.triplet_proj = nn.Linear(6, d_model)
        tri_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=max(1, min(4, nhead // 2)),
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.triplet_encoder = nn.TransformerEncoder(
            tri_layer, num_layers=1, enable_nested_tensor=False
        )

        # --- Jet ← Triplet cross-attention (jet attends only to triplets it's in) ---
        self.jet_tri_attn = nn.MultiheadAttention(
            d_model, max(1, min(4, nhead // 2)), dropout=dropout, batch_first=True
        )
        self.jet_tri_ln = nn.LayerNorm(d_model)

        # --- Per-hypothesis signal-jet self-attention (pair-symmetric scorer) ---
        # 6 signal jets (3 g1 + 3 g2) attend to each other with ΔR-geometric bias;
        # then g1 and g2 are mean-pooled separately and combined symmetrically.
        self.sig_attn = nn.MultiheadAttention(
            d_model, max(1, min(4, nhead // 2)), dropout=dropout, batch_first=True
        )
        self.sig_attn_ln = nn.LayerNorm(d_model)
        self.sig_attn_bias = nn.Parameter(torch.zeros(max(1, min(4, nhead // 2))))

        # ISR MLP: refines the ISR token (or the zero-pad token for 6-jet)
        self.isr_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Assignment feature block ---
        # Per-hypothesis:  [g1+g2, g1*g2, (g1-g2)^2, isr_emb] = 4 * d_model
        # + physics:       6 inter + 9 g1 + 9 g2 + 5 ISR-system = 29 dims
        self.n_assign_physics = 6 + 9 + 9 + 5
        self.physics_norm = nn.LayerNorm(self.n_assign_physics)

        self.score_mlp = nn.Sequential(
            nn.Linear(4 * d_model + self.n_assign_physics, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # --- Optional ISR auxiliary head (BCE, per-jet "is this the ISR?") ---
        self.isr_aux_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # --- Mass-adversarial head (gradient reversal) ---
        self.gradient_reversal = GradientReversalLayer()
        self.mass_adversary = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    # ------------------------------------------------------------------ #
    # Encoding: jets → contextual embeddings (with triplet cross-attn)
    # ------------------------------------------------------------------ #

    def _valid_mask(self, four_momenta: torch.Tensor) -> torch.Tensor:
        """True for jets with any non-zero momentum component.  (B, J)."""
        return (four_momenta.abs().sum(dim=-1) > 0)

    def _build_attn_bias(
        self, four_momenta: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Produce (B*nhead, J, J) additive attention bias from pair features."""
        B, J = valid_mask.shape
        pair_feats = compute_pair_bias_features(four_momenta, valid_mask)    # (B, J, J, 3)
        bias = self.pair_bias_mlp(pair_feats)                                # (B, J, J, H)
        bias = bias.permute(0, 3, 1, 2).contiguous()                         # (B, H, J, J)
        return bias.reshape(B * self.nhead, J, J)

    def encode(self, four_momenta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (jet_emb (B, J, D), global_emb (B, D))."""
        B, J, _ = four_momenta.shape
        valid = self._valid_mask(four_momenta)                               # (B, J)
        key_pad = ~valid                                                     # True = ignore

        jet_feats = compute_jet_features(four_momenta, valid)                # (B, J, 11)
        x = torch.cat([four_momenta, jet_feats], dim=-1)
        x = self.input_proj(x)
        slots = torch.arange(J, device=four_momenta.device)
        x = x + self.slot_embedding(slots).unsqueeze(0)

        attn_bias = self._build_attn_bias(four_momenta, valid)               # (B*H, J, J)

        # Main self-attention stack (pair bias reused each layer)
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, key_padding_mask=key_pad)

        # --- Triplet bank ---
        tri_feats = compute_triplet_features(four_momenta, self.triplet_indices)   # (B, T, 6)
        t = self.triplet_proj(tri_feats)                                           # (B, T, D)
        # Mask out triplets containing at least one invalid jet.
        # jet_in_triplet is (J, T); invalid_jet_in_triplet = invalid_mask @ jet_in_triplet
        invalid = (~valid).float()                                                 # (B, J)
        tri_invalid = invalid @ self.jet_in_triplet.float()                        # (B, T)
        tri_pad = tri_invalid > 0                                                  # True = drop
        # Guard against the degenerate case where *every* triplet is invalid
        # (shouldn't happen for 6/7-jet events but keeps masking safe).
        t = self.triplet_encoder(t, src_key_padding_mask=tri_pad)                  # (B, T, D)

        # Jet ← Triplet cross-attention: each jet only attends to triplets it's in.
        # attn_mask shape for MHA: (B*Hmini, J, T) with float("-inf") where forbidden.
        n_heads_mini = self.jet_tri_attn.num_heads
        allowed = self.jet_in_triplet.to(torch.bool)                               # (J, T)
        allowed_batched = allowed.unsqueeze(0).expand(B, J, -1)                    # (B, J, T)
        allowed_batched = allowed_batched & (~tri_pad).unsqueeze(1)                # drop bad triplets
        # If a jet is invalid, block every triplet (it shouldn't query anyway).
        allowed_batched = allowed_batched & valid.unsqueeze(-1)
        # Ensure each jet has at least one allowed key so softmax is well defined;
        # for rows with no allowed entries, allow everything (its output will be
        # overwritten by the valid mask when we zero-out invalid jets below).
        none_allowed = ~allowed_batched.any(dim=-1, keepdim=True)                  # (B, J, 1)
        allowed_batched = allowed_batched | none_allowed

        mha_mask = (~allowed_batched).unsqueeze(1).expand(B, n_heads_mini, J, -1)
        mha_mask = mha_mask.reshape(B * n_heads_mini, J, -1)
        mha_mask_float = torch.zeros_like(mha_mask, dtype=x.dtype)
        mha_mask_float.masked_fill_(mha_mask, float("-inf"))

        x_q = self.jet_tri_ln(x)
        tri_update, _ = self.jet_tri_attn(x_q, t, t, attn_mask=mha_mask_float, need_weights=False)
        # Zero-out update for invalid jets so padded slots don't accumulate state.
        tri_update = tri_update * valid.unsqueeze(-1).float()
        x = x + tri_update

        # Global event summary: mean over valid jets
        w = valid.float().unsqueeze(-1)
        global_emb = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return x, global_emb

    # ------------------------------------------------------------------ #
    # Per-hypothesis scoring
    # ------------------------------------------------------------------ #

    def _score_assignments(
        self,
        jet_emb: torch.Tensor,
        four_momenta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score all assignments.  Returns (logits (B, N), mass_asym_flat (B, N))."""
        B, J, D = jet_emb.shape
        N = self.num_assignments

        # --- Gather per-assignment jet embeddings and 4-vectors ---
        # group1_indices: (N, 3)
        g1_idx = self.group1_indices.unsqueeze(0).expand(B, -1, -1)          # (B, N, 3)
        g2_idx = self.group2_indices.unsqueeze(0).expand(B, -1, -1)          # (B, N, 3)

        g1_emb = torch.gather(
            jet_emb.unsqueeze(1).expand(B, N, J, D),
            2,
            g1_idx.unsqueeze(-1).expand(B, N, 3, D),
        )                                                                    # (B, N, 3, D)
        g2_emb = torch.gather(
            jet_emb.unsqueeze(1).expand(B, N, J, D),
            2,
            g2_idx.unsqueeze(-1).expand(B, N, 3, D),
        )

        g1_4 = torch.gather(
            four_momenta.unsqueeze(1).expand(B, N, J, 4),
            2,
            g1_idx.unsqueeze(-1).expand(B, N, 3, 4),
        )                                                                    # (B, N, 3, 4)
        g2_4 = torch.gather(
            four_momenta.unsqueeze(1).expand(B, N, J, 4),
            2,
            g2_idx.unsqueeze(-1).expand(B, N, 3, 4),
        )

        g1_4vec = g1_4.sum(dim=2)                                            # (B, N, 4)
        g2_4vec = g2_4.sum(dim=2)

        # --- Intra-signal self-attention (6 tokens: g1 ⊕ g2) with ΔR bias ---
        sig_emb = torch.cat([g1_emb, g2_emb], dim=2)                         # (B, N, 6, D)
        sig_4 = torch.cat([g1_4, g2_4], dim=2)                               # (B, N, 6, 4)
        _, _, _, _, _, eta_s, phi_s, _ = _extract_kinematics(sig_4)          # (B, N, 6)
        deta = eta_s.unsqueeze(-1) - eta_s.unsqueeze(-2)
        dphi = _wrap_dphi(phi_s.unsqueeze(-1) - phi_s.unsqueeze(-2))
        dr = torch.sqrt(deta * deta + dphi * dphi + 1e-8)
        log_dr = torch.log(dr + 1e-6)                                        # (B, N, 6, 6)

        n_mini_sig = self.sig_attn.num_heads
        bias_sig = log_dr.unsqueeze(2) * self.sig_attn_bias.view(1, 1, n_mini_sig, 1, 1)
        # reshape to (B*N*n_mini, 6, 6)
        bias_sig = bias_sig.reshape(B * N * n_mini_sig, 6, 6)

        sig_flat = sig_emb.reshape(B * N, 6, D)
        sig_ln = self.sig_attn_ln(sig_flat)
        sig_out, _ = self.sig_attn(sig_ln, sig_ln, sig_ln, attn_mask=bias_sig, need_weights=False)
        sig_flat = sig_flat + sig_out                                        # (B*N, 6, D)
        sig_emb = sig_flat.reshape(B, N, 6, D)

        g1_pooled = sig_emb[:, :, :3, :].mean(dim=2)                         # (B, N, D)
        g2_pooled = sig_emb[:, :, 3:, :].mean(dim=2)

        # Pair-symmetric combiners: sum, product, squared difference
        sym_sum = g1_pooled + g2_pooled
        sym_prod = g1_pooled * g2_pooled
        sym_sqd = (g1_pooled - g2_pooled) ** 2

        # --- ISR embedding per hypothesis ---
        if self.has_isr:
            isr_idx = self.isr_indices_assign.clamp(min=0)                   # (N,)
            isr_emb = jet_emb[:, isr_idx, :]                                 # (B, N, D)
            # 6-jet entries have isr_indices == -1, clamped to 0; we need them
            # zeroed so the scorer sees a clean "no ISR" token.
            isr_valid = (self.isr_indices_assign >= 0).float().view(1, N, 1)
            isr_emb = isr_emb * isr_valid
            isr_idx_4 = isr_idx.view(1, N, 1, 1).expand(B, N, 1, 4)
            isr_4 = torch.gather(
                four_momenta.unsqueeze(1).expand(B, N, J, 4), 2, isr_idx_4
            ).squeeze(2) * isr_valid                                         # (B, N, 4)
        else:
            isr_emb = torch.zeros(B, N, D, device=jet_emb.device, dtype=jet_emb.dtype)
            isr_4 = torch.zeros(B, N, 4, device=four_momenta.device, dtype=four_momenta.dtype)

        isr_emb = self.isr_mlp(isr_emb)

        # --- Physics features ---
        inter = _inter_group_features(g1_4vec, g2_4vec)                      # (B, N, 6)
        intra_g1 = _intra_group_features(g1_4)                               # (B, N, 9)
        intra_g2 = _intra_group_features(g2_4)                               # (B, N, 9)
        ht = four_momenta[..., 1:3].pow(2).sum(-1).clamp(min=1e-12).sqrt().sum(dim=-1, keepdim=True)
        ht = ht.clamp(min=1e-6)                                              # (B, 1)
        isr_sys = _isr_system_features(isr_4, g1_4vec, g2_4vec, ht)          # (B, N, 5)

        # Pair-symmetric intra: average and abs-difference of g1 and g2 features
        intra_sum = intra_g1 + intra_g2                                      # (B, N, 9)
        intra_absdiff = (intra_g1 - intra_g2).abs()                          # (B, N, 9)
        # Store asymmetry as the second inter feature so mass_asym_flat is easy.
        mass_asym_flat = inter[..., 1]                                       # (B, N)

        physics = torch.cat([inter, intra_sum, intra_absdiff, isr_sys], dim=-1)   # (B, N, 29)
        physics = self.physics_norm(physics)

        feats = torch.cat([sym_sum, sym_prod, sym_sqd, isr_emb, physics], dim=-1)
        logits = self.score_mlp(feats).squeeze(-1)                           # (B, N)

        return logits, mass_asym_flat


    # ------------------------------------------------------------------ #
    # Auxiliary heads
    # ------------------------------------------------------------------ #

    def predict_mass(self, global_emb: torch.Tensor) -> torch.Tensor:
        reversed_emb = self.gradient_reversal(global_emb)
        return self.mass_adversary(reversed_emb)

    def isr_aux_logits(self, jet_emb: torch.Tensor) -> torch.Tensor:
        """Per-jet ISR score.  (B, J).  Only meaningful when has_isr=True."""
        return self.isr_aux_head(jet_emb).squeeze(-1)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, four_momenta: torch.Tensor) -> dict[str, torch.Tensor]:
        jet_emb, global_emb = self.encode(four_momenta)
        logits, mass_asym_flat = self._score_assignments(jet_emb, four_momenta)
        mass_pred = self.predict_mass(global_emb)

        out = {
            "logits": logits,
            "mass_asym_flat": mass_asym_flat,
            "mass_pred": mass_pred,
        }
        if self.has_isr:
            out["isr_aux_logits"] = self.isr_aux_logits(jet_emb)
        return out


# --------------------------------------------------------------------------- #
# Classical mass-asymmetry baseline (unchanged public API)
# --------------------------------------------------------------------------- #


class MassAsymmetryClassicalSolver(nn.Module):
    """Classical jet-assignment solver: argmin |m1-m2|/(m1+m2).

    Returns logits = -asymmetry so argmax → minimum asymmetry, matching the
    inference interface of :class:`JetAssignmentTransformer`.
    """

    def __init__(self, num_jets: int = 7):
        super().__init__()
        self.num_jets = num_jets

        at = build_assignment_tensors(num_jets)
        self.register_buffer("group1_indices", at["group1_indices"])
        self.register_buffer("group2_indices", at["group2_indices"])
        self.num_assignments = at["num_assignments"]

    def forward(self, four_momenta: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = four_momenta.shape[0]
        na = self.num_assignments

        fm_expanded = four_momenta.unsqueeze(1).expand(-1, na, -1, -1)

        g1_idx = (
            self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)
        )
        g1_sum = torch.gather(fm_expanded, 2, g1_idx).sum(dim=2)

        g2_idx = (
            self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)
        )
        g2_sum = torch.gather(fm_expanded, 2, g2_idx).sum(dim=2)

        m1 = compute_invariant_mass(g1_sum)
        m2 = compute_invariant_mass(g2_sum)

        asymmetry = torch.abs(m1 - m2) / (m1 + m2).clamp(min=1e-8)
        logits = -asymmetry
        return {"logits": logits}
