"""
Transformer-based jet assignment model — Joint Assignment Transformer (JAT).

Architecture:
  1. Input augmentation: append physics-derived features (log_pT, η, sin_φ, cos_φ)
     to raw (E, px, py, pz), giving an 8-dimensional per-jet representation.
  2. Linear projection of the 8-D augmented features to d_model.
  3. Transformer encoder with self-attention over jet tokens and physics-informed
     attention biases (pT hierarchy and ΔR proximity, both per-head learnable).
     No position embedding — jets are permutation-invariant; pT rank is already
     encoded via the log_pT augmented feature and pt_bias_weight.
  4. GroupTransformer: mini-Transformer applied to each candidate 3-jet group
     to produce a permutation-equivariant group representation.
  5. Unified flat assignment scoring (N-way, N=70 for 7-jet or N=10 for 6-jet):
     For each assignment, the scorer sees:
       pair_sum  = g1_rep + g2_rep       (symmetric: swap g1↔g2 invariant)
       pair_diff = |g1_rep − g2_rep|     (symmetric; → 0 when groups are equal)
       isr_emb   = jet_emb[isr_idx]      (ISR candidate in assignment context)
       global_emb = mean(jet_emb)         (event-level summary)
       physics   = 24-dim features        (mass, ECF, Dalitz, ΔR)
     This replaces the prior factored P(ISR) × P(grouping|ISR) decomposition,
     eliminating the chicken-and-egg dependency and the teacher-forcing
     train/inference mismatch that limited accuracy to ~42%.
  6. Adversarial mass decorrelation head (gradient reversal).
  7. Physics features per assignment:
     - 6 inter-group: mass_sum, mass_asym, mass_ratio, m1, m2, ΔR(G1,G2)
     - 9 intra-group per group: pT hierarchy, Lund kT, ECF₂, ECF₃, D₂, Dalitz masses
     Total n_group_physics = 24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .combinatorics import build_assignment_tensors, build_factored_tensors
from .utils import compute_invariant_mass


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


class GroupTransformer(nn.Module):
    """Mini Transformer to pool a fixed-size set of jet embeddings.

    Applies num_layers Transformer layers over the jets in a candidate group
    and returns the mean-pooled representation, capturing intra-group angular
    structure and relative momentum ordering that sum-pooling discards.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a set of jet embeddings via Transformer + mean-readout.

        Args:
            x: (N, n_jets_in_group, d_model)

        Returns:
            (N, d_model) pooled group representation
        """
        return self.encoder(x).mean(dim=1)


class JetAssignmentTransformer(nn.Module):
    """Joint-assignment Transformer for gluino pair production.

    Directly scores all possible assignments (ISR jet + two groups of 3)
    end-to-end without factorization.

    The prior factored model (P(ISR) × P(grouping|ISR)) had three structural
    problems:
      1. Teacher-forcing train/inference mismatch: grouping head saw ground-truth
         ISR during training but had to use predicted ISR at inference time.
      2. Position embedding biased the model toward labeling the last (lowest-pT)
         jet as ISR — exactly wrong for the hard case where ISR has high pT.
      3. Sequential dependency: ISR head needed grouping quality, grouping assumed
         ISR was known — a circular approximation.

    This model solves all three by scoring all N assignments jointly, letting
    ISR identity emerge from which exclusion produces the most symmetric pair.

    Architecture:
      1. Jet encoder: Transformer over all J jets with pT-hierarchy and ΔR
         attention biases; no position embedding.
      2. GroupTransformer applied to all candidate 3-jet groups in parallel.
      3. Per-assignment symmetric pair features:
           pair_sum  = g1_rep + g2_rep     (swap-invariant centroid)
           pair_diff = |g1_rep − g2_rep|   (→ 0 for kinematically equal groups)
      4. ISR embedding: jet_emb[isr_idx] for each assignment — evaluated jointly
         with the pair quality, not as a separate sequential decision.
      5. Unified MLP scorer → N-way logits.
      6. Loss: flat N-way cross-entropy (no teacher forcing).
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        # dim_feedforward increased from 256 to 512 and dropout from 0.1 to 0.2
        # vs the original factored model: the new unified scorer MLP combines ISR,
        # grouping and physics into a single pass so needs higher capacity.
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        num_jets: int = 7,
        input_dim: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_jets = num_jets
        self.has_isr = num_jets >= 7

        # Store raw input dim so ONNX export can create a correctly-sized dummy.
        self.raw_input_dim = input_dim
        # 4 physics-derived features appended to raw 4-vector: log_pT, η, sin_φ, cos_φ
        self._n_derived = 4
        self.input_proj = nn.Linear(input_dim + self._n_derived, d_model)

        # Physics-informed attention biases (per-head, init=0 → no effect at start)
        self.pt_bias_weight = nn.Parameter(torch.zeros(nhead))
        # ΔR bias: jets from the same parton shower cluster in angular space.
        self.dr_bias_weight = nn.Parameter(torch.zeros(nhead))

        # No position embedding: jets are permutation-invariant.
        # pT rank is already encoded via the augmented log_pT feature and the
        # learnable pt_bias_weight attention bias.  A position embedding encodes
        # pT-rank as an absolute prior that biases the model toward identifying
        # the last (lowest-pT) jet as ISR — exactly wrong for the hard case
        # where ISR is not the softest jet in the event.

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Shared mini-Transformer for intra-group attention pooling.
        _GROUP_HEAD_SIZE = 32
        group_nhead = min(4, max(1, d_model // _GROUP_HEAD_SIZE))
        self.group_transformer = GroupTransformer(
            d_model=d_model,
            nhead=group_nhead,
            num_layers=1,
            dropout=dropout,
        )

        # 6 inter-group features + 9 intra-group × 2 groups = 24 per assignment
        self.n_group_physics = 24
        self.physics_norm = nn.LayerNorm(self.n_group_physics)

        # Flat assignment index tensors (N = num_assignments: 70 for 7-jet, 10 for 6-jet)
        at = build_assignment_tensors(num_jets)
        self.register_buffer("isr_indices", at["isr_indices"])       # (N,), -1 for 6-jet
        self.register_buffer("group1_indices", at["group1_indices"])  # (N, 3)
        self.register_buffer("group2_indices", at["group2_indices"])  # (N, 3)
        self.num_assignments = at["num_assignments"]

        if self.has_isr:
            ft = build_factored_tensors(num_jets)
            # flat_to_factored: maps flat assignment idx → (isr_idx, grouping_idx)
            # Used only for ISR/grouping accuracy monitoring, not for the loss.
            self.register_buffer("flat_to_factored", ft["flat_to_factored"])  # (N, 2)
            self.register_buffer("factored_to_flat", ft["factored_to_flat"])  # (7, 10)
            self.num_groupings = ft["num_groupings"]

        # Unified assignment scorer.
        # Input for each assignment:
        #   pair_sum      (d_model) — symmetric centroid of both groups
        #   pair_abs_diff (d_model) — |g1−g2|, near-0 for kinematically equal groups
        #   isr_emb       (d_model) — ISR candidate jet embedding
        #   global_emb    (d_model) — event-level mean pool
        #   physics       (n_group_physics=24) — mass, ECF, Dalitz, ΔR features
        scorer_in_dim = 4 * d_model + self.n_group_physics
        self.assignment_scorer = nn.Sequential(
            nn.Linear(scorer_in_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Adversarial mass decorrelation head (gradient reversal)
        self.gradient_reversal = GradientReversalLayer()
        self.mass_adversary = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def _compute_pt_bias(self, four_momenta: torch.Tensor) -> torch.Tensor:
        """Log pT ratio matrix: log(pT_i/pT_j). Returns (batch, J, J)."""
        px, py = four_momenta[..., 1], four_momenta[..., 2]
        pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
        log_pt = torch.log(pt)
        return log_pt.unsqueeze(-1) - log_pt.unsqueeze(-2)

    def _compute_deltaR_bias(self, four_momenta: torch.Tensor) -> torch.Tensor:
        """Pairwise ΔR matrix: sqrt(Δη² + Δφ²). Returns (batch, J, J).

        Jets from the same parton cluster in ΔR.  Adding ΔR_ij (scaled by a
        learnable per-head weight, typically negative) as an attention bias
        allows the model to directly encode jet-proximity structure.
        """
        px, py, pz = four_momenta[..., 1], four_momenta[..., 2], four_momenta[..., 3]
        pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
        eta = torch.asinh(pz / pt)
        sin_phi = py / pt
        cos_phi = px / pt
        deta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
        cos_dphi = (
            cos_phi.unsqueeze(-1) * cos_phi.unsqueeze(-2)
            + sin_phi.unsqueeze(-1) * sin_phi.unsqueeze(-2)
        ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        dphi = torch.acos(cos_dphi)
        return torch.sqrt(deta**2 + dphi**2)

    def _augment_jet_features(self, four_momenta: torch.Tensor) -> torch.Tensor:
        """Append physics-derived features to raw 4-vectors.

        Augments (E, px, py, pz) with (log_pT, η, sin_φ, cos_φ), producing an
        8-dimensional per-jet representation.  The derived features are in natural
        physical ranges and directly encode the primary ISR discriminants:
          - log_pT: log(pT/HT) after HT normalisation
          - η: pseudo-rapidity — ISR jets are on average more forward
          - sin_φ, cos_φ: angular position without periodicity artefacts
        """
        px = four_momenta[..., 1]
        py = four_momenta[..., 2]
        pz = four_momenta[..., 3]
        pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
        log_pt = torch.log(pt)
        eta = torch.asinh(pz / pt)
        sin_phi = py / pt
        cos_phi = px / pt
        derived = torch.stack([log_pt, eta, sin_phi, cos_phi], dim=-1)
        return torch.cat([four_momenta, derived], dim=-1)

    @staticmethod
    def _wrap_dphi(dphi: torch.Tensor) -> torch.Tensor:
        """Wrap Δφ into [-π, π]."""
        return dphi - 2 * torch.pi * torch.round(dphi / (2 * torch.pi))

    def encode_jets(self, four_momenta: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(self._augment_jet_features(four_momenta))
        # No position embedding added here — see __init__ for rationale.

        batch_size = four_momenta.shape[0]
        pt_bias = self._compute_pt_bias(four_momenta)
        dr_bias = self._compute_deltaR_bias(four_momenta)
        attn_bias = (
            pt_bias.unsqueeze(1) * self.pt_bias_weight.view(1, self.nhead, 1, 1)
            + dr_bias.unsqueeze(1) * self.dr_bias_weight.view(1, self.nhead, 1, 1)
        ).reshape(batch_size * self.nhead, self.num_jets, self.num_jets)

        return self.transformer_encoder(x, mask=attn_bias)

    @staticmethod
    def _intra_group_features(jets_4vec: torch.Tensor) -> torch.Tensor:
        """Compute 9 QCD-discriminating features from a 3-jet candidate group.

        Features capture pT hierarchy, Lund-plane splittings, energy correlation
        functions (ECF₂, ECF₃, D₂), and Dalitz pairwise invariant masses.

        Args:
            jets_4vec: (..., 3, 4) individual jet 4-vectors (E, px, py, pz)

        Returns:
            (..., 9) per-group features:
              [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2,
               dalitz_max_ratio, dalitz_min_ratio]
        """
        E = jets_4vec[..., 0].clamp(min=1e-8)
        px = jets_4vec[..., 1]
        py = jets_4vec[..., 2]
        pz = jets_4vec[..., 3]
        pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)

        pt_max = pt.max(dim=-1).values
        pt_min = pt.min(dim=-1).values.clamp(min=1e-8)
        pt_mean = pt.mean(dim=-1).clamp(min=1e-8)
        pt_std = torch.sqrt(torch.var(pt, dim=-1, unbiased=False).clamp(min=0))

        max_pt_ratio = pt_max / pt_min
        pt_cv = pt_std / pt_mean

        eta = torch.asinh(pz / pt)
        phi = torch.atan2(py, px)

        pairs = [(0, 1), (0, 2), (1, 2)]
        z_lund_list = []
        kt_list = []
        dr_list = []

        for i, j in pairs:
            pt_i, pt_j = pt[..., i], pt[..., j]
            pt_soft = torch.min(pt_i, pt_j)
            z_ij = pt_soft / (pt_i + pt_j).clamp(min=1e-8)

            deta = eta[..., i] - eta[..., j]
            dphi = JetAssignmentTransformer._wrap_dphi(phi[..., i] - phi[..., j])
            dr = torch.sqrt(deta**2 + dphi**2 + 1e-8)

            z_lund_list.append(z_ij)
            kt_list.append(pt_soft * dr)
            dr_list.append(dr)

        min_z = torch.stack(z_lund_list, dim=-1).min(dim=-1).values
        max_kt = torch.stack(kt_list, dim=-1).max(dim=-1).values

        E_sum = E.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        z_E = E / E_sum

        ecf2 = torch.zeros_like(pt_max)
        for k, (i, j) in enumerate(pairs):
            ecf2 = ecf2 + z_E[..., i] * z_E[..., j] * dr_list[k]

        ecf3 = (
            z_E[..., 0] * z_E[..., 1] * z_E[..., 2]
            * dr_list[0] * dr_list[1] * dr_list[2]
        )

        d2 = ecf3 / ecf2.clamp(min=1e-4) ** 2

        p_group = jets_4vec.sum(dim=-2)
        m2_group = (
            p_group[..., 0]**2 - p_group[..., 1]**2
            - p_group[..., 2]**2 - p_group[..., 3]**2
        )
        m_group = torch.sqrt(m2_group.clamp(min=1e-8))

        dalitz_list = []
        for i, j in pairs:
            p_ij = jets_4vec[..., i, :] + jets_4vec[..., j, :]
            m2_ij = (
                p_ij[..., 0]**2 - p_ij[..., 1]**2
                - p_ij[..., 2]**2 - p_ij[..., 3]**2
            )
            dalitz_list.append(torch.sqrt(m2_ij.clamp(min=1e-8)) / m_group.clamp(min=1e-8))

        dalitz_t = torch.stack(dalitz_list, dim=-1)
        dalitz_max = dalitz_t.max(dim=-1).values
        dalitz_min = dalitz_t.min(dim=-1).values

        return torch.stack(
            [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2, dalitz_max, dalitz_min],
            dim=-1,
        )

    @staticmethod
    def _mass_features(
        g1_4vec: torch.Tensor,
        g2_4vec: torch.Tensor,
        g1_jets: torch.Tensor | None = None,
        g2_jets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute physics features from two group four-vectors.

        Returns (..., 24) when individual jet 4-vectors are provided:
          - 6 inter-group features: mass_sum, mass_asym, mass_ratio, m1, m2, deltaR
          - 9 intra-group features for group 1 (pT hierarchy, Lund, ECFs, Dalitz)
          - 9 intra-group features for group 2

        Returns (..., 6) when g1_jets / g2_jets are omitted (fallback).
        """

        def inv_mass(p):
            m2 = p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2
            return torch.sqrt(torch.clamp(m2, min=1e-8))

        m1 = inv_mass(g1_4vec)
        m2 = inv_mass(g2_4vec)
        mass_sum = m1 + m2
        mass_asym = torch.abs(m1 - m2) / mass_sum.clamp(min=1e-8)
        mass_ratio = torch.min(m1, m2) / torch.max(m1, m2).clamp(min=1e-8)

        def eta_phi(p):
            px, py, pz = p[..., 1], p[..., 2], p[..., 3]
            pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
            return torch.asinh(pz / pt), torch.atan2(py, px)

        eta1, phi1 = eta_phi(g1_4vec)
        eta2, phi2 = eta_phi(g2_4vec)
        dphi = JetAssignmentTransformer._wrap_dphi(phi1 - phi2)
        delta_r = torch.sqrt((eta1 - eta2) ** 2 + dphi**2)

        inter = torch.stack([mass_sum, mass_asym, mass_ratio, m1, m2, delta_r], dim=-1)

        if g1_jets is not None and g2_jets is not None:
            intra1 = JetAssignmentTransformer._intra_group_features(g1_jets)
            intra2 = JetAssignmentTransformer._intra_group_features(g2_jets)
            return torch.cat([inter, intra1, intra2], dim=-1)   # (..., 24)

        return inter                                             # (..., 6)

    def _compute_assignment_physics(
        self, four_momenta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Physics features for all N assignments in flat ordering.

        Gathers the 3-jet 4-vectors for every candidate group across all N
        assignments simultaneously and computes the 24 physics features via
        _mass_features.

        Returns:
            physics:       (batch, N, n_group_physics=24)
            mass_asym_flat: (batch, N)  index 1 in the physics tensor
        """
        batch_size = four_momenta.shape[0]
        N = self.num_assignments

        # Expand four_momenta for gathering: (batch, N, J, 4)
        fm = four_momenta[:, :, :4].unsqueeze(1).expand(-1, N, -1, -1)
        g1_4idx = self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)
        g2_4idx = self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)

        g1_jets = torch.gather(fm, 2, g1_4idx)   # (batch, N, 3, 4)
        g2_jets = torch.gather(fm, 2, g2_4idx)
        g1_4vec = g1_jets.sum(dim=2)              # (batch, N, 4)
        g2_4vec = g2_jets.sum(dim=2)

        physics = self._mass_features(g1_4vec, g2_4vec, g1_jets, g2_jets)  # (batch, N, 24)
        mass_asym_flat = physics[..., 1]  # index 1 = mass_asym
        return physics, mass_asym_flat

    def predict_mass(self, jet_embeddings: torch.Tensor) -> torch.Tensor:
        pooled = jet_embeddings.mean(dim=1)
        reversed_pooled = self.gradient_reversal(pooled)
        return self.mass_adversary(reversed_pooled)

    def forward(self, four_momenta: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full forward pass.

        Returns a dict with:
          - ``logits``: (batch, N) score per flat assignment.
            ``logits.argmax(dim=-1)`` gives the predicted assignment.
          - ``mass_asym_flat``: (batch, N) |m1−m2|/(m1+m2) per assignment.
            Used by the optional mass-symmetry auxiliary loss.
          - ``mass_pred``: (batch, 1) adversarial mass prediction.
        """
        batch_size = four_momenta.shape[0]
        N = self.num_assignments

        # 1. Encode all jets: (batch, J, d_model)
        jet_emb = self.encode_jets(four_momenta)

        # 2. Global event embedding: mean pool over jets → (batch, d_model)
        global_emb = jet_emb.mean(dim=1)

        # 3. Group representations via GroupTransformer.
        #    Fancy-index jet embeddings for all assignments:
        #    group1_indices: (N, 3) → jet_emb[:, group1_indices, :]: (batch, N, 3, d_model)
        g1_jets_emb = jet_emb[:, self.group1_indices, :]  # (batch, N, 3, d_model)
        g2_jets_emb = jet_emb[:, self.group2_indices, :]

        #    GroupTransformer: (batch*N, 3, d_model) → (batch*N, d_model)
        g1_rep = self.group_transformer(
            g1_jets_emb.reshape(batch_size * N, 3, self.d_model)
        ).reshape(batch_size, N, self.d_model)
        g2_rep = self.group_transformer(
            g2_jets_emb.reshape(batch_size * N, 3, self.d_model)
        ).reshape(batch_size, N, self.d_model)

        # 4. Symmetric pair features (invariant to swapping the two gluino groups).
        #    pair_sum:      centroid — both groups contribute equally.
        #    pair_abs_diff: |g1−g2| → approaches 0 when groups have equal kinematics,
        #                   providing a direct measure of the gluino mass equality signal.
        pair_sum      = g1_rep + g2_rep            # (batch, N, d_model)
        pair_abs_diff = (g1_rep - g2_rep).abs()    # (batch, N, d_model)

        # 5. ISR jet embedding for each assignment.
        #    For 7-jet mode: look up the specific ISR candidate's embedding.
        #    For 6-jet mode (no ISR): use zero embedding.
        if self.has_isr:
            # isr_indices: (N,) with values 0..num_jets-1 for 7-jet assignments
            isr_emb = jet_emb[:, self.isr_indices, :]  # (batch, N, d_model)
        else:
            isr_emb = torch.zeros(
                batch_size, N, self.d_model, device=four_momenta.device
            )

        # 6. Global context broadcast to all assignments: (batch, N, d_model)
        global_expanded = global_emb.unsqueeze(1).expand(-1, N, -1)

        # 7. Physics features: (batch, N, 24) and mass_asym: (batch, N)
        physics, mass_asym_flat = self._compute_assignment_physics(four_momenta)
        physics_normed = self.physics_norm(physics)

        # 8. Score each assignment with the unified MLP.
        scorer_input = torch.cat(
            [pair_sum, pair_abs_diff, isr_emb, global_expanded, physics_normed],
            dim=-1,
        )  # (batch, N, 4*d_model + n_group_physics)
        logits = self.assignment_scorer(scorer_input).squeeze(-1)  # (batch, N)

        # 9. Adversarial mass prediction (gradient reversed for decorrelation)
        mass_pred = self.predict_mass(jet_emb)

        return {
            "logits": logits,
            "mass_asym_flat": mass_asym_flat,
            "mass_pred": mass_pred,
        }


class MassAsymmetryClassicalSolver(nn.Module):
    """Classical jet assignment solver that minimises mass asymmetry.

    For every event all combinatorial assignments are enumerated (same set
    as used by the ML model).  The invariant masses of the two candidate
    groups are computed for each assignment and the one with the smallest
    relative mass asymmetry

        A = |m1 - m2| / (m1 + m2)

    is selected.  Logits are returned as the *negative* asymmetry so that
    ``argmax(logits)`` gives the best (minimum-asymmetry) assignment —
    matching the inference interface of :class:`JetAssignmentTransformer`.

    Args:
        num_jets: Number of input jets (6 or 7).
    """

    def __init__(self, num_jets: int = 7):
        super().__init__()
        self.num_jets = num_jets

        at = build_assignment_tensors(num_jets)
        self.register_buffer("group1_indices", at["group1_indices"])
        self.register_buffer("group2_indices", at["group2_indices"])
        self.num_assignments = at["num_assignments"]

    def forward(self, four_momenta: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute negative mass-asymmetry scores for all assignments.

        Args:
            four_momenta: (batch, num_jets, 4) tensor with (E, px, py, pz).
                          Values should be in the *un-normalised* physical
                          units so that meaningful invariant masses can be
                          computed.

        Returns:
            dict with ``logits`` of shape (batch, num_assignments).
            ``logits.argmax(dim=-1)`` gives the minimum-asymmetry assignment.
        """
        batch_size = four_momenta.shape[0]
        na = self.num_assignments

        # Expand for batched gathering: (batch, na, num_jets, 4)
        fm_expanded = four_momenta.unsqueeze(1).expand(-1, na, -1, -1)

        # Gather and sum group1 four-momenta → (batch, na, 4)
        g1_idx = (
            self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)
        )
        g1_sum = torch.gather(fm_expanded, 2, g1_idx).sum(dim=2)

        # Gather and sum group2 four-momenta → (batch, na, 4)
        g2_idx = (
            self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)
        )
        g2_sum = torch.gather(fm_expanded, 2, g2_idx).sum(dim=2)

        # Invariant masses for each assignment → (batch, na)
        m1 = compute_invariant_mass(g1_sum)
        m2 = compute_invariant_mass(g2_sum)

        # Relative mass asymmetry; clamp denominator to avoid division by zero
        asymmetry = torch.abs(m1 - m2) / (m1 + m2).clamp(min=1e-8)

        # Negative asymmetry as logits: argmax selects minimum asymmetry
        logits = -asymmetry
        return {"logits": logits}
