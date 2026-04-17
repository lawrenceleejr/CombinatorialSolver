"""
Transformer-based jet assignment model with factored architecture.

Architecture:
  1. Linear projection of (E, px, py, pz) to d_model
  2. Transformer encoder with self-attention over jet tokens
  3. Factored scoring (7+ jets):
     a. ISR head: per-jet classification (which jet is ISR?)
     b. Grouping head: per-grouping scoring (how to split remaining 6 into 2x3?)
     c. Combined logits: log P(assignment) = log P(ISR) + log P(grouping|ISR)
  4. Flat scoring (6 jets): direct 10-way classification
  5. Adversarial mass decorrelation head (gradient reversal)
  6. GroupTransformer: intra-group mini-Transformer replaces sum-pooling to capture
     multi-particle angular correlations within each candidate 3-jet group
  7. Extended physics features per assignment:
     - 6 inter-group features (mass sum/asymmetry/ratio, deltaR, individual masses)
     - 9 intra-group features per group (pT hierarchy, Lund-plane kT, ECF₂/ECF₃/D₂,
       Dalitz pairwise masses) × 2 groups = 18 additional features
     Total n_group_physics = 24
"""

import torch
import torch.nn as nn

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
    """Transformer encoder + factored jet assignment scorer.

    For 7+ jets (ISR mode):
      - ISR head: num_jets-way classification over jets
      - Grouping head: 10-way classification for each ISR choice
      - Combined: flat logits via log P(ISR=j) + log P(grouping=k|ISR=j)

    For 6 jets (no ISR):
      - Direct 10-way assignment scoring

    Group pooling uses a shared GroupTransformer (mini-Transformer) rather than
    sum-pooling to preserve intra-group angular structure.

    Physics features per assignment include 6 inter-group features plus 9
    intra-group features per group (18 total), giving n_group_physics=24.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_jets: int = 7,
        input_dim: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_jets = num_jets
        self.has_isr = num_jets >= 7

        self.input_proj = nn.Linear(input_dim, d_model)
        # Per-head learnable weight for pT hierarchy attention bias; init to 0 (no effect at start)
        self.pt_bias_weight = nn.Parameter(torch.zeros(nhead))
        self.pos_embedding = nn.Embedding(num_jets, d_model)

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
        # Aim for ~32 features per head (a common effective head size), capped at 4 heads
        # to keep the group sub-network lightweight relative to the main encoder.
        # nhead must evenly divide d_model; d_model // 32 gives the target head count.
        _GROUP_HEAD_SIZE = 32
        group_nhead = min(4, max(1, d_model // _GROUP_HEAD_SIZE))
        self.group_transformer = GroupTransformer(
            d_model=d_model,
            nhead=group_nhead,
            num_layers=1,
            dropout=dropout,
        )

        # 6 inter-group features + 9 intra-group features per group × 2 groups = 24
        self.n_group_physics = 24

        # Normalize physics features before feeding to scorer MLPs.
        # The 24 features span very different scales (ratios ∈ [0,1] vs masses
        # vs angular quantities), so LayerNorm stabilises the scorer inputs.
        self.physics_norm = nn.LayerNorm(self.n_group_physics)

        if self.has_isr:
            ft = build_factored_tensors(num_jets)
            self.register_buffer("f_group1", ft["group1_indices"])
            self.register_buffer("f_group2", ft["group2_indices"])
            self.register_buffer("flat_to_factored", ft["flat_to_factored"])
            self.register_buffer("factored_to_flat", ft["factored_to_flat"])
            self.num_groupings = ft["num_groupings"]
            self.num_assignments = num_jets * ft["num_groupings"]

            self.n_isr_physics = 3
            self.isr_head = nn.Sequential(
                nn.Linear(2 * d_model + self.n_isr_physics, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

            self.grouping_scorer = nn.Sequential(
                nn.Linear(2 * d_model + self.n_group_physics, 2 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        else:
            at = build_assignment_tensors(num_jets)
            self.register_buffer("group1_indices", at["group1_indices"])
            self.register_buffer("group2_indices", at["group2_indices"])
            self.num_assignments = at["num_assignments"]

            self.score_mlp = nn.Sequential(
                nn.Linear(2 * d_model + self.n_group_physics, 2 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

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

    @staticmethod
    def _wrap_dphi(dphi: torch.Tensor) -> torch.Tensor:
        """Wrap Δφ into [-π, π]."""
        return dphi - 2 * torch.pi * torch.round(dphi / (2 * torch.pi))

    def encode_jets(self, four_momenta: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(four_momenta)
        positions = torch.arange(self.num_jets, device=four_momenta.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)

        # pT hierarchy attention bias: scale log(pT_i/pT_j) per head, add to attention logits
        batch_size = four_momenta.shape[0]
        pt_bias = self._compute_pt_bias(four_momenta)  # (batch, J, J)
        pt_bias_expanded = (
            pt_bias.unsqueeze(1) * self.pt_bias_weight.view(1, self.nhead, 1, 1)
        ).reshape(batch_size * self.nhead, self.num_jets, self.num_jets)

        x = self.transformer_encoder(x, mask=pt_bias_expanded)
        return x

    def _isr_physics(self, four_momenta: torch.Tensor) -> torch.Tensor:
        """Per-jet ISR physics features: pT fraction, |eta|, min deltaR."""
        px, py, pz = four_momenta[..., 1], four_momenta[..., 2], four_momenta[..., 3]
        pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
        ht = pt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        pt_frac = pt / ht

        eta = torch.asinh(pz / pt)
        abs_eta = torch.abs(eta)

        phi = torch.atan2(py, px)
        deta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
        dphi = self._wrap_dphi(phi.unsqueeze(-1) - phi.unsqueeze(-2))
        dr = torch.sqrt(deta**2 + dphi**2 + 1e-8)
        eye = torch.eye(self.num_jets, device=four_momenta.device).unsqueeze(0)
        dr = dr + eye * 100.0
        min_dr = dr.min(dim=-1).values

        return torch.stack([pt_frac, abs_eta, min_dr], dim=-1)

    def _compute_isr_logits(
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor
    ) -> torch.Tensor:
        """Score each jet as ISR candidate. Returns (batch, num_jets).

        Uses leave-one-out context: for each jet j, the context is the mean of
        all *other* jets' embeddings.  This gives the ISR head a clean comparison
        between each jet and the rest of the event, which is the key signal for
        identifying an outlier ISR jet.
        """
        total = jet_embeddings.sum(dim=1, keepdim=True)              # (batch, 1, d_model)
        loo_ctx = (total - jet_embeddings) / (self.num_jets - 1)     # (batch, num_jets, d_model)
        physics = self._isr_physics(four_momenta)
        features = torch.cat([jet_embeddings, loo_ctx, physics], dim=-1)
        return self.isr_head(features).squeeze(-1)

    def _group_physics_factored(self, four_momenta: torch.Tensor) -> torch.Tensor:
        """Compute group physics for all ISR x grouping combos.

        Returns (batch, num_jets * num_groupings, n_group_physics).
        """
        batch_size = four_momenta.shape[0]
        n_combos = self.num_jets * self.num_groupings

        g1_flat = self.f_group1.reshape(-1, 3)
        g2_flat = self.f_group2.reshape(-1, 3)

        fm = four_momenta[:, :, :4].unsqueeze(1).expand(-1, n_combos, -1, -1)
        g1_idx = g1_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)
        g2_idx = g2_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)

        g1_jets = torch.gather(fm, 2, g1_idx)   # (batch, n_combos, 3, 4)
        g2_jets = torch.gather(fm, 2, g2_idx)
        g1_4vec = g1_jets.sum(dim=2)
        g2_4vec = g2_jets.sum(dim=2)

        return self._mass_features(g1_4vec, g2_4vec, g1_jets, g2_jets)

    @staticmethod
    def _intra_group_features(jets_4vec: torch.Tensor) -> torch.Tensor:
        """Compute 9 QCD-discriminating features from a 3-jet candidate group.

        Features capture pT hierarchy, Lund-plane splittings, energy correlation
        functions (ECF₂, ECF₃, D₂), and Dalitz pairwise invariant masses —
        all of which distinguish QCD-like (hierarchical, collinear) topologies
        from isotropic high-mass signal decays.

        Args:
            jets_4vec: (..., 3, 4) individual jet 4-vectors (E, px, py, pz)

        Returns:
            (..., 9) per-group features:
              [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2,
               dalitz_max_ratio, dalitz_min_ratio]
        """
        E = jets_4vec[..., 0].clamp(min=1e-8)   # (..., 3) energy
        px = jets_4vec[..., 1]
        py = jets_4vec[..., 2]
        pz = jets_4vec[..., 3]
        pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)  # (..., 3)

        # --- pT hierarchy ---
        pt_max = pt.max(dim=-1).values                                  # (...,)
        pt_min = pt.min(dim=-1).values.clamp(min=1e-8)
        pt_mean = pt.mean(dim=-1).clamp(min=1e-8)
        # Use torch.var for numerical stability (two-pass, unbiased=False for 3-element groups)
        pt_std = torch.sqrt(torch.var(pt, dim=-1, unbiased=False).clamp(min=0))

        max_pt_ratio = pt_max / pt_min                                  # (...,)
        pt_cv = pt_std / pt_mean                                        # (...,)

        # --- Angular quantities ---
        eta = torch.asinh(pz / pt)                                      # (..., 3)
        phi = torch.atan2(py, px)                                       # (..., 3)

        # --- All 3 intra-group pairs ---
        pairs = [(0, 1), (0, 2), (1, 2)]
        z_lund_list = []
        kt_list = []
        dr_list = []

        for i, j in pairs:
            pt_i, pt_j = pt[..., i], pt[..., j]
            pt_soft = torch.min(pt_i, pt_j)
            # Splitting fraction z = pT_soft / (pT_soft + pT_hard) ∈ [0, 0.5]
            z_ij = pt_soft / (pt_i + pt_j).clamp(min=1e-8)

            deta = eta[..., i] - eta[..., j]
            dphi = JetAssignmentTransformer._wrap_dphi(phi[..., i] - phi[..., j])
            dr = torch.sqrt(deta**2 + dphi**2 + 1e-8)

            z_lund_list.append(z_ij)
            kt_list.append(pt_soft * dr)       # Lund-plane kT
            dr_list.append(dr)

        min_z = torch.stack(z_lund_list, dim=-1).min(dim=-1).values    # most asymmetric split
        max_kt = torch.stack(kt_list, dim=-1).max(dim=-1).values       # hardest Lund emission

        # --- Energy Correlation Functions (β=1, energy fraction z_k = E_k / E_group) ---
        E_sum = E.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        z_E = E / E_sum                                                 # (..., 3)

        ecf2 = torch.zeros_like(pt_max)
        for k, (i, j) in enumerate(pairs):
            ecf2 = ecf2 + z_E[..., i] * z_E[..., j] * dr_list[k]

        # ECF₃: single triple (0,1,2)
        ecf3 = (
            z_E[..., 0] * z_E[..., 1] * z_E[..., 2]
            * dr_list[0] * dr_list[1] * dr_list[2]
        )

        # D₂ = ECF₃ / ECF₂² — probes 2-prong vs 3-prong substructure.
        # Clamp ECF₂ before squaring to avoid underflow when jets are nearly collinear.
        d2 = ecf3 / ecf2.clamp(min=1e-4) ** 2

        # --- Dalitz pairwise invariant masses, normalized by group mass ---
        p_group = jets_4vec.sum(dim=-2)                                 # (..., 4)
        m2_group = (
            p_group[..., 0]**2 - p_group[..., 1]**2
            - p_group[..., 2]**2 - p_group[..., 3]**2
        )
        m_group = torch.sqrt(m2_group.clamp(min=1e-8))                 # (...,)

        dalitz_list = []
        for i, j in pairs:
            p_ij = jets_4vec[..., i, :] + jets_4vec[..., j, :]
            m2_ij = (
                p_ij[..., 0]**2 - p_ij[..., 1]**2
                - p_ij[..., 2]**2 - p_ij[..., 3]**2
            )
            dalitz_list.append(torch.sqrt(m2_ij.clamp(min=1e-8)) / m_group.clamp(min=1e-8))

        dalitz_t = torch.stack(dalitz_list, dim=-1)                    # (..., 3)
        dalitz_max = dalitz_t.max(dim=-1).values
        dalitz_min = dalitz_t.min(dim=-1).values

        return torch.stack(
            [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2, dalitz_max, dalitz_min],
            dim=-1,
        )                                                               # (..., 9)

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

    def _compute_grouping_logits(
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score all groupings for each ISR choice.

        Group embeddings are pooled via the shared GroupTransformer (one layer of
        self-attention over the 3 jets in each candidate group) instead of
        sum-pooling, preserving intra-group angular ordering.

        Returns:
            grouping_logits: (batch, num_jets, num_groupings)
            mass_asym_flat: (batch, num_assignments) mass asymmetry per flat assignment
        """
        batch_size = jet_embeddings.shape[0]
        n_combos = self.num_jets * self.num_groupings

        g1_flat = self.f_group1.reshape(-1, 3)
        g2_flat = self.f_group2.reshape(-1, 3)

        je = jet_embeddings.unsqueeze(1).expand(-1, n_combos, -1, -1)
        g1_idx = g1_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.d_model)
        g2_idx = g2_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.d_model)

        g1_jets_emb = torch.gather(je, 2, g1_idx)   # (batch, n_combos, 3, d_model)
        g2_jets_emb = torch.gather(je, 2, g2_idx)

        # Intra-group attention pooling via shared GroupTransformer
        g1_pooled = self.group_transformer(
            g1_jets_emb.contiguous().reshape(batch_size * n_combos, 3, self.d_model)
        ).reshape(batch_size, n_combos, self.d_model)
        g2_pooled = self.group_transformer(
            g2_jets_emb.contiguous().reshape(batch_size * n_combos, 3, self.d_model)
        ).reshape(batch_size, n_combos, self.d_model)

        sym_sum = g1_pooled + g2_pooled
        sym_prod = g1_pooled * g2_pooled

        physics = self._group_physics_factored(four_momenta)  # (batch, n_combos, n_group_physics)
        physics = self.physics_norm(physics)

        combined = torch.cat([sym_sum, sym_prod, physics], dim=-1)
        scores = self.grouping_scorer(combined).squeeze(-1)
        grouping_logits = scores.reshape(batch_size, self.num_jets, self.num_groupings)

        # Remap mass asymmetry (physics dim 1) to flat assignment ordering
        mass_asym_factored = physics[:, :, 1]  # (batch, n_combos)
        source_idx = self.flat_to_factored[:, 0] * self.num_groupings + self.flat_to_factored[:, 1]
        mass_asym_flat = mass_asym_factored[:, source_idx]  # (batch, num_assignments)

        return grouping_logits, mass_asym_flat

    def _combine_logits(
        self, isr_logits: torch.Tensor, grouping_logits: torch.Tensor
    ) -> torch.Tensor:
        """Combine ISR and grouping logits into flat assignment logits.

        Returns (batch, num_assignments) in canonical flat ordering.
        """
        batch_size = isr_logits.shape[0]
        combined = isr_logits.unsqueeze(-1) + grouping_logits
        combined_flat = combined.reshape(batch_size, -1)

        f2f = self.flat_to_factored
        source_idx = f2f[:, 0] * self.num_groupings + f2f[:, 1]
        return combined_flat[:, source_idx]

    def _score_assignments_flat(
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat scoring for 6-jet mode.

        Returns:
            logits: (batch, num_assignments)
            mass_asym_flat: (batch, num_assignments)
        """
        batch_size = jet_embeddings.shape[0]
        na = self.num_assignments

        je = jet_embeddings.unsqueeze(1).expand(-1, na, -1, -1)
        g1_idx = self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.d_model
        )
        g2_idx = self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.d_model
        )

        g1_jets_emb = torch.gather(je, 2, g1_idx)   # (batch, na, 3, d_model)
        g2_jets_emb = torch.gather(je, 2, g2_idx)

        # Intra-group attention pooling via shared GroupTransformer
        g1_pooled = self.group_transformer(
            g1_jets_emb.contiguous().reshape(batch_size * na, 3, self.d_model)
        ).reshape(batch_size, na, self.d_model)
        g2_pooled = self.group_transformer(
            g2_jets_emb.contiguous().reshape(batch_size * na, 3, self.d_model)
        ).reshape(batch_size, na, self.d_model)

        sym_sum = g1_pooled + g2_pooled
        sym_prod = g1_pooled * g2_pooled

        fm = four_momenta[:, :, :4].unsqueeze(1).expand(-1, na, -1, -1)
        g1_4idx = self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, 4
        )
        g2_4idx = self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, 4
        )
        g1_jets = torch.gather(fm, 2, g1_4idx)      # (batch, na, 3, 4)
        g2_jets = torch.gather(fm, 2, g2_4idx)
        g1_4vec = g1_jets.sum(dim=2)
        g2_4vec = g2_jets.sum(dim=2)

        physics = self._mass_features(g1_4vec, g2_4vec, g1_jets, g2_jets)
        physics = self.physics_norm(physics)
        combined = torch.cat([sym_sum, sym_prod, physics], dim=-1)
        logits = self.score_mlp(combined).squeeze(-1)
        mass_asym_flat = physics[..., 1]             # index 1 = mass_asym
        return logits, mass_asym_flat

    def predict_mass(self, jet_embeddings: torch.Tensor) -> torch.Tensor:
        pooled = jet_embeddings.mean(dim=1)
        reversed_pooled = self.gradient_reversal(pooled)
        return self.mass_adversary(reversed_pooled)

    def forward(self, four_momenta: torch.Tensor) -> dict[str, torch.Tensor]:
        jet_embeddings = self.encode_jets(four_momenta)
        mass_pred = self.predict_mass(jet_embeddings)
        if self.has_isr:
            isr_logits = self._compute_isr_logits(jet_embeddings, four_momenta)
            grouping_logits, mass_asym_flat = self._compute_grouping_logits(
                jet_embeddings, four_momenta
            )
            logits = self._combine_logits(isr_logits, grouping_logits)
            return {
                "logits": logits,
                "isr_logits": isr_logits,
                "grouping_logits": grouping_logits,
                "mass_asym_flat": mass_asym_flat,
                "mass_pred": mass_pred,
            }
        else:
            logits, mass_asym_flat = self._score_assignments_flat(jet_embeddings, four_momenta)
            return {"logits": logits, "mass_asym_flat": mass_asym_flat, "mass_pred": mass_pred}


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
