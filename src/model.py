"""
Transformer-based jet assignment model with factored architecture.

Architecture:
  1. Linear projection of (E, px, py, pz) to d_model
  2. Transformer encoder with self-attention over jet tokens
  3. Factored scoring (7+ jets):
     a. Grouping head: per-grouping scoring (how to split remaining 6 into 2x3?)
     b. ISR head: per-jet classification informed by grouping quality — for each
        ISR candidate, an attention-weighted summary of its grouping features is
        fed into the ISR scorer, creating interplay between ISR identification and
        combinatorial assignment (if removing jet j yields groupings that look
        like pair production, jet j is more likely ISR)
     c. Combined logits: log P(assignment) = log P(ISR) + log P(grouping|ISR)
  4. Flat scoring (6 jets): direct 10-way classification
  5. Adversarial mass decorrelation head (gradient reversal)
  6. GroupTransformer: intra-group mini-Transformer replaces sum-pooling to capture
     multi-particle angular correlations within each candidate 3-jet group.
     Inspired by arXiv:2202.03772, per-group intra-group physics observables
     (10 features: pT hierarchy, Lund kT, ECF₂/ECF₃/D₂, all 3 Dalitz ratios) are
     All three sorted Dalitz ratios (min, mid, max normalised pairwise sub-pair
     masses) are included to detect on-shell resonances in cascade decays
     P→j+R→j+(jj), which produce a sharp Dalitz edge absent in direct 3-body
     decays and QCD.
     computed directly from raw kinematics and projected to a d_model-dimensional
     conditioning token that is appended to each group's jet-embedding sequence
     BEFORE the GroupTransformer self-attention.  This lets the intra-group
     attention condition on global physics context while the features themselves
     are computed late (from raw 4-momenta, not from the latent space).
  7. Extended physics features per assignment:
     - 6 inter-group features (mass sum/asymmetry/ratio, deltaR, individual masses)
     - 10 intra-group features per group (pT hierarchy, Lund-plane kT, ECF₂/ECF₃/D₂,
       all 3 Dalitz pairwise masses) × 2 groups = 20 additional features
     Total n_group_physics = 26
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
    """Mini Transformer to pool a set of jet (+ optional physics) embeddings.

    Applies num_layers Transformer layers over the tokens in a candidate group
    and returns the mean-pooled representation, capturing intra-group angular
    structure and relative momentum ordering that sum-pooling discards.

    The sequence length is variable: callers may append a physics conditioning
    token to the 3 jet-embedding tokens so that the intra-group attention can
    condition on raw-kinematics-derived observables (arXiv:2202.03772 style).
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
        """Pool a sequence of tokens via Transformer + mean-readout.

        Args:
            x: (N, seq_len, d_model)  seq_len is 3 (jets only) or 4 (jets +
               physics conditioning token)

        Returns:
            (N, d_model) pooled group representation
        """
        return self.encoder(x).mean(dim=1)


class JetAssignmentTransformer(nn.Module):
    """Transformer encoder + factored jet assignment scorer.

    For 7+ jets (ISR mode):
      - Grouping head: 10-way classification for each ISR choice
      - ISR head: num_jets-way classification informed by grouping quality —
        an attention-pooled summary of each ISR candidate's grouping features
        feeds into the ISR scorer so ISR and grouping are explored jointly
      - Combined: flat logits via log P(ISR=j) + log P(grouping=k|ISR=j)

    For 6 jets (no ISR):
      - Direct 10-way assignment scoring

    Group pooling uses a shared GroupTransformer (mini-Transformer).  Before
    each GroupTransformer call the 9 per-group intra-group physics observables
    (pT hierarchy, Lund kT, ECF₂/ECF₃/D₂, Dalitz ratios) are computed from
    raw kinematics and projected to a d_model conditioning token that is
    appended to the 3 jet-embedding tokens.  The intra-group self-attention
    therefore sees [jet₁, jet₂, jet₃, phys_cond] and can condition on
    physics-derived context while those features remain anchored to the raw
    four-momenta rather than to the latent space.

    Physics features per assignment include 6 inter-group features plus 10
    intra-group features per group (20 total), giving n_group_physics=26.
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
        group_num_layers: int = 1,
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
            num_layers=group_num_layers,
            dropout=dropout,
        )

        # Number of intra-group physics features (per group) used as conditioning.
        # These 10 features (pT hierarchy, Lund kT, ECF₂/ECF₃/D₂, all 3 Dalitz
        # ratios) are computed from raw kinematics before GroupTransformer self-
        # attention and projected to d_model so the attention can condition on them.
        # The three Dalitz ratios (min, mid, max pairwise sub-pair masses normalised
        # by triplet mass) give the full Dalitz-plane structure needed to detect
        # on-shell intermediate resonances in cascade decays P→j+R→j+(jj).
        self.n_intra_physics = 10
        self.group_physics_proj = nn.Sequential(
            nn.Linear(self.n_intra_physics, d_model),
            nn.GELU(),
        )

        # 6 inter-group features + 10 intra-group features per group × 2 groups = 26
        self.n_group_physics = 26

        # Normalize physics features before feeding to scorer MLPs.
        # The 26 features span very different scales (ratios ∈ [0,1] vs masses
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
            # Project pooled grouping features to d_model for the ISR head.
            # For each ISR candidate, attention-weighted grouping features are
            # projected to d_model and fed into the ISR scorer, enabling the
            # ISR decision to see how well the remaining jets form pair-production
            # groupings.
            self.grouping_summary_proj = nn.Sequential(
                nn.Linear(3 * d_model + self.n_group_physics, d_model),
                nn.GELU(),
            )
            self.isr_head = nn.Sequential(
                # Input: [jet_emb, global_ctx, isr_physics, grouping_summary]
                # = d_model + d_model + n_isr_physics + d_model
                nn.Linear(3 * d_model + self.n_isr_physics, 2 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

            # Learnable scale for the ISR logit contribution to the combined
            # flat assignment logits.  Initialised to 1.0 (no change from the
            # additive baseline) so the model can learn whether to amplify or
            # attenuate the ISR signal relative to the grouping signal.
            self.isr_aux_logit_scale = nn.Parameter(torch.ones(1))

            self.grouping_scorer = nn.Sequential(
                nn.Linear(3 * d_model + self.n_group_physics, 2 * d_model),
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
                nn.Linear(3 * d_model + self.n_group_physics, 2 * d_model),
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
    def wrap_dphi(dphi: torch.Tensor) -> torch.Tensor:
        """Wrap Δφ into [-π, π]."""
        return dphi - 2 * torch.pi * torch.round(dphi / (2 * torch.pi))

    @staticmethod
    def _wrap_dphi(dphi: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for wrap_dphi."""
        return JetAssignmentTransformer.wrap_dphi(dphi)

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
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor,
        grouping_context: torch.Tensor,
    ) -> torch.Tensor:
        """Score each jet as ISR candidate. Returns (batch, num_jets).

        Uses leave-one-out context: for each jet j, the context is the mean of
        all *other* jets' embeddings.  This gives the ISR head a clean comparison
        between each jet and the rest of the event, which is the key signal for
        identifying an outlier ISR jet.
        """
        # _compute_isr_logits is only called when has_isr=True (num_jets >= 7),
        # so num_jets - 1 >= 6 and division is safe.  The guard prevents a
        # confusing ZeroDivisionError if the method is ever called with num_jets=1.
        n_others = max(self.num_jets - 1, 1)
        total = jet_embeddings.sum(dim=1, keepdim=True)    # (batch, 1, d_model)
        loo_ctx = (total - jet_embeddings) / n_others      # (batch, num_jets, d_model)
        physics = self._isr_physics(four_momenta)
        features = torch.cat([jet_embeddings, loo_ctx, physics, grouping_context], dim=-1)
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
    def intra_group_features(jets_4vec: torch.Tensor) -> torch.Tensor:
        """Compute 10 QCD-discriminating features from a 3-jet candidate group.

        Features capture pT hierarchy, Lund-plane splittings, energy correlation
        functions (ECF₂, ECF₃, D₂), and all three Dalitz pairwise invariant masses —
        all of which distinguish QCD-like (hierarchical, collinear) topologies
        from isotropic high-mass signal decays.  The full set of three sorted Dalitz
        ratios (min, mid, max) is required to detect cascade decays P→j+R→j+(jj):
        the resonance sub-pair has m(jj)=m_R, creating a sharp Dalitz edge that is
        absent in both direct 3-body decays (which fill the Dalitz plane broadly) and
        QCD (which has no internal high-mass scale).  dalitz_min alone loses the
        identity of the resonance pair; having all three unambiguously exposes it.

        Args:
            jets_4vec: (..., 3, 4) individual jet 4-vectors (E, px, py, pz)

        Returns:
            (..., 10) per-group features:
              [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2,
               dalitz_max_ratio, dalitz_min_ratio, dalitz_mid_ratio]
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
            dphi = JetAssignmentTransformer.wrap_dphi(phi[..., i] - phi[..., j])
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
        # D₂ is low for cascade triplets (j2,j3 form the resonance sub-pair → 2-prong).
        d2 = ecf3 / ecf2.clamp(min=1e-4) ** 2

        # --- All 3 Dalitz pairwise invariant masses, normalized by group mass ---
        # These three ratios fully characterise the Dalitz plane.  In cascade decays
        # P→j1+R→j1+(j2+j3), one pair has m(j_i,j_j)/m_triplet ≈ m_R/m_P (the
        # resonance sub-pair), while the other two pairs take different values.
        # Providing all three sorted values (min, mid, max) lets the model unambiguously
        # identify which sub-pair is the resonance — impossible with only (min, max).
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
        dalitz_sorted, _ = dalitz_t.sort(dim=-1)                       # (..., 3) ascending
        dalitz_min = dalitz_sorted[..., 0]
        dalitz_mid = dalitz_sorted[..., 1]
        dalitz_max = dalitz_sorted[..., 2]

        return torch.stack(
            [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2,
             dalitz_max, dalitz_min, dalitz_mid],
            dim=-1,
        )                                                               # (..., 10)

    @staticmethod
    def _intra_group_features(jets_4vec: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for intra_group_features."""
        return JetAssignmentTransformer.intra_group_features(jets_4vec)

    @staticmethod
    def _mass_features(
        g1_4vec: torch.Tensor,
        g2_4vec: torch.Tensor,
        g1_jets: torch.Tensor | None = None,
        g2_jets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute physics features from two group four-vectors.

        Returns (..., 26) when individual jet 4-vectors are provided:
          - 6 inter-group features: mass_sum, mass_asym, mass_ratio, m1, m2, deltaR
          - 10 intra-group features for group 1 (pT hierarchy, Lund, ECFs, all 3 Dalitz)
          - 10 intra-group features for group 2

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
        dphi = JetAssignmentTransformer.wrap_dphi(phi1 - phi2)
        delta_r = torch.sqrt((eta1 - eta2) ** 2 + dphi**2)

        inter = torch.stack([mass_sum, mass_asym, mass_ratio, m1, m2, delta_r], dim=-1)

        if g1_jets is not None and g2_jets is not None:
            intra1 = JetAssignmentTransformer.intra_group_features(g1_jets)
            intra2 = JetAssignmentTransformer.intra_group_features(g2_jets)
            return torch.cat([inter, intra1, intra2], dim=-1)   # (..., 26)

        return inter                                             # (..., 6)

    def _compute_grouping_logits(
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score all groupings for each ISR choice.

        Per-group intra-group physics observables (9 features each) are computed
        from raw kinematics BEFORE the GroupTransformer self-attention and
        projected to a d_model conditioning token that is appended to the 3
        jet-embedding tokens.  The GroupTransformer therefore attends over
        [jet₁, jet₂, jet₃, phys_cond], allowing intra-group attention to
        condition on physics context while the features remain tied to raw
        four-momenta (arXiv:2202.03772-inspired late-physics injection).

        Also produces a per-ISR-candidate summary of grouping features via
        attention-weighted pooling (softmax over grouping scores), enabling
        downstream ISR scoring to see how well the remaining jets form
        pair-production groupings.

        Returns:
            grouping_logits: (batch, num_jets, num_groupings)
            mass_asym_flat: (batch, num_assignments) mass asymmetry per flat assignment
            mass_sum_flat: (batch, num_assignments) mass sum (m1+m2) per flat assignment
            grouping_summary: (batch, num_jets, d_model) per-ISR-candidate quality summary
        """
        batch_size = jet_embeddings.shape[0]
        n_combos = self.num_jets * self.num_groupings

        g1_flat = self.f_group1.reshape(-1, 3)
        g2_flat = self.f_group2.reshape(-1, 3)

        # ── gather jet embeddings ──────────────────────────────────────────
        je = jet_embeddings.unsqueeze(1).expand(-1, n_combos, -1, -1)
        g1_idx_emb = g1_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.d_model)
        g2_idx_emb = g2_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.d_model)

        g1_jets_emb = torch.gather(je, 2, g1_idx_emb)   # (batch, n_combos, 3, d_model)
        g2_jets_emb = torch.gather(je, 2, g2_idx_emb)

        # ── gather raw 4-momenta for physics conditioning ──────────────────
        fm = four_momenta[:, :, :4].unsqueeze(1).expand(-1, n_combos, -1, -1)
        g1_idx_4 = g1_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)
        g2_idx_4 = g2_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)

        g1_jets_4mom = torch.gather(fm, 2, g1_idx_4)    # (batch, n_combos, 3, 4)
        g2_jets_4mom = torch.gather(fm, 2, g2_idx_4)

        # ── intra physics → conditioning token (BEFORE GroupTransformer) ───
        # Shape: (batch*n_combos, 3, 4) → intra_group_features → (batch, n_combos, 9)
        g1_intra = self.intra_group_features(
            g1_jets_4mom.reshape(batch_size * n_combos, 3, 4)
        ).reshape(batch_size, n_combos, self.n_intra_physics)
        g2_intra = self.intra_group_features(
            g2_jets_4mom.reshape(batch_size * n_combos, 3, 4)
        ).reshape(batch_size, n_combos, self.n_intra_physics)

        # Project 9-dim intra physics to d_model; reshape to (batch, n_combos, 1, d_model)
        g1_phys_tok = self.group_physics_proj(
            g1_intra.reshape(batch_size * n_combos, self.n_intra_physics)
        ).reshape(batch_size, n_combos, 1, self.d_model)
        g2_phys_tok = self.group_physics_proj(
            g2_intra.reshape(batch_size * n_combos, self.n_intra_physics)
        ).reshape(batch_size, n_combos, 1, self.d_model)

        # Append physics conditioning token: [jet₁, jet₂, jet₃, phys_cond]
        g1_input = torch.cat([g1_jets_emb, g1_phys_tok], dim=2)  # (batch, n_combos, 4, d_model)
        g2_input = torch.cat([g2_jets_emb, g2_phys_tok], dim=2)

        # ── GroupTransformer over 4-token sequences ────────────────────────
        g1_pooled = self.group_transformer(
            g1_input.contiguous().reshape(batch_size * n_combos, 4, self.d_model)
        ).reshape(batch_size, n_combos, self.d_model)
        g2_pooled = self.group_transformer(
            g2_input.contiguous().reshape(batch_size * n_combos, 4, self.d_model)
        ).reshape(batch_size, n_combos, self.d_model)

        sym_sum = g1_pooled + g2_pooled
        sym_prod = g1_pooled * g2_pooled
        sym_diff = (g1_pooled - g2_pooled).abs()

        # ── full 26-feature physics (computed from already-gathered 4-momenta) ──
        # _mass_features returns (..., 26): 6 inter-group features
        # [mass_sum, mass_asym, mass_ratio, m1, m2, deltaR] followed by
        # 10 intra-group features for group 1 and 10 for group 2 (20 total).
        g1_4vec = g1_jets_4mom.sum(dim=2)   # (batch, n_combos, 4)
        g2_4vec = g2_jets_4mom.sum(dim=2)
        physics = self._mass_features(g1_4vec, g2_4vec, g1_jets_4mom, g2_jets_4mom)

        # Extract raw mass sum and asymmetry BEFORE LayerNorm so that the
        # across-assignment ranking (argmin mass_asym = classical best assignment)
        # is preserved, and so that mass_sum retains its physical scale for the
        # entropy-weighted low-mass loss.
        # _mass_features inter-group feature order: [mass_sum, mass_asym, ...]
        mass_sum_factored = physics[:, :, 0]   # (batch, n_combos)
        mass_asym_factored = physics[:, :, 1]  # (batch, n_combos)

        physics = self.physics_norm(physics)

        combined = torch.cat([sym_sum, sym_prod, sym_diff, physics], dim=-1)
        scores = self.grouping_scorer(combined).squeeze(-1)
        grouping_logits = scores.reshape(batch_size, self.num_jets, self.num_groupings)

        # Attention-pooled grouping summary per ISR candidate:
        # softmax over the 10 grouping scores weights the grouping features,
        # then project to d_model. This tells the ISR head how good the
        # pair-production interpretations look when each jet is removed.
        combined_per_isr = combined.reshape(batch_size, self.num_jets, self.num_groupings, -1)
        grp_weights = grouping_logits.softmax(dim=-1).unsqueeze(-1)     # (batch, J, 10, 1)
        grp_context = (grp_weights * combined_per_isr).sum(dim=2)       # (batch, J, 2*d+phys)
        grouping_summary = self.grouping_summary_proj(grp_context)      # (batch, J, d_model)
        source_idx = self.flat_to_factored[:, 0] * self.num_groupings + self.flat_to_factored[:, 1]
        mass_asym_flat = mass_asym_factored[:, source_idx]  # (batch, num_assignments)
        mass_sum_flat = mass_sum_factored[:, source_idx]    # (batch, num_assignments)

        return grouping_logits, mass_asym_flat, mass_sum_flat, grouping_summary

    def _combine_logits(
        self, isr_logits: torch.Tensor, grouping_logits: torch.Tensor
    ) -> torch.Tensor:
        """Combine ISR and grouping logits into flat assignment logits.

        Returns (batch, num_assignments) in canonical flat ordering.
        """
        batch_size = isr_logits.shape[0]
        combined = self.isr_aux_logit_scale * isr_logits.unsqueeze(-1) + grouping_logits
        combined_flat = combined.reshape(batch_size, -1)

        f2f = self.flat_to_factored
        source_idx = f2f[:, 0] * self.num_groupings + f2f[:, 1]
        return combined_flat[:, source_idx]

    def _score_assignments_flat(
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat scoring for 6-jet mode.

        Per-group intra-group physics observables are injected as a conditioning
        token into the GroupTransformer sequence (same mechanism as
        _compute_grouping_logits) so that intra-group attention can condition
        on raw-kinematics-derived physics context.

        Returns:
            logits: (batch, num_assignments)
            mass_asym_flat: (batch, num_assignments)
            mass_sum_flat: (batch, num_assignments)
        """
        batch_size = jet_embeddings.shape[0]
        na = self.num_assignments

        # ── gather jet embeddings ──────────────────────────────────────────
        je = jet_embeddings.unsqueeze(1).expand(-1, na, -1, -1)
        g1_idx_emb = self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.d_model
        )
        g2_idx_emb = self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.d_model
        )

        g1_jets_emb = torch.gather(je, 2, g1_idx_emb)   # (batch, na, 3, d_model)
        g2_jets_emb = torch.gather(je, 2, g2_idx_emb)

        # ── gather raw 4-momenta ──────────────────────────────────────────
        fm = four_momenta[:, :, :4].unsqueeze(1).expand(-1, na, -1, -1)
        g1_4idx = self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, 4
        )
        g2_4idx = self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, 4
        )
        g1_jets = torch.gather(fm, 2, g1_4idx)      # (batch, na, 3, 4)
        g2_jets = torch.gather(fm, 2, g2_4idx)

        # ── intra physics → conditioning token (BEFORE GroupTransformer) ───
        g1_intra = self.intra_group_features(
            g1_jets.reshape(batch_size * na, 3, 4)
        ).reshape(batch_size, na, self.n_intra_physics)
        g2_intra = self.intra_group_features(
            g2_jets.reshape(batch_size * na, 3, 4)
        ).reshape(batch_size, na, self.n_intra_physics)

        g1_phys_tok = self.group_physics_proj(
            g1_intra.reshape(batch_size * na, self.n_intra_physics)
        ).reshape(batch_size, na, 1, self.d_model)
        g2_phys_tok = self.group_physics_proj(
            g2_intra.reshape(batch_size * na, self.n_intra_physics)
        ).reshape(batch_size, na, 1, self.d_model)

        # Append physics conditioning token: [jet₁, jet₂, jet₃, phys_cond]
        g1_input = torch.cat([g1_jets_emb, g1_phys_tok], dim=2)  # (batch, na, 4, d_model)
        g2_input = torch.cat([g2_jets_emb, g2_phys_tok], dim=2)

        # ── GroupTransformer over 4-token sequences ────────────────────────
        g1_pooled = self.group_transformer(
            g1_input.contiguous().reshape(batch_size * na, 4, self.d_model)
        ).reshape(batch_size, na, self.d_model)
        g2_pooled = self.group_transformer(
            g2_input.contiguous().reshape(batch_size * na, 4, self.d_model)
        ).reshape(batch_size, na, self.d_model)

        sym_sum = g1_pooled + g2_pooled
        sym_prod = g1_pooled * g2_pooled
        sym_diff = (g1_pooled - g2_pooled).abs()

        g1_4vec = g1_jets.sum(dim=2)
        g2_4vec = g2_jets.sum(dim=2)

        physics = self._mass_features(g1_4vec, g2_4vec, g1_jets, g2_jets)
        # Extract raw mass sum and asymmetry BEFORE LayerNorm so that the
        # across-assignment ranking is preserved and mass_sum retains physical scale.
        # _mass_features inter-group feature order: [mass_sum, mass_asym, ...]
        mass_sum_flat = physics[..., 0]              # index 0 = mass_sum
        mass_asym_flat = physics[..., 1]             # index 1 = mass_asym
        physics = self.physics_norm(physics)
        combined = torch.cat([sym_sum, sym_prod, sym_diff, physics], dim=-1)
        logits = self.score_mlp(combined).squeeze(-1)
        return logits, mass_asym_flat, mass_sum_flat

    def predict_mass(self, jet_embeddings: torch.Tensor) -> torch.Tensor:
        pooled = jet_embeddings.mean(dim=1)
        reversed_pooled = self.gradient_reversal(pooled)
        return self.mass_adversary(reversed_pooled)

    def forward(self, four_momenta: torch.Tensor) -> dict[str, torch.Tensor]:
        jet_embeddings = self.encode_jets(four_momenta)
        mass_pred = self.predict_mass(jet_embeddings)
        if self.has_isr:
            # Compute groupings first so the ISR head can see grouping quality
            grouping_logits, mass_asym_flat, mass_sum_flat, grp_summary = (
                self._compute_grouping_logits(jet_embeddings, four_momenta)
            )
            isr_logits = self._compute_isr_logits(
                jet_embeddings, four_momenta, grp_summary
            )
            logits = self._combine_logits(isr_logits, grouping_logits)
            return {
                "logits": logits,
                "isr_logits": isr_logits,
                "grouping_logits": grouping_logits,
                "mass_asym_flat": mass_asym_flat,
                "mass_sum_flat": mass_sum_flat,
                "mass_pred": mass_pred,
            }
        else:
            logits, mass_asym_flat, mass_sum_flat = self._score_assignments_flat(
                jet_embeddings, four_momenta
            )
            return {
                "logits": logits,
                "mass_asym_flat": mass_asym_flat,
                "mass_sum_flat": mass_sum_flat,
                "mass_pred": mass_pred,
            }


class MassAsymmetryClassicalSolver(nn.Module):
    """Classical jet assignment solver with mass-difference-first ranking.

    For every event all combinatorial assignments are enumerated (same set
    as used by the ML model).  The invariant masses of the two candidate
    groups are computed for each assignment and scored in two stages:

      1) Primary classical objective: minimise absolute mass difference
           D = |m1 - m2|
      2) Secondary refinements: use additional physics-inspired features
         (pT hierarchy, angular geometry, Dalitz-like balance, and opening-
         angle/pT consistency scaled to sqrt(s)=13 TeV) as a small tie-breaker

    Logits are returned as the negative staged score so that ``argmax(logits)``
    gives the best assignment —
    matching the inference interface of :class:`JetAssignmentTransformer`.

    Args:
        num_jets: Number of input jets (6 or 7).
    """
    # LHC proton-proton center-of-mass energy in GeV (sqrt(s)=13 TeV = 13000 GeV).
    COM_ENERGY_GEV = 13000.0
    # Theoretical maximum of E_total/sqrt(s): 1.0 when all COM energy is captured.
    ENERGY_FRACTION_BASELINE = 1.0
    # Unit offset keeps opening-angle scaling active even at low energy fraction.
    OPENING_SCALE_OFFSET = 1.0
    # Small ΔR contribution to angular penalty (secondary to Δφ back-to-backness).
    DELTA_R_WEIGHT = 0.1
    # Indices match intra_group_features return order:
    # [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2,
    #  dalitz_max_ratio, dalitz_min_ratio, dalitz_mid_ratio]
    MAX_PT_RATIO_IDX = 0
    PT_CV_IDX = 1
    DALITZ_MAX_RATIO_IDX = 7
    DALITZ_MIN_RATIO_IDX = 8
    DALITZ_MID_RATIO_IDX = 9
    # Secondary-feature blend used only for tie-breaking after primary mass difference.
    SECONDARY_WEIGHTS = {
        "asymmetry": 0.45,
        "pt_hierarchy": 0.20,
        "angular": 0.15,
        "dalitz": 0.10,
        "kine13": 0.10,
    }
    # Keeps secondary term lexicographic-like versus GeV-scale primary |m1-m2|.
    SECONDARY_TIEBREAK_SCALE = 1.0e-3

    def __init__(self, num_jets: int = 7):
        super().__init__()
        self.num_jets = num_jets

        at = build_assignment_tensors(num_jets)
        self.register_buffer("group1_indices", at["group1_indices"])
        self.register_buffer("group2_indices", at["group2_indices"])
        self.num_assignments = at["num_assignments"]

    def forward(self, four_momenta: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute staged classical scores for all assignments.

        Args:
            four_momenta: (batch, num_jets, 4) tensor with (E, px, py, pz).
                          Values should be in the *un-normalised* physical
                          units so that meaningful invariant masses can be
                          computed.

        Returns:
            dict with ``logits`` of shape (batch, num_assignments).
            ``logits.argmax(dim=-1)`` gives the assignment with minimum
            primary mass difference and best secondary physics tie-break.
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

        # Primary objective: minimize absolute parent mass difference.
        mass_diff = torch.abs(m1 - m2)

        # Secondary refinements
        # Relative mass asymmetry (classical normalization)
        asymmetry = torch.abs(m1 - m2) / (m1 + m2).clamp(min=1e-8)

        # Intra-group QCD-sensitive features (reuse model feature definitions)
        g1_jets = torch.gather(fm_expanded, 2, g1_idx)
        g2_jets = torch.gather(fm_expanded, 2, g2_idx)
        intra1 = JetAssignmentTransformer.intra_group_features(g1_jets)
        intra2 = JetAssignmentTransformer.intra_group_features(g2_jets)

        # pT hierarchy penalty (prefer less hierarchical candidate parents)
        pt_hierarchy = (
            0.5 * (
                (intra1[..., self.MAX_PT_RATIO_IDX] - 1.0)
                + (intra2[..., self.MAX_PT_RATIO_IDX] - 1.0)
            )
            + 0.5 * (intra1[..., self.PT_CV_IDX] + intra2[..., self.PT_CV_IDX])
        )

        # Angular relationships between reconstructed parent candidates
        px1, py1, pz1 = g1_sum[..., 1], g1_sum[..., 2], g1_sum[..., 3]
        px2, py2, pz2 = g2_sum[..., 1], g2_sum[..., 2], g2_sum[..., 3]
        pt1 = torch.sqrt(px1**2 + py1**2).clamp(min=1e-8)
        pt2 = torch.sqrt(px2**2 + py2**2).clamp(min=1e-8)
        eta1 = torch.asinh(pz1 / pt1)
        eta2 = torch.asinh(pz2 / pt2)
        phi1 = torch.atan2(py1, px1)
        phi2 = torch.atan2(py2, px2)
        dphi = JetAssignmentTransformer.wrap_dphi(phi1 - phi2)
        delta_r = torch.sqrt((eta1 - eta2) ** 2 + dphi**2 + 1e-8)
        angular_penalty = (
            torch.abs(torch.pi - torch.abs(dphi)) / torch.pi
            + self.DELTA_R_WEIGHT * delta_r
        )

        # Dalitz-like inter-group consistency: both groups should have the same
        # internal Dalitz structure (all three sorted pairwise mass ratios should
        # match between the two candidate parents).  Using all three ratios (max,
        # min, mid) is important for cascade decays where one Dalitz ratio is
        # pinned to m_R/m_P and must be consistent between the two groups.
        dalitz_penalty = (
            torch.abs(
                intra1[..., self.DALITZ_MAX_RATIO_IDX] - intra2[..., self.DALITZ_MAX_RATIO_IDX]
            )
            + torch.abs(
                intra1[..., self.DALITZ_MIN_RATIO_IDX] - intra2[..., self.DALITZ_MIN_RATIO_IDX]
            )
            + torch.abs(
                intra1[..., self.DALITZ_MID_RATIO_IDX] - intra2[..., self.DALITZ_MID_RATIO_IDX]
            )
        )

        # Opening-angle/pT consistency with explicit sqrt(s)=13 TeV scale
        pt_balance = torch.abs(pt1 - pt2) / (pt1 + pt2).clamp(min=1e-8)
        dphi_norm = torch.abs(dphi) / torch.pi
        # Balanced parent pT (low pt_balance) should align with back-to-back opening (high dphi_norm).
        # Expected normalized opening increases as pT balance improves (pt_balance -> 0).
        expected_dphi_norm = self.ENERGY_FRACTION_BASELINE - pt_balance
        opening_pt_consistency = torch.abs(expected_dphi_norm - dphi_norm)
        energy_fraction = (g1_sum[..., 0] + g2_sum[..., 0]) / self.COM_ENERGY_GEV
        energy_overflow = torch.relu(energy_fraction - self.ENERGY_FRACTION_BASELINE)
        kine13_penalty = (
            opening_pt_consistency
            * (self.OPENING_SCALE_OFFSET + energy_fraction.clamp(min=0.0))
            + energy_overflow
        )

        secondary_penalty = (
            self.SECONDARY_WEIGHTS["asymmetry"] * asymmetry
            + self.SECONDARY_WEIGHTS["pt_hierarchy"] * pt_hierarchy
            + self.SECONDARY_WEIGHTS["angular"] * angular_penalty
            + self.SECONDARY_WEIGHTS["dalitz"] * dalitz_penalty
            + self.SECONDARY_WEIGHTS["kine13"] * kine13_penalty
        )

        # Hard physicality guard: convert overflow fraction back to GeV scale so it
        # is comparable to the primary |m1-m2| term and can strongly reject unphysical
        # interpretations even before tiny secondary tie-break terms are applied.
        physicality_penalty = energy_overflow * self.COM_ENERGY_GEV
        # Lexicographic-style score: primary mass difference first, then refinement.
        staged_score = mass_diff + physicality_penalty + self.SECONDARY_TIEBREAK_SCALE * secondary_penalty
        logits = -staged_score
        return {"logits": logits}
