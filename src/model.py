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
     multi-particle angular correlations within each candidate 3-jet group
  7. Extended physics features per assignment:
     - 7 inter-group features (mass sum/asymmetry/ratio, deltaR, individual
       masses, |cos θ*| production angle)
     - 11 intra-group features per group (pT hierarchy, Lund-plane kT,
       ECF₂/ECF₃/D₂, Dalitz pairwise masses, rest-frame Dalitz energy
       fractions) × 2 groups = 22 additional features
     Total n_group_physics = 29
  8. Pairwise-interaction attention bias (ln ΔR, ln kT, ln z, ln m²_ij,
     ln pT_i/pT_j → learned per-head bias on every encoder layer)
  9. Event-level signal-vs-QCD head on pooled embeddings + event shapes
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


class PairwiseInteractionBias(nn.Module):
    """Learned per-head attention bias from pairwise jet physics (ParT-style).

    For every jet pair (i, j) a small MLP maps IRC-motivated pairwise features
    to one bias value per attention head, which is added to the attention
    logits of every encoder layer.  This is the interaction-matrix idea from
    the Particle Transformer (Qu, Li, Qian 2022): the strongest known
    architectural gain for jet-level transformers, because it hands the
    attention mechanism the QCD splitting variables (ln ΔR, ln kT, ln z,
    ln m²_ij) it would otherwise have to rediscover from raw four-vectors.

    Features per pair (all logs clamped to finite ranges):
      0. ln ΔR_ij            — angular separation
      1. ln kT_ij            — Lund-plane transverse momentum of the splitting
      2. ln z_ij             — soft momentum fraction min(pT)/(pT_i+pT_j)
      3. ln m²_ij            — pairwise invariant mass squared
      4. ln pT_i/pT_j        — pT hierarchy (antisymmetric; generalises the
                               previous scalar pt_bias_weight)

    The final linear layer is zero-initialised so training starts from an
    unbiased (vanilla-attention) model, exactly like the old pt_bias_weight.
    """

    N_FEATURES = 5

    def __init__(self, nhead: int, hidden: int = 32):
        super().__init__()
        self.nhead = nhead
        self.mlp = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, nhead),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def compute_features(four_momenta: torch.Tensor) -> torch.Tensor:
        """Pairwise features from (batch, J, 4) four-momenta → (batch, J, J, 5)."""
        E = four_momenta[..., 0]
        px, py, pz = four_momenta[..., 1], four_momenta[..., 2], four_momenta[..., 3]
        pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
        eta = torch.asinh(pz / pt)
        phi = torch.atan2(py, px)
        log_pt = torch.log(pt)

        deta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
        dphi = JetAssignmentTransformer.wrap_dphi(phi.unsqueeze(-1) - phi.unsqueeze(-2))
        dr = torch.sqrt(deta**2 + dphi**2 + 1e-8)

        pt_i = pt.unsqueeze(-1)
        pt_j = pt.unsqueeze(-2)
        pt_min = torch.minimum(pt_i, pt_j)
        z = pt_min / (pt_i + pt_j).clamp(min=1e-8)
        kt = pt_min * dr

        # Pairwise invariant mass squared m²_ij = (p_i + p_j)²
        e_sum = E.unsqueeze(-1) + E.unsqueeze(-2)
        px_sum = px.unsqueeze(-1) + px.unsqueeze(-2)
        py_sum = py.unsqueeze(-1) + py.unsqueeze(-2)
        pz_sum = pz.unsqueeze(-1) + pz.unsqueeze(-2)
        m2 = e_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2

        # Clamp all logs so the diagonal (i == j → ΔR = 0, m² = 0) stays finite;
        # the diagonal bias is zeroed by the caller anyway.
        ln_dr = torch.log(dr.clamp(min=1e-4))
        ln_kt = torch.log(kt.clamp(min=1e-8))
        ln_z = torch.log(z.clamp(min=1e-8))
        ln_m2 = torch.log(m2.clamp(min=1e-8))
        ln_pt_ratio = log_pt.unsqueeze(-1) - log_pt.unsqueeze(-2)

        return torch.stack([ln_dr, ln_kt, ln_z, ln_m2, ln_pt_ratio], dim=-1)

    def forward(self, four_momenta: torch.Tensor) -> torch.Tensor:
        """Return additive attention bias of shape (batch * nhead, J, J)."""
        feats = self.compute_features(four_momenta)          # (B, J, J, 5)
        bias = self.mlp(feats).permute(0, 3, 1, 2)           # (B, nhead, J, J)
        # Zero the diagonal: self-attention logits should not be biased by the
        # (clamped, unphysical) i == j features.
        num_jets = four_momenta.shape[1]
        eye = torch.eye(num_jets, device=four_momenta.device, dtype=bias.dtype)
        bias = bias * (1.0 - eye)
        batch_size = four_momenta.shape[0]
        return bias.reshape(batch_size * self.nhead, num_jets, num_jets)


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
      - Grouping head: 10-way classification for each ISR choice
      - ISR head: num_jets-way classification informed by grouping quality —
        an attention-pooled summary of each ISR candidate's grouping features
        feeds into the ISR scorer so ISR and grouping are explored jointly
      - Combined: flat logits via log P(ISR=j) + log P(grouping=k|ISR=j)

    For 6 jets (no ISR):
      - Direct 10-way assignment scoring

    Group pooling uses a shared GroupTransformer (mini-Transformer) rather than
    sum-pooling to preserve intra-group angular structure.

    Physics features per assignment include 7 inter-group features plus 11
    intra-group features per group (22 total), giving n_group_physics=29.
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

        # Raw model input stays (E, px, py, pz); four derived per-jet features
        # (log pT, log E, η, log m) are appended inside encode_jets so the
        # projection sees both the linear four-vector (exact for sums/masses)
        # and the log/angular parametrisation the physics actually lives in.
        self.raw_input_dim = input_dim
        self.n_token_features = input_dim + 4
        self.input_proj = nn.Linear(self.n_token_features, d_model)
        # Learned pairwise-interaction attention bias (ln ΔR, ln kT, ln z,
        # ln m²_ij, ln pT_i/pT_j → per-head bias); zero-initialised so it is a
        # no-op at the start of training.  Supersedes the old scalar
        # pt_bias_weight (ln pT_i/pT_j is feature 4 of the new bias).
        self.pairwise_bias = PairwiseInteractionBias(nhead)
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

        # 7 inter-group features + 11 intra-group features per group × 2 groups = 29
        self.n_group_physics = 29

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

        # Event-level signal vs QCD discriminant head.  Trained with BCE when a
        # QCD sample is provided (lambda_event), decorrelated from the
        # reconstructed average mass with a DisCo penalty (lambda_disco) so the
        # score can be cut on without sculpting the bump-hunt variable.
        # Input: mean- and max-pooled jet embeddings (2 * d_model) plus 6
        # dimensionless kinematics-only event shapes (LayerNormed): pair
        # production is back-to-back and per-hemisphere isotropic, QCD
        # multijet is planar and hierarchical.
        self.n_event_shapes = 6
        self.event_shape_norm = nn.LayerNorm(self.n_event_shapes)
        self.event_head = nn.Sequential(
            nn.Linear(2 * d_model + self.n_event_shapes, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def wrap_dphi(dphi: torch.Tensor) -> torch.Tensor:
        """Wrap Δφ into [-π, π]."""
        return dphi - 2 * torch.pi * torch.round(dphi / (2 * torch.pi))

    @staticmethod
    def _wrap_dphi(dphi: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for wrap_dphi."""
        return JetAssignmentTransformer.wrap_dphi(dphi)

    @staticmethod
    def _token_features(four_momenta: torch.Tensor) -> torch.Tensor:
        """Append derived per-jet features to the raw four-vector.

        Returns (..., 8): [E, px, py, pz, log pT, log E, η, log m].  The log
        and angular features present the encoder with the parametrisation in
        which jet physics is (approximately) linear — log-scale energies and
        rapidity — while the raw four-vector is kept because group sums and
        invariant masses are linear in it.
        """
        E = four_momenta[..., 0]
        px, py, pz = four_momenta[..., 1], four_momenta[..., 2], four_momenta[..., 3]
        pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
        eta = torch.asinh(pz / pt)
        m2 = (E**2 - px**2 - py**2 - pz**2).clamp(min=1e-8)
        derived = torch.stack(
            [torch.log(pt), torch.log(E.clamp(min=1e-8)), eta, 0.5 * torch.log(m2)],
            dim=-1,
        )
        return torch.cat([four_momenta, derived], dim=-1)

    def encode_jets(self, four_momenta: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(self._token_features(four_momenta))
        positions = torch.arange(self.num_jets, device=four_momenta.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)

        # Pairwise-interaction attention bias (ParT-style): learned per-head
        # bias from ln ΔR, ln kT, ln z, ln m²_ij, ln pT_i/pT_j — added to the
        # attention logits of every encoder layer.
        pair_bias = self.pairwise_bias(four_momenta)  # (batch * nhead, J, J)

        x = self.transformer_encoder(x, mask=pair_bias)
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
        """Compute 11 QCD-discriminating features from a 3-jet candidate group.

        Features capture pT hierarchy, Lund-plane splittings, energy correlation
        functions (ECF₂, ECF₃, D₂), Dalitz pairwise invariant masses, and
        rest-frame Dalitz energy fractions — all of which distinguish QCD-like
        (hierarchical, collinear) topologies from isotropic high-mass signal
        decays.

        The rest-frame energy fractions x_i = 2 E*_i / m_group are computed
        Lorentz-invariantly as x_i = 2 (P·p_i) / m², where P is the group
        four-momentum — no explicit boost needed.  A genuine 3-body decay
        shares energy democratically (x_i cluster near 2/3); a fake triplet
        built from QCD radiation collapses onto the Dalitz boundary (one jet
        carries x → 1).

        Args:
            jets_4vec: (..., 3, 4) individual jet 4-vectors (E, px, py, pz)

        Returns:
            (..., 11) per-group features:
              [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2,
               dalitz_max_ratio, dalitz_min_ratio, x_rest_max, x_rest_min]
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

        # --- Rest-frame Dalitz energy fractions x_i = 2 E*_i / m_group ---
        # Lorentz-invariant form: E*_i = (P · p_i) / m_group with P the group
        # four-momentum, so x_i = 2 (P · p_i) / m².  Σ x_i = 2 exactly.
        dot = (
            p_group[..., 0:1] * E
            - p_group[..., 1:2] * px
            - p_group[..., 2:3] * py
            - p_group[..., 3:4] * pz
        )                                                               # (..., 3)
        x_rest = 2.0 * dot / m2_group.clamp(min=1e-8).unsqueeze(-1)    # (..., 3)
        x_rest_max = x_rest.max(dim=-1).values
        x_rest_min = x_rest.min(dim=-1).values

        return torch.stack(
            [max_pt_ratio, pt_cv, min_z, max_kt, ecf2, ecf3, d2, dalitz_max, dalitz_min,
             x_rest_max, x_rest_min],
            dim=-1,
        )                                                               # (..., 11)

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

        Returns (..., 29) when individual jet 4-vectors are provided:
          - 7 inter-group features: mass_sum, mass_asym, mass_ratio, m1, m2,
            deltaR, |cos θ*|
          - 11 intra-group features for group 1 (pT hierarchy, Lund, ECFs,
            Dalitz, rest-frame energy fractions)
          - 11 intra-group features for group 2

        Returns (..., 7) when g1_jets / g2_jets are omitted (fallback).

        |cos θ*| = |tanh(Δy/2)| is the production angle of the parent
        candidates in their partonic centre-of-mass frame: pair production of
        heavy states is central (flat-ish in cos θ*), while QCD dijet-like
        configurations are forward-peaked (t-channel gluon exchange) — a
        classic resonance-search discriminant that costs one tanh.
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

        # |cos θ*| from the rapidity difference of the two parent candidates
        # (longitudinal-boost invariant by construction).
        def rapidity(p):
            E, pz = p[..., 0], p[..., 3]
            return 0.5 * torch.log(
                (E + pz).clamp(min=1e-8) / (E - pz).clamp(min=1e-8)
            )

        cos_theta_star = torch.tanh(0.5 * (rapidity(g1_4vec) - rapidity(g2_4vec))).abs()

        inter = torch.stack(
            [mass_sum, mass_asym, mass_ratio, m1, m2, delta_r, cos_theta_star], dim=-1
        )

        if g1_jets is not None and g2_jets is not None:
            intra1 = JetAssignmentTransformer.intra_group_features(g1_jets)
            intra2 = JetAssignmentTransformer.intra_group_features(g2_jets)
            return torch.cat([inter, intra1, intra2], dim=-1)   # (..., 29)

        return inter                                             # (..., 7)

    def _compute_grouping_logits(
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score all groupings for each ISR choice.

        Group embeddings are pooled via the shared GroupTransformer (one layer of
        self-attention over the 3 jets in each candidate group) instead of
        sum-pooling, preserving intra-group angular ordering.

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
        sym_diff = (g1_pooled - g2_pooled).abs()

        physics = self._group_physics_factored(four_momenta)  # (batch, n_combos, n_group_physics)

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

        Returns:
            logits: (batch, num_assignments)
            mass_asym_flat: (batch, num_assignments)
            mass_sum_flat: (batch, num_assignments)
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
        sym_diff = (g1_pooled - g2_pooled).abs()

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

    @staticmethod
    def event_shape_features(four_momenta: torch.Tensor) -> torch.Tensor:
        """Six dimensionless, HT-scale-invariant event shapes from (B, J, 4).

        [transverse sphericity, leading-jet pT fraction, pT hierarchy
        log(pT_max/pT_min), rapidity span, min ΔR, mean ΔR].  All are pure
        four-vector kinematics (no jet-internal information) and carry the
        global-topology signal that separates back-to-back pair production
        from planar, hierarchical QCD multijet events.  The transverse
        sphericity uses the closed-form eigenvalues of the 2×2 transverse
        momentum tensor (ONNX-friendly; no eigensolver).
        """
        E = four_momenta[..., 0]
        px, py, pz = four_momenta[..., 1], four_momenta[..., 2], four_momenta[..., 3]
        pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
        ht = pt.sum(dim=-1).clamp(min=1e-8)

        # Transverse sphericity S_T = 2 λ2 / (λ1 + λ2) of Σ p_i p_i^T (2×2).
        sxx = (px * px).sum(dim=-1)
        syy = (py * py).sum(dim=-1)
        sxy = (px * py).sum(dim=-1)
        trace = (sxx + syy).clamp(min=1e-8)
        disc = torch.sqrt(((sxx - syy) ** 2 + 4.0 * sxy**2).clamp(min=0.0))
        s_t = (trace - disc) / trace   # = 2 λ2 / (λ1 + λ2) ∈ [0, 1]

        f_lead = pt.max(dim=-1).values / ht
        hierarchy = torch.log(
            pt.max(dim=-1).values / pt.min(dim=-1).values.clamp(min=1e-8)
        ).clamp(max=10.0)

        # Rapidity span (longitudinal-boost invariant).
        y = 0.5 * torch.log((E + pz).clamp(min=1e-8) / (E - pz).clamp(min=1e-8))
        y_span = y.max(dim=-1).values - y.min(dim=-1).values

        eta = torch.asinh(pz / pt)
        phi = torch.atan2(py, px)
        deta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
        dphi = JetAssignmentTransformer.wrap_dphi(phi.unsqueeze(-1) - phi.unsqueeze(-2))
        dr = torch.sqrt(deta**2 + dphi**2 + 1e-8)
        num_jets = four_momenta.shape[1]
        eye = torch.eye(num_jets, device=four_momenta.device) * 100.0
        min_dr = (dr + eye).min(dim=-1).values.min(dim=-1).values
        n_pairs = num_jets * (num_jets - 1)
        mean_dr = (dr * (1.0 - torch.eye(num_jets, device=four_momenta.device))).sum(
            dim=(-1, -2)
        ) / n_pairs

        return torch.stack([s_t, f_lead, hierarchy, y_span, min_dr, mean_dr], dim=-1)

    def predict_event_logit(
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor
    ) -> torch.Tensor:
        """Event-level signal-vs-QCD logit from pooled embeddings + event shapes."""
        shapes = self.event_shape_norm(self.event_shape_features(four_momenta))
        pooled = torch.cat(
            [jet_embeddings.mean(dim=1), jet_embeddings.max(dim=1).values, shapes],
            dim=-1,
        )
        return self.event_head(pooled)

    def forward(self, four_momenta: torch.Tensor) -> dict[str, torch.Tensor]:
        jet_embeddings = self.encode_jets(four_momenta)
        mass_pred = self.predict_mass(jet_embeddings)
        event_logit = self.predict_event_logit(jet_embeddings, four_momenta)
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
                "event_logit": event_logit,
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
                "event_logit": event_logit,
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
    #  dalitz_max_ratio, dalitz_min_ratio, x_rest_max, x_rest_min]
    MAX_PT_RATIO_IDX = 0
    PT_CV_IDX = 1
    DALITZ_MAX_RATIO_IDX = 7
    DALITZ_MIN_RATIO_IDX = 8
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

        # Dalitz-like inter-group consistency
        dalitz_penalty = (
            torch.abs(
                intra1[..., self.DALITZ_MAX_RATIO_IDX] - intra2[..., self.DALITZ_MAX_RATIO_IDX]
            )
            + torch.abs(
                intra1[..., self.DALITZ_MIN_RATIO_IDX] - intra2[..., self.DALITZ_MIN_RATIO_IDX]
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
