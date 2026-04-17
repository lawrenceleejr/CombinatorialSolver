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
"""

import torch
import torch.nn as nn

from .combinatorics import build_assignment_tensors, build_factored_tensors


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


class JetAssignmentTransformer(nn.Module):
    """Transformer encoder + factored jet assignment scorer.

    For 7+ jets (ISR mode):
      - ISR head: num_jets-way classification over jets
      - Grouping head: 10-way classification for each ISR choice
      - Combined: flat logits via log P(ISR=j) + log P(grouping=k|ISR=j)

    For 6 jets (no ISR):
      - Direct 10-way assignment scoring
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_jets: int = 7,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_jets = num_jets
        self.has_isr = num_jets >= 7

        self.input_proj = nn.Linear(4, d_model)
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

        self.n_group_physics = 6

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

    def encode_jets(self, four_momenta: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(four_momenta)
        positions = torch.arange(self.num_jets, device=four_momenta.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)
        x = self.transformer_encoder(x)
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
        dphi = phi.unsqueeze(-1) - phi.unsqueeze(-2)
        dphi = dphi - 2 * torch.pi * torch.round(dphi / (2 * torch.pi))
        dr = torch.sqrt(deta**2 + dphi**2 + 1e-8)
        eye = torch.eye(self.num_jets, device=four_momenta.device).unsqueeze(0)
        dr = dr + eye * 100.0
        min_dr = dr.min(dim=-1).values

        return torch.stack([pt_frac, abs_eta, min_dr], dim=-1)

    def _compute_isr_logits(
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor
    ) -> torch.Tensor:
        """Score each jet as ISR candidate. Returns (batch, num_jets)."""
        global_ctx = jet_embeddings.mean(dim=1, keepdim=True).expand_as(jet_embeddings)
        physics = self._isr_physics(four_momenta)
        features = torch.cat([jet_embeddings, global_ctx, physics], dim=-1)
        return self.isr_head(features).squeeze(-1)

    def _group_physics_factored(self, four_momenta: torch.Tensor) -> torch.Tensor:
        """Compute group physics for all ISR x grouping combos.

        Returns (batch, num_jets * num_groupings, 6).
        """
        batch_size = four_momenta.shape[0]
        n_combos = self.num_jets * self.num_groupings

        g1_flat = self.f_group1.reshape(-1, 3)
        g2_flat = self.f_group2.reshape(-1, 3)

        fm = four_momenta.unsqueeze(1).expand(-1, n_combos, -1, -1)
        g1_idx = g1_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)
        g2_idx = g2_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 4)

        g1_4vec = torch.gather(fm, 2, g1_idx).sum(dim=2)
        g2_4vec = torch.gather(fm, 2, g2_idx).sum(dim=2)

        return self._mass_features(g1_4vec, g2_4vec)

    @staticmethod
    def _mass_features(g1_4vec: torch.Tensor, g2_4vec: torch.Tensor) -> torch.Tensor:
        """Compute 6 physics features from two group four-vectors."""

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
        dphi = phi1 - phi2
        dphi = dphi - 2 * torch.pi * torch.round(dphi / (2 * torch.pi))
        delta_r = torch.sqrt((eta1 - eta2) ** 2 + dphi**2)

        return torch.stack([mass_sum, mass_asym, mass_ratio, m1, m2, delta_r], dim=-1)

    def _compute_grouping_logits(
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor
    ) -> torch.Tensor:
        """Score all groupings for each ISR choice.

        Returns (batch, num_jets, num_groupings).
        """
        batch_size = jet_embeddings.shape[0]
        n_combos = self.num_jets * self.num_groupings

        g1_flat = self.f_group1.reshape(-1, 3)
        g2_flat = self.f_group2.reshape(-1, 3)

        je = jet_embeddings.unsqueeze(1).expand(-1, n_combos, -1, -1)
        g1_idx = g1_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.d_model)
        g2_idx = g2_flat.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.d_model)

        g1_pooled = torch.gather(je, 2, g1_idx).sum(dim=2)
        g2_pooled = torch.gather(je, 2, g2_idx).sum(dim=2)

        sym_sum = g1_pooled + g2_pooled
        sym_prod = g1_pooled * g2_pooled

        physics = self._group_physics_factored(four_momenta)

        combined = torch.cat([sym_sum, sym_prod, physics], dim=-1)
        scores = self.grouping_scorer(combined).squeeze(-1)

        return scores.reshape(batch_size, self.num_jets, self.num_groupings)

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
    ) -> torch.Tensor:
        """Flat scoring for 6-jet mode. Returns (batch, num_assignments)."""
        batch_size = jet_embeddings.shape[0]
        na = self.num_assignments

        je = jet_embeddings.unsqueeze(1).expand(-1, na, -1, -1)
        g1_idx = self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.d_model
        )
        g2_idx = self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.d_model
        )

        g1_pooled = torch.gather(je, 2, g1_idx).sum(dim=2)
        g2_pooled = torch.gather(je, 2, g2_idx).sum(dim=2)

        sym_sum = g1_pooled + g2_pooled
        sym_prod = g1_pooled * g2_pooled

        fm = four_momenta.unsqueeze(1).expand(-1, na, -1, -1)
        g1_4idx = self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, 4
        )
        g2_4idx = self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, 4
        )
        g1_4vec = torch.gather(fm, 2, g1_4idx).sum(dim=2)
        g2_4vec = torch.gather(fm, 2, g2_4idx).sum(dim=2)

        physics = self._mass_features(g1_4vec, g2_4vec)
        combined = torch.cat([sym_sum, sym_prod, physics], dim=-1)
        return self.score_mlp(combined).squeeze(-1)

    def predict_mass(self, jet_embeddings: torch.Tensor) -> torch.Tensor:
        pooled = jet_embeddings.mean(dim=1)
        reversed_pooled = self.gradient_reversal(pooled)
        return self.mass_adversary(reversed_pooled)

    def forward(self, four_momenta: torch.Tensor) -> dict[str, torch.Tensor]:
        jet_embeddings = self.encode_jets(four_momenta)
        mass_pred = self.predict_mass(jet_embeddings)

        if self.has_isr:
            isr_logits = self._compute_isr_logits(jet_embeddings, four_momenta)
            grouping_logits = self._compute_grouping_logits(
                jet_embeddings, four_momenta
            )
            logits = self._combine_logits(isr_logits, grouping_logits)
            return {
                "logits": logits,
                "isr_logits": isr_logits,
                "grouping_logits": grouping_logits,
                "mass_pred": mass_pred,
            }
        else:
            logits = self._score_assignments_flat(jet_embeddings, four_momenta)
            return {"logits": logits, "mass_pred": mass_pred}
