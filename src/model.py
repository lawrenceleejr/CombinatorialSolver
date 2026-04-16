"""
Transformer-based jet assignment model.

Architecture:
  1. Linear projection of (E, px, py, pz) to d_model
  2. Transformer encoder with self-attention over jet tokens
  3. Assignment scoring: enumerate all possible assignments,
     aggregate jet embeddings per group, score with MLP
  4. Physics-informed features: per-assignment invariant masses, mass
     asymmetry, and angular features computed directly from four-vectors
  5. Adversarial mass decorrelation head (gradient reversal)

Supports both 6-jet (10 assignments, no ISR) and 7-jet (70 assignments, with ISR).
"""

import torch
import torch.nn as nn

from .combinatorics import build_assignment_tensors


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""

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
    """Transformer encoder + combinatorial assignment scorer.

    Args:
        d_model: Embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Feedforward dimension in transformer.
        dropout: Dropout rate.
        num_jets: Number of input jets (6 or 7).
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

        # Input projection: 4-vector (E, px, py, pz) -> d_model
        self.input_proj = nn.Linear(4, d_model)

        # Learned positional encoding for jet positions (ordered by pT)
        self.pos_embedding = nn.Embedding(num_jets, d_model)

        # Transformer encoder
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

        # Build assignment index tensors
        at = build_assignment_tensors(num_jets)
        self.register_buffer("isr_indices", at["isr_indices"])
        self.register_buffer("group1_indices", at["group1_indices"])
        self.register_buffer("group2_indices", at["group2_indices"])
        self.num_assignments = at["num_assignments"]

        # Physics-informed features per assignment:
        # Group features (6): m(g1), m(g2), mass_asym, mass_sum, mass_ratio, deltaR
        # ISR features (4, only for 7+ jets): isr_pt_frac, isr_abs_eta,
        #   min_deltaR(isr, other jets), isr_mass_pull (how much ISR hurts mass balance)
        self.n_group_physics = 6
        self.n_isr_physics = 4 if self.has_isr else 0
        self.n_physics_features = self.n_group_physics + self.n_isr_physics

        # Assignment scoring MLP
        # For 7 jets: input is [isr_embed, sym_sum, sym_prod, physics] = 3*d_model + n_phys
        # For 6 jets: input is [sym_sum, sym_prod, physics] = 2*d_model + n_phys
        scorer_input_dim = (3 * d_model if self.has_isr else 2 * d_model) + self.n_physics_features
        self.score_mlp = nn.Sequential(
            nn.Linear(scorer_input_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Adversarial mass prediction head (with gradient reversal)
        self.gradient_reversal = GradientReversalLayer()
        self.mass_adversary = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def encode_jets(self, four_momenta: torch.Tensor) -> torch.Tensor:
        """Encode jet four-momenta through transformer.

        Args:
            four_momenta: (batch, num_jets, 4)

        Returns:
            (batch, num_jets, d_model) jet embeddings
        """
        x = self.input_proj(four_momenta)

        positions = torch.arange(self.num_jets, device=four_momenta.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)

        x = self.transformer_encoder(x)
        return x

    def _compute_physics_features(
        self, four_momenta: torch.Tensor
    ) -> torch.Tensor:
        """Compute physics-informed features for each assignment.

        Directly computes invariant masses, mass asymmetry, and angular
        features from the raw four-vectors for each candidate assignment.

        Args:
            four_momenta: (batch, num_jets, 4) in (E, px, py, pz)

        Returns:
            (batch, num_assignments, n_physics_features) tensor
        """
        batch_size = four_momenta.shape[0]
        na = self.num_assignments

        # Expand four_momenta for gathering: (batch, na, num_jets, 4)
        fm = four_momenta.unsqueeze(1).expand(-1, na, -1, -1)

        # Gather group 4-vectors
        g1_idx = self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, 4
        )
        g2_idx = self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, 4
        )
        g1_4vec = torch.gather(fm, 2, g1_idx).sum(dim=2)  # (batch, na, 4)
        g2_4vec = torch.gather(fm, 2, g2_idx).sum(dim=2)  # (batch, na, 4)

        # Invariant masses: m^2 = E^2 - px^2 - py^2 - pz^2
        def inv_mass(p):
            m2 = p[..., 0]**2 - p[..., 1]**2 - p[..., 2]**2 - p[..., 3]**2
            return torch.sqrt(torch.clamp(m2, min=1e-8))

        m1 = inv_mass(g1_4vec)  # (batch, na)
        m2 = inv_mass(g2_4vec)

        # Symmetric mass features
        mass_sum = m1 + m2
        mass_asym = torch.abs(m1 - m2) / mass_sum.clamp(min=1e-8)
        mass_min = torch.min(m1, m2)
        mass_max = torch.max(m1, m2)
        mass_ratio = mass_min / mass_max.clamp(min=1e-8)

        # Angular separation between group centroids
        # Use pseudo-rapidity and azimuthal angle of summed 4-vectors
        def eta_phi(p):
            px, py, pz = p[..., 1], p[..., 2], p[..., 3]
            pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
            eta = torch.asinh(pz / pt)
            phi = torch.atan2(py, px)
            return eta, phi

        eta1, phi1 = eta_phi(g1_4vec)
        eta2, phi2 = eta_phi(g2_4vec)
        dphi = phi1 - phi2
        # Wrap dphi to [-pi, pi]
        dphi = dphi - 2 * torch.pi * torch.round(dphi / (2 * torch.pi))
        deta = eta1 - eta2
        delta_r = torch.sqrt(deta**2 + dphi**2)

        # Group features: (batch, na, 6)
        group_features = [mass_sum, mass_asym, mass_ratio, m1, m2, delta_r]

        if self.has_isr:
            # ISR-specific features to help identify which jet is ISR
            isr_idx_expanded = self.isr_indices.unsqueeze(0).unsqueeze(-1).expand(
                batch_size, -1, 4
            )  # (batch, na, 4)
            isr_4vec = torch.gather(
                four_momenta.unsqueeze(1).expand(-1, na, -1, -1),
                2, isr_idx_expanded.unsqueeze(2)
            ).squeeze(2)  # (batch, na, 4)

            # ISR pT fraction: ISR jets tend to be softer
            isr_px, isr_py = isr_4vec[..., 1], isr_4vec[..., 2]
            isr_pt = torch.sqrt(isr_px**2 + isr_py**2).clamp(min=1e-8)
            all_px, all_py = four_momenta[..., 1], four_momenta[..., 2]
            ht = torch.sqrt(all_px**2 + all_py**2).sum(dim=-1, keepdim=True)  # (batch, 1)
            isr_pt_frac = isr_pt / ht.clamp(min=1e-8)  # (batch, na)

            # ISR |eta|: ISR can be more forward
            isr_pz = isr_4vec[..., 3]
            isr_eta = torch.asinh(isr_pz / isr_pt)
            isr_abs_eta = torch.abs(isr_eta)

            # Min deltaR between ISR candidate and all other jets
            def jet_eta_phi(p):
                px, py, pz = p[..., 1], p[..., 2], p[..., 3]
                pt = torch.sqrt(px**2 + py**2).clamp(min=1e-8)
                return torch.asinh(pz / pt), torch.atan2(py, px)

            all_eta, all_phi = jet_eta_phi(four_momenta)  # (batch, num_jets)
            isr_eta_r = isr_eta.unsqueeze(-1)  # (batch, na, 1)
            isr_phi_r = torch.atan2(isr_py, isr_px).unsqueeze(-1)
            all_eta_r = all_eta.unsqueeze(1)  # (batch, 1, num_jets)
            all_phi_r = all_phi.unsqueeze(1)

            d_eta = isr_eta_r - all_eta_r
            d_phi = isr_phi_r - all_phi_r
            d_phi = d_phi - 2 * torch.pi * torch.round(d_phi / (2 * torch.pi))
            dr_all = torch.sqrt(d_eta**2 + d_phi**2 + 1e-8)  # (batch, na, num_jets)
            # Set self-distance to large value before taking min
            isr_mask = torch.zeros(batch_size, na, self.num_jets, device=four_momenta.device)
            isr_positions = self.isr_indices.unsqueeze(0).expand(batch_size, -1)
            isr_mask.scatter_(2, isr_positions.unsqueeze(-1), 1.0)
            dr_all = dr_all + isr_mask * 100.0
            isr_min_dr = dr_all.min(dim=-1).values  # (batch, na)

            # Mass pull: how much adding ISR to each group worsens mass balance
            g1_plus_isr = g1_4vec + isr_4vec
            g2_plus_isr = g2_4vec + isr_4vec
            m1_with_isr = inv_mass(g1_plus_isr)
            m2_with_isr = inv_mass(g2_plus_isr)
            mass_pull = torch.min(
                torch.abs(m1_with_isr - m2) / mass_sum.clamp(min=1e-8),
                torch.abs(m1 - m2_with_isr) / mass_sum.clamp(min=1e-8),
            )

            group_features.extend([isr_pt_frac, isr_abs_eta, isr_min_dr, mass_pull])

        physics = torch.stack(group_features, dim=-1)
        return physics

    def score_assignments(
        self, jet_embeddings: torch.Tensor, four_momenta: torch.Tensor
    ) -> torch.Tensor:
        """Score all assignments given jet embeddings and raw four-vectors.

        Args:
            jet_embeddings: (batch, num_jets, d_model)
            four_momenta: (batch, num_jets, 4) raw (E, px, py, pz)

        Returns:
            (batch, num_assignments) logits
        """
        batch_size = jet_embeddings.shape[0]
        na = self.num_assignments

        # Expand jet embeddings for gathering: (batch, num_assignments, num_jets, d_model)
        je_expanded = jet_embeddings.unsqueeze(1).expand(-1, na, -1, -1)

        # Gather group1 embeddings and sum-pool
        g1_idx = self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.d_model
        )
        g1_pooled = torch.gather(je_expanded, 2, g1_idx).sum(dim=2)

        # Gather group2 embeddings and sum-pool
        g2_idx = self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.d_model
        )
        g2_pooled = torch.gather(je_expanded, 2, g2_idx).sum(dim=2)

        # Provably symmetric features: sum and product are invariant to swap
        sym_sum = g1_pooled + g2_pooled      # (batch, na, d_model)
        sym_prod = g1_pooled * g2_pooled     # (batch, na, d_model)

        # Physics features computed directly from four-vectors
        physics = self._compute_physics_features(four_momenta)

        if self.has_isr:
            # Gather ISR embeddings
            isr_idx = self.isr_indices.unsqueeze(0).unsqueeze(-1).expand(
                batch_size, -1, self.d_model
            )
            isr_embed = torch.gather(
                je_expanded, 2, isr_idx.unsqueeze(2)
            ).squeeze(2)  # (batch, na, d_model)

            combined = torch.cat([isr_embed, sym_sum, sym_prod, physics], dim=-1)
        else:
            combined = torch.cat([sym_sum, sym_prod, physics], dim=-1)

        scores = self.score_mlp(combined).squeeze(-1)
        return scores

    def predict_mass(self, jet_embeddings: torch.Tensor) -> torch.Tensor:
        """Adversarial mass prediction from pooled jet embeddings."""
        pooled = jet_embeddings.mean(dim=1)
        reversed_pooled = self.gradient_reversal(pooled)
        return self.mass_adversary(reversed_pooled)

    def forward(self, four_momenta: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            four_momenta: (batch, num_jets, 4)

        Returns:
            dict with logits (batch, num_assignments) and mass_pred (batch, 1)
        """
        jet_embeddings = self.encode_jets(four_momenta)
        logits = self.score_assignments(jet_embeddings, four_momenta)
        mass_pred = self.predict_mass(jet_embeddings)
        return {"logits": logits, "mass_pred": mass_pred}
