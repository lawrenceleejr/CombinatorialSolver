"""
Transformer-based jet assignment model.

Architecture:
  1. Linear projection of (E, px, py, pz) to d_model
  2. Transformer encoder with self-attention over 7 jet tokens
  3. Assignment scoring: enumerate all 70 possible assignments,
     aggregate jet embeddings per group, score with MLP
  4. Adversarial mass decorrelation head (gradient reversal)
"""

import torch
import torch.nn as nn

from .combinatorics import build_assignment_tensors


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for adversarial training.

    Forward pass: identity. Backward pass: negate and scale gradients.
    """

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
        num_jets: Number of input jets (default 7).
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
        self.register_buffer("isr_indices", at["isr_indices"])      # (70,)
        self.register_buffer("group1_indices", at["group1_indices"])  # (70, 3)
        self.register_buffer("group2_indices", at["group2_indices"])  # (70, 3)
        self.num_assignments = len(at["isr_indices"])

        # Assignment scoring MLP
        # Input: isr_embed (d_model) + symmetric group embed (2 * d_model)
        self.score_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
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
            four_momenta: (batch, num_jets, 4) — E, px, py, pz

        Returns:
            (batch, num_jets, d_model) jet embeddings
        """
        batch_size = four_momenta.shape[0]

        # Project input
        x = self.input_proj(four_momenta)  # (batch, 7, d_model)

        # Add positional encoding
        positions = torch.arange(self.num_jets, device=four_momenta.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)  # broadcast over batch

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, 7, d_model)
        return x

    def score_assignments(self, jet_embeddings: torch.Tensor) -> torch.Tensor:
        """Score all 70 assignments given jet embeddings.

        Args:
            jet_embeddings: (batch, num_jets, d_model)

        Returns:
            (batch, 70) logits over assignments
        """
        batch_size = jet_embeddings.shape[0]
        device = jet_embeddings.device

        # Gather ISR embeddings: (batch, 70, d_model)
        isr_idx = self.isr_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, self.d_model
        )  # (batch, 70, d_model)
        isr_embed = torch.gather(
            jet_embeddings.unsqueeze(1).expand(-1, self.num_assignments, -1, -1),
            2,
            isr_idx.unsqueeze(2),
        ).squeeze(2)  # (batch, 70, d_model)

        # Gather group1 embeddings and sum-pool: (batch, 70, d_model)
        g1_idx = self.group1_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.d_model
        )  # (batch, 70, 3, d_model)
        g1_embed = torch.gather(
            jet_embeddings.unsqueeze(1).expand(-1, self.num_assignments, -1, -1),
            2,
            g1_idx,
        )  # (batch, 70, 3, d_model)
        g1_pooled = g1_embed.sum(dim=2)  # (batch, 70, d_model)

        # Gather group2 embeddings and sum-pool: (batch, 70, d_model)
        g2_idx = self.group2_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, self.d_model
        )
        g2_embed = torch.gather(
            jet_embeddings.unsqueeze(1).expand(-1, self.num_assignments, -1, -1),
            2,
            g2_idx,
        )
        g2_pooled = g2_embed.sum(dim=2)  # (batch, 70, d_model)

        # Symmetric aggregation: sort by L2 norm so score(g1,g2) == score(g2,g1)
        g1_norm = g1_pooled.norm(dim=-1, keepdim=True)  # (batch, 70, 1)
        g2_norm = g2_pooled.norm(dim=-1, keepdim=True)

        # Where g1_norm >= g2_norm, keep order; otherwise swap
        swap = (g1_norm < g2_norm).expand_as(g1_pooled)
        first = torch.where(swap, g2_pooled, g1_pooled)
        second = torch.where(swap, g1_pooled, g2_pooled)

        # Concatenate: [isr_embed, first_group, second_group]
        combined = torch.cat([isr_embed, first, second], dim=-1)  # (batch, 70, 3*d_model)

        # Score each assignment
        scores = self.score_mlp(combined).squeeze(-1)  # (batch, 70)
        return scores

    def predict_mass(self, jet_embeddings: torch.Tensor) -> torch.Tensor:
        """Adversarial mass prediction from pooled jet embeddings.

        Args:
            jet_embeddings: (batch, num_jets, d_model)

        Returns:
            (batch, 1) predicted parent mass
        """
        # Mean-pool over jets
        pooled = jet_embeddings.mean(dim=1)  # (batch, d_model)

        # Gradient reversal + mass prediction
        reversed_pooled = self.gradient_reversal(pooled)
        mass_pred = self.mass_adversary(reversed_pooled)  # (batch, 1)
        return mass_pred

    def forward(
        self, four_momenta: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            four_momenta: (batch, num_jets, 4) — E, px, py, pz

        Returns:
            dict with:
                - logits: (batch, 70) assignment logits
                - mass_pred: (batch, 1) adversarial mass prediction
        """
        jet_embeddings = self.encode_jets(four_momenta)
        logits = self.score_assignments(jet_embeddings)
        mass_pred = self.predict_mass(jet_embeddings)

        return {"logits": logits, "mass_pred": mass_pred}
