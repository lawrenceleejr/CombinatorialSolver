"""
Export trained models to ONNX format.

Two ONNX files are produced:

  <output_dir>/ml_model.onnx
      The trained transformer model.  Input: normalised four-momenta.

  <output_dir>/classical_mass_asymmetry.onnx
      A purely classical solver that selects the jet assignment minimising
      the relative mass asymmetry |m1-m2|/(m1+m2) between the two
      reconstructed candidates.  Input: *un-normalised* physical four-momenta
      (same units as the HDF5 data files) so that meaningful invariant masses
      can be computed.

Both models share the same output interface:

  Input  : ``four_momenta``  shape (batch, num_jets, 4)  float32
  Output : ``logits``        shape (batch, num_assignments) float32

``logits.argmax(axis=-1)`` gives the predicted assignment index for each
event, directly comparable between the two models.
"""

import argparse
from pathlib import Path

import torch

from .model import JetAssignmentTransformer, MassAsymmetryClassicalSolver
from .utils import get_config, get_device


class _LogitsOnly(torch.nn.Module):
    """Thin wrapper that extracts the ``logits`` tensor from a model's output dict."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, four_momenta: torch.Tensor) -> torch.Tensor:
        return self.inner(four_momenta)["logits"]


def export_ml_model(
    checkpoint_path: str,
    output_path: str,
    config_path: str | None = None,
) -> None:
    """Export the trained transformer model to ONNX.

    Args:
        checkpoint_path: Path to a ``best_model.pt`` checkpoint.
        output_path: Destination ``.onnx`` file.
        config_path: Optional config override.
    """
    device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", get_config(config_path))
    mc = config["model"]
    dc = config["data"]
    num_jets = dc["num_jets"]

    model = JetAssignmentTransformer(
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=mc.get("dropout", 0.1),
        num_jets=num_jets,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wrapped = _LogitsOnly(model).to(device)
    wrapped.eval()

    dummy = torch.zeros(1, num_jets, 4, device=device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        output_path,
        input_names=["four_momenta"],
        output_names=["logits"],
        dynamic_axes={"four_momenta": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ML model exported → {output_path}")


def export_classical_solver(
    output_path: str,
    num_jets: int = 7,
) -> None:
    """Export the classical mass-asymmetry solver to ONNX.

    The classical solver does not require a checkpoint — it is a purely
    deterministic computation over the jet four-momenta.

    Args:
        output_path: Destination ``.onnx`` file.
        num_jets: Number of jets per event (6 or 7).
    """
    solver = MassAsymmetryClassicalSolver(num_jets=num_jets)
    solver.eval()

    wrapped = _LogitsOnly(solver)
    wrapped.eval()

    dummy = torch.zeros(1, num_jets, 4)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        output_path,
        input_names=["four_momenta"],
        output_names=["logits"],
        dynamic_axes={"four_momenta": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Classical solver exported → {output_path}")


def export_all(
    checkpoint_path: str,
    output_dir: str = "onnx_models",
    config_path: str | None = None,
) -> None:
    """Export both the ML model and the classical solver to ONNX.

    Args:
        checkpoint_path: Path to ``best_model.pt``.
        output_dir: Directory in which both ONNX files are written.
        config_path: Optional config override for the ML model.
    """
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", get_config(config_path))
    num_jets = config["data"]["num_jets"]

    export_ml_model(
        checkpoint_path=checkpoint_path,
        output_path=str(Path(output_dir) / "ml_model.onnx"),
        config_path=config_path,
    )
    export_classical_solver(
        output_path=str(Path(output_dir) / "classical_mass_asymmetry.onnx"),
        num_jets=num_jets,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export jet assignment models to ONNX"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (best_model.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_models",
        help="Directory for output ONNX files (default: onnx_models/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config YAML override",
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Export only the classical solver (no checkpoint needed with --num-jets)",
    )
    parser.add_argument(
        "--num-jets",
        type=int,
        default=7,
        help="Number of jets per event for classical-only export (default: 7)",
    )
    args = parser.parse_args()

    if args.classical_only:
        export_classical_solver(
            output_path=str(Path(args.output_dir) / "classical_mass_asymmetry.onnx"),
            num_jets=args.num_jets,
        )
    else:
        export_all(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            config_path=args.config,
        )
