"""
Utility functions: device selection, config loading, logging.
"""

from pathlib import Path

import torch
import yaml


def get_device() -> torch.device:
    """Select the best available device: MPS -> CUDA -> CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def default_config() -> dict:
    """Return default configuration."""
    return {
        "model": {
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 256,
            "dropout": 0.1,
        },
        "training": {
            "batch_size": 512,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "num_epochs": 100,
            "warmup_epochs": 5,
            "lambda_adv": 1.0,
            "lambda_adv_rampup": 10,
            "lambda_distill": 2.0,
            "lambda_distill_epochs": 20,
            "distill_temperature": 4.0,
            "lambda_entropy_asym": 0.0,
            "lambda_entropy_mass": 0.0,
        },
        "data": {
            "normalize_by_ht": True,
            "num_jets": 7,
        },
    }


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override into base config."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_config(config_path: str | None = None) -> dict:
    """Load config from file (if provided) merged with defaults."""
    config = default_config()
    if config_path is not None:
        file_config = load_config(config_path)
        config = merge_configs(config, file_config)
    return config


def load_compatible_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    """Load a checkpoint, tolerating ONLY a missing/extra ``score_head``.

    Checkpoints written before the event-level score head was added lack its
    weights; loading them with ``strict=True`` would fail even though every
    other parameter matches.  This loads with ``strict=False`` but re-raises if
    anything *other* than ``score_head.*`` is missing or unexpected, so real
    architecture mismatches still fail loudly.  When the score head is left at
    its random initialisation a warning is printed — the score output is
    meaningless for such checkpoints and must not be cut on.
    """
    result = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in result.missing_keys if not k.startswith("score_head.")]
    bad_unexpected = [k for k in result.unexpected_keys if not k.startswith("score_head.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch beyond the score head: "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )
    if any(k.startswith("score_head.") for k in result.missing_keys):
        print(
            "  Note: checkpoint has no score_head weights (pre-score-head "
            "checkpoint). The NN signal score is untrained/meaningless for "
            "this model — do not cut on it."
        )


def compute_invariant_mass(four_momenta: torch.Tensor) -> torch.Tensor:
    """Compute invariant mass from summed four-momenta.

    Args:
        four_momenta: (..., 4) tensor with (E, px, py, pz)

    Returns:
        (...,) tensor of invariant masses
    """
    e = four_momenta[..., 0]
    px = four_momenta[..., 1]
    py = four_momenta[..., 2]
    pz = four_momenta[..., 3]
    m2 = e**2 - px**2 - py**2 - pz**2
    return torch.sqrt(torch.clamp(m2, min=0.0))
