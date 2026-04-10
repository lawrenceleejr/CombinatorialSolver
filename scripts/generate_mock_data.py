"""
Generate synthetic HDF5 data matching the MadGraphMLProducer format.

Creates fake pair-produced resonance events where each parent decays
to 3 jets, plus one ISR jet. Useful for testing the full pipeline
without real Monte Carlo data.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def generate_jet(pt_mean: float, eta_max: float = 2.5) -> np.ndarray:
    """Generate a single jet's (pt, eta, phi, mass)."""
    pt = np.random.exponential(pt_mean)
    pt = max(pt, 25.0)  # minimum pT cut
    eta = np.random.uniform(-eta_max, eta_max)
    phi = np.random.uniform(-np.pi, np.pi)
    mass = np.random.exponential(10.0)  # light jet mass ~ few GeV
    return np.array([pt, eta, phi, mass])


def generate_event(parent_mass: float, parent_pdg: int = 1000021) -> dict:
    """Generate one event with two parent particles decaying to 3 jets each + ISR.

    The parent is generated roughly at rest in the transverse plane (back-to-back)
    with the given invariant mass. Each parent decays isotropically to 3 jets in its
    rest frame, then boosted.

    Args:
        parent_mass: Invariant mass of each parent particle in GeV.
        parent_pdg: PDG ID to assign to parent particles.

    Returns:
        dict with jet_features (20, 7), event_features (7), jet_mask (20)
    """
    # Generate two parent particles (roughly back-to-back in transverse plane)
    parent_pt = np.random.exponential(parent_mass * 0.3)
    parent_phi1 = np.random.uniform(-np.pi, np.pi)
    parent_phi2 = parent_phi1 + np.pi  # back-to-back
    parent_eta1 = np.random.normal(0, 1.5)
    parent_eta2 = np.random.normal(0, 1.5)

    jets = []

    # For each parent, generate 3 decay jets
    for p_idx, (p_phi, p_eta) in enumerate(
        [(parent_phi1, parent_eta1), (parent_phi2, parent_eta2)]
    ):
        pid = parent_pdg + p_idx  # differentiate the two parents
        # Jet pT fractions sum to ~1 (energy sharing)
        fracs = np.random.dirichlet([2, 2, 2])
        jet_pt_total = np.sqrt(parent_pt**2 + parent_mass**2) * 0.8

        for frac in fracs:
            pt = max(jet_pt_total * frac + np.random.normal(0, 10), 25.0)
            eta = p_eta + np.random.normal(0, 0.5)
            phi = p_phi + np.random.normal(0, 0.4)
            # Wrap phi
            phi = ((phi + np.pi) % (2 * np.pi)) - np.pi
            mass = np.random.exponential(5.0)

            jets.append({
                "kinematics": np.array([pt, eta, phi, mass]),
                "parent_pdg": pid,
                "is_signal": 1,
            })

    # ISR jet (typically softer, random direction)
    isr_pt = np.random.exponential(40.0)
    isr_pt = max(isr_pt, 25.0)
    isr_eta = np.random.uniform(-2.5, 2.5)
    isr_phi = np.random.uniform(-np.pi, np.pi)
    isr_mass = np.random.exponential(3.0)
    jets.append({
        "kinematics": np.array([isr_pt, isr_eta, isr_phi, isr_mass]),
        "parent_pdg": 0,
        "is_signal": 0,
    })

    # Sort by pT (descending)
    jets.sort(key=lambda j: j["kinematics"][0], reverse=True)

    # Build jet_features array (20, 7)
    jet_features = np.zeros((20, 7), dtype=np.float32)
    jet_mask = np.zeros(20, dtype=np.float32)

    for i, jet in enumerate(jets[:20]):
        jet_features[i, :4] = jet["kinematics"]
        jet_features[i, 4] = np.random.randint(5, 50)  # n_constituents
        jet_features[i, 5] = jet["parent_pdg"]
        jet_features[i, 6] = jet["is_signal"]
        jet_mask[i] = 1.0

    # Event features
    all_pt = [j["kinematics"][0] for j in jets]
    ht = sum(all_pt)
    met_pt = np.random.exponential(30.0)
    met_phi = np.random.uniform(-np.pi, np.pi)
    met_x = met_pt * np.cos(met_phi)
    met_y = met_pt * np.sin(met_phi)
    n_signal = sum(1 for j in jets if j["is_signal"] == 1)

    event_features = np.array(
        [len(jets), met_x, met_y, met_pt, ht, n_signal, 1.0],
        dtype=np.float32,
    )

    return {
        "jet_features": jet_features,
        "jet_mask": jet_mask,
        "event_features": event_features,
    }


def generate_dataset(
    output_path: str,
    n_events: int = 10000,
    parent_masses: list[float] | None = None,
):
    """Generate a full HDF5 dataset.

    Args:
        output_path: Path for the output HDF5 file.
        n_events: Number of events to generate.
        parent_masses: List of parent masses to sample from. If None,
            uses [300, 500, 700, 1000, 1500] GeV.
    """
    if parent_masses is None:
        parent_masses = [300.0, 500.0, 700.0, 1000.0, 1500.0]

    jet_features_all = np.zeros((n_events, 20, 7), dtype=np.float32)
    jet_mask_all = np.zeros((n_events, 20), dtype=np.float32)
    event_features_all = np.zeros((n_events, 7), dtype=np.float32)
    # Also create dummy particle features/mask for format compatibility
    particle_features_all = np.zeros((n_events, 20, 100, 5), dtype=np.float32)
    particle_mask_all = np.zeros((n_events, 20, 100), dtype=np.float32)

    for i in range(n_events):
        mass = np.random.choice(parent_masses)
        event = generate_event(parent_mass=mass)
        jet_features_all[i] = event["jet_features"]
        jet_mask_all[i] = event["jet_mask"]
        event_features_all[i] = event["event_features"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("jet_features", data=jet_features_all)
        f.create_dataset("jet_mask", data=jet_mask_all)
        f.create_dataset("event_features", data=event_features_all)
        f.create_dataset("particle_features", data=particle_features_all)
        f.create_dataset("particle_mask", data=particle_mask_all)

    print(f"Generated {n_events} events -> {output_path}")
    print(f"  Parent masses sampled from: {parent_masses} GeV")
    print(f"  File size: {Path(output_path).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock HDF5 data")
    parser.add_argument("--output", type=str, default="data/mock_data.h5", help="Output path")
    parser.add_argument("--n-events", type=int, default=10000, help="Number of events")
    parser.add_argument(
        "--masses", type=float, nargs="+", default=None,
        help="Parent masses in GeV (default: 300 500 700 1000 1500)",
    )
    args = parser.parse_args()
    generate_dataset(args.output, args.n_events, args.masses)
