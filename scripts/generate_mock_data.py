"""
Generate synthetic HDF5 data matching the MadGraphMLProducer format.

Creates fake pair-produced resonance events where each parent decays
to 3 jets. Matches the real data layout including TARGETS and INPUTS groups.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def generate_event(parent_mass: float, include_isr: bool = False) -> dict:
    """Generate one event with two parent particles decaying to 3 jets each.

    Jets are stored in truth-group order (g1 first, g2 second, optional ISR last).
    The TARGETS encode this original ordering.

    Args:
        parent_mass: Invariant mass of each parent particle in GeV.
        include_isr: If True, add a 7th ISR jet.
    """
    jets = []

    # Generate two parent particles (roughly back-to-back in transverse plane)
    parent_pt = np.random.exponential(parent_mass * 0.3)
    parent_phi1 = np.random.uniform(-np.pi, np.pi)
    parent_phi2 = parent_phi1 + np.pi
    parent_eta1 = np.random.normal(0, 1.5)
    parent_eta2 = np.random.normal(0, 1.5)

    for p_phi, p_eta in [(parent_phi1, parent_eta1), (parent_phi2, parent_eta2)]:
        fracs = np.random.dirichlet([2, 2, 2])
        jet_pt_total = np.sqrt(parent_pt**2 + parent_mass**2) * 0.8

        for frac in fracs:
            pt = max(jet_pt_total * frac + np.random.normal(0, 10), 25.0)
            eta = p_eta + np.random.normal(0, 0.5)
            phi = p_phi + np.random.normal(0, 0.4)
            phi = ((phi + np.pi) % (2 * np.pi)) - np.pi
            mass = np.random.exponential(0.005)  # near-massless partons
            jets.append({"pt": pt, "eta": eta, "phi": phi, "mass": mass})

    if include_isr:
        isr_pt = max(np.random.exponential(40.0), 25.0)
        jets.append({
            "pt": isr_pt,
            "eta": np.random.uniform(-2.5, 2.5),
            "phi": np.random.uniform(-np.pi, np.pi),
            "mass": np.random.exponential(0.003),
        })

    n_jets = len(jets)
    max_jets = 20

    # Build arrays in truth order (g1=[0,1,2], g2=[3,4,5], ISR=6 if present)
    pt = np.zeros(max_jets, dtype=np.float32)
    eta = np.zeros(max_jets, dtype=np.float32)
    phi = np.zeros(max_jets, dtype=np.float32)
    mass = np.zeros(max_jets, dtype=np.float32)
    mask = np.zeros(max_jets, dtype=bool)

    for i, j in enumerate(jets):
        pt[i] = j["pt"]
        eta[i] = j["eta"]
        phi[i] = j["phi"]
        mass[i] = j["mass"]
        mask[i] = True

    # Compute energy
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

    ht = pt[mask].sum()

    # jet_features: [pt, eta, phi, mass, parent_idx, is_signal]
    jet_features = np.zeros((max_jets, 6), dtype=np.float32)
    for i in range(n_jets):
        jet_features[i, 0] = pt[i]
        jet_features[i, 1] = eta[i]
        jet_features[i, 2] = phi[i]
        jet_features[i, 3] = mass[i]
        if i < 3:
            jet_features[i, 4] = 1.0  # parent_idx for g1
        elif i < 6:
            jet_features[i, 4] = 2.0  # parent_idx for g2
        else:
            jet_features[i, 4] = 0.0  # ISR
        jet_features[i, 5] = 0.0  # is_signal (matches real data convention)

    event_features = np.array(
        [n_jets, 0.0, 0.0, 0.0, ht, 6, 1.0], dtype=np.float32
    )

    return {
        "jet_features": jet_features,
        "jet_mask": mask,
        "event_features": event_features,
        "pt": pt, "eta": eta, "phi": phi, "mass": mass,
        "energy": energy, "mask": mask,
    }


def generate_dataset(
    output_path: str,
    n_events: int = 10000,
    parent_masses: list[float] | None = None,
    include_isr: bool = False,
):
    """Generate a full HDF5 dataset matching the real data layout."""
    if parent_masses is None:
        parent_masses = [300.0, 500.0, 700.0, 1000.0, 1500.0]

    max_jets = 20
    n_jets_per_event = 7 if include_isr else 6

    # Preallocate
    jet_features_all = np.zeros((n_events, max_jets, 6), dtype=np.float32)
    jet_mask_all = np.zeros((n_events, max_jets), dtype=bool)
    event_features_all = np.zeros((n_events, 7), dtype=np.float32)
    particle_features_all = np.zeros((n_events, max_jets, 100, 5), dtype=np.float32)
    particle_mask_all = np.zeros((n_events, max_jets, 100), dtype=bool)

    # INPUTS/Source arrays
    src_pt = np.zeros((n_events, max_jets), dtype=np.float32)
    src_eta = np.zeros((n_events, max_jets), dtype=np.float32)
    src_phi = np.zeros((n_events, max_jets), dtype=np.float32)
    src_mass = np.zeros((n_events, max_jets), dtype=np.float32)
    src_btag = np.zeros((n_events, max_jets), dtype=np.float32)
    src_mask = np.zeros((n_events, max_jets), dtype=bool)

    # source group arrays (E, eta, phi, pt)
    source_e = np.zeros((n_events, max_jets), dtype=np.float32)
    source_eta = np.zeros((n_events, max_jets), dtype=np.float32)
    source_phi = np.zeros((n_events, max_jets), dtype=np.float32)
    source_pt = np.zeros((n_events, max_jets), dtype=np.float32)
    source_mask = np.zeros((n_events, max_jets), dtype=bool)

    # TARGETS
    g1_j1 = np.zeros(n_events, dtype=np.int32)
    g1_j2 = np.ones(n_events, dtype=np.int32)
    g1_j3 = np.full(n_events, 2, dtype=np.int32)
    g2_j1 = np.full(n_events, 3, dtype=np.int32)
    g2_j2 = np.full(n_events, 4, dtype=np.int32)
    g2_j3 = np.full(n_events, 5, dtype=np.int32)

    for i in range(n_events):
        mass = np.random.choice(parent_masses)
        event = generate_event(parent_mass=mass, include_isr=include_isr)

        jet_features_all[i] = event["jet_features"]
        jet_mask_all[i] = event["jet_mask"]
        event_features_all[i] = event["event_features"]

        src_pt[i] = event["pt"]
        src_eta[i] = event["eta"]
        src_phi[i] = event["phi"]
        src_mass[i] = event["mass"]
        src_mask[i] = event["mask"]

        source_e[i] = event["energy"]
        source_eta[i] = event["eta"]
        source_phi[i] = event["phi"]
        source_pt[i] = event["pt"]
        source_mask[i] = event["mask"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Standard datasets
        f.create_dataset("jet_features", data=jet_features_all)
        f.create_dataset("jet_mask", data=jet_mask_all)
        f.create_dataset("event_features", data=event_features_all)
        f.create_dataset("particle_features", data=particle_features_all)
        f.create_dataset("particle_mask", data=particle_mask_all)

        # INPUTS/Source group
        inputs_src = f.create_group("INPUTS/Source")
        inputs_src.create_dataset("pt", data=src_pt)
        inputs_src.create_dataset("eta", data=src_eta)
        inputs_src.create_dataset("phi", data=src_phi)
        inputs_src.create_dataset("mass", data=src_mass)
        inputs_src.create_dataset("btag", data=src_btag)
        inputs_src.create_dataset("MASK", data=src_mask)

        # source group
        source_grp = f.create_group("source")
        source_grp.create_dataset("e", data=source_e)
        source_grp.create_dataset("eta", data=source_eta)
        source_grp.create_dataset("phi", data=source_phi)
        source_grp.create_dataset("pt", data=source_pt)
        source_grp.create_dataset("mask", data=source_mask)

        # TARGETS group
        g1 = f.create_group("TARGETS/g1")
        g1.create_dataset("j1", data=g1_j1)
        g1.create_dataset("j2", data=g1_j2)
        g1.create_dataset("j3", data=g1_j3)
        g2 = f.create_group("TARGETS/g2")
        g2.create_dataset("j1", data=g2_j1)
        g2.create_dataset("j2", data=g2_j2)
        g2.create_dataset("j3", data=g2_j3)

        # EventVars
        ev = f.create_group("EventVars")
        ev.create_dataset("normweight", data=np.ones(n_events, dtype=np.float32))

    print(f"Generated {n_events} events -> {output_path}")
    print(f"  Parent masses sampled from: {parent_masses} GeV")
    print(f"  Jets per event: {n_jets_per_event} ({'with' if include_isr else 'without'} ISR)")
    print(f"  File size: {Path(output_path).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock HDF5 data")
    parser.add_argument("--output", type=str, default="data/mock_data.h5")
    parser.add_argument("--n-events", type=int, default=10000)
    parser.add_argument("--masses", type=float, nargs="+", default=None)
    parser.add_argument("--include-isr", action="store_true", help="Add ISR jet (7 jets)")
    args = parser.parse_args()
    generate_dataset(args.output, args.n_events, args.masses, args.include_isr)
