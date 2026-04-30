"""
Generate synthetic HDF5 data matching the MadGraphMLProducer format.

Creates fake pair-produced resonance events where each parent decays
to 3 jets. Two topologies are supported:
  - Direct 3-body: P → j1 + j2 + j3 (flat Dirichlet pT split)
  - Cascade 2+2:   P → j1 + R → j1 + (j2 + j3) with an on-shell
    intermediate resonance R of mass m_R < m_P.  The kinematics are
    generated using proper relativistic two-body decays (isotropic in
    each rest frame) followed by Lorentz boosts to the lab frame.
    Cascade events produce large within-triplet pT hierarchies when
    m_R ≪ m_P and a sharp Dalitz edge at m(j2,j3)=m_R that is absent
    in both direct decays and QCD.

Matches the real data layout including TARGETS and INPUTS groups.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Relativistic kinematics helpers
# ---------------------------------------------------------------------------

def _two_body_decay_rest_frame(
    m_parent: float, m_child1: float, m_child2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic two-body decay in the parent rest frame.

    Args:
        m_parent: Parent invariant mass (GeV).
        m_child1: Child-1 invariant mass (GeV); use 0.0 for massless.
        m_child2: Child-2 invariant mass (GeV); use 0.0 for massless.

    Returns:
        (p4_child1, p4_child2): each a (4,) array (E, px, py, pz) in the
        parent rest frame.  Momentum is conserved: p1 + p2 = 0.
    """
    lam = (m_parent**2 - (m_child1 + m_child2)**2) * (m_parent**2 - (m_child1 - m_child2)**2)
    p_mag = np.sqrt(max(lam, 0.0)) / (2.0 * m_parent)
    E1 = np.sqrt(p_mag**2 + m_child1**2)
    E2 = m_parent - E1  # energy conservation

    # Uniform isotropic direction
    cos_theta = np.random.uniform(-1.0, 1.0)
    sin_theta = np.sqrt(max(1.0 - cos_theta**2, 0.0))
    phi = np.random.uniform(0.0, 2.0 * np.pi)

    px = p_mag * sin_theta * np.cos(phi)
    py = p_mag * sin_theta * np.sin(phi)
    pz = p_mag * cos_theta

    return (
        np.array([E1,  px,  py,  pz], dtype=np.float64),
        np.array([E2, -px, -py, -pz], dtype=np.float64),
    )


def _boost_to_lab(p4_rest: np.ndarray, parent_p4_lab: np.ndarray) -> np.ndarray:
    """Boost a 4-vector from the parent rest frame to the lab frame.

    Applies the standard active Lorentz boost defined by the parent's
    lab-frame 4-momentum.  Works for any parent velocity including β → 0.

    Args:
        p4_rest: (4,) or (N, 4) 4-vector(s) in the parent rest frame.
        parent_p4_lab: (4,) parent 4-momentum (E, px, py, pz) in lab frame.

    Returns:
        (4,) or (N, 4) boosted 4-vector(s) in the lab frame.
    """
    scalar = p4_rest.ndim == 1
    p = np.atleast_2d(p4_rest).astype(np.float64)   # (N, 4)

    E_par = parent_p4_lab[0]
    p_par = parent_p4_lab[1:]                        # (3,)
    beta_sq = np.dot(p_par, p_par) / (E_par**2)

    if beta_sq < 1e-14:
        return p4_rest.copy()                        # essentially at rest already

    beta_mag = np.sqrt(beta_sq)
    beta_hat = p_par / (E_par * beta_mag)            # unit vector along boost
    gamma = E_par / np.sqrt(E_par**2 - np.dot(p_par, p_par))

    E_in = p[:, 0]                                   # (N,)
    p_in = p[:, 1:]                                  # (N, 3)

    p_par_proj = p_in @ beta_hat                     # (N,) – along boost direction
    p_perp = p_in - np.outer(p_par_proj, beta_hat)  # (N, 3) – perpendicular

    E_out = gamma * (E_in + beta_mag * p_par_proj)
    p_par_out = gamma * (p_par_proj + beta_mag * E_in)
    p_out = np.outer(p_par_out, beta_hat) + p_perp  # (N, 3)

    result = np.column_stack([E_out, p_out])         # (N, 4)
    return result[0] if scalar else result


def _build_event_arrays(jets: list[dict], include_isr: bool) -> dict:
    """Build the numpy arrays for one event from a list of jet dicts.

    Shared by both generate_event and generate_event_cascade.

    Args:
        jets: List of dicts with keys pt, eta, phi, mass.
              g1 jets must be the first 3 entries, g2 the next 3, optional ISR last.
        include_isr: Whether an ISR jet is present (used only for logging).

    Returns:
        Dict of arrays in the format expected by generate_dataset.
    """
    n_jets = len(jets)
    max_jets = 20

    pt = np.zeros(max_jets, dtype=np.float32)
    eta = np.zeros(max_jets, dtype=np.float32)
    phi = np.zeros(max_jets, dtype=np.float32)
    mass_arr = np.zeros(max_jets, dtype=np.float32)
    mask = np.zeros(max_jets, dtype=bool)

    for i, j in enumerate(jets):
        pt[i] = j["pt"]
        eta[i] = j["eta"]
        phi[i] = j["phi"]
        mass_arr[i] = j["mass"]
        mask[i] = True

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass_arr**2)
    ht = pt[mask].sum()

    jet_features = np.zeros((max_jets, 6), dtype=np.float32)
    for i in range(n_jets):
        jet_features[i, 0] = pt[i]
        jet_features[i, 1] = eta[i]
        jet_features[i, 2] = phi[i]
        jet_features[i, 3] = mass_arr[i]
        if i < 3:
            jet_features[i, 4] = 1.0
        elif i < 6:
            jet_features[i, 4] = 2.0
        else:
            jet_features[i, 4] = 0.0  # ISR
        jet_features[i, 5] = 0.0

    event_features = np.array([n_jets, 0.0, 0.0, 0.0, ht, 6, 1.0], dtype=np.float32)

    return {
        "jet_features": jet_features,
        "jet_mask": mask,
        "event_features": event_features,
        "pt": pt, "eta": eta, "phi": phi, "mass": mass_arr,
        "energy": energy, "mask": mask,
    }


def generate_event(parent_mass: float, include_isr: bool = False) -> dict:
    """Generate one direct 3-body event: P → j1 + j2 + j3 for each parent.

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

    return _build_event_arrays(jets, include_isr)


def generate_event_cascade(
    parent_mass: float,
    include_isr: bool = False,
    resonance_mass_frac: float | None = None,
) -> dict:
    """Generate one cascade event: P → j1 + R → j1 + (j2 + j3) for each parent.

    Uses proper relativistic two-body decay kinematics:
      1. P → j1 + R  (isotropic in P's rest frame)
      2. R → j2 + j3 (isotropic in R's rest frame)
    followed by Lorentz boosts to the lab frame.

    Key kinematic features of the cascade topology:
      - m(j2, j3) = m_R  (sharp sub-resonance peak; absent in direct/QCD)
      - Large within-triplet pT hierarchy when m_R ≪ m_P: j1 recoils hard
        against the boosted resonance, which then decays to two softer jets.
      - Small D₂ (2-prong structure from the j2+j3 resonance pair).
      - All three sorted Dalitz ratios are needed to unambiguously identify
        which jet pair is the resonance.

    Args:
        parent_mass: Invariant mass of each parent P in GeV.
        include_isr: If True, add a 7th ISR jet.
        resonance_mass_frac: m_R / m_P ratio.  Sampled uniformly from
            [0.15, 0.85] when None, ensuring visible mass splittings.
    """
    if resonance_mass_frac is None:
        resonance_mass_frac = np.random.uniform(0.15, 0.85)
    m_R = parent_mass * resonance_mass_frac

    jets = []

    # Generate two parent particles (roughly back-to-back in transverse plane)
    parent_pt = np.random.exponential(parent_mass * 0.3)
    parent_phi1 = np.random.uniform(-np.pi, np.pi)
    parent_phi2 = parent_phi1 + np.pi
    parent_eta1 = np.random.normal(0, 1.5)
    parent_eta2 = np.random.normal(0, 1.5)

    for p_phi, p_eta in [(parent_phi1, parent_eta1), (parent_phi2, parent_eta2)]:
        # Parent 4-momentum in the lab frame
        p_pT = parent_pt
        p_px = p_pT * np.cos(p_phi)
        p_py = p_pT * np.sin(p_phi)
        p_pz = p_pT * np.sinh(p_eta)
        p_E = np.sqrt(p_px**2 + p_py**2 + p_pz**2 + parent_mass**2)
        parent_p4 = np.array([p_E, p_px, p_py, p_pz])

        # Step 1: P → j1 + R  (j1 massless, R has mass m_R)
        p4_j1_prest, p4_R_prest = _two_body_decay_rest_frame(parent_mass, 0.0, m_R)

        # Step 2: R → j2 + j3  (both massless, isotropic in R's rest frame)
        p4_j2_rrest, p4_j3_rrest = _two_body_decay_rest_frame(m_R, 0.0, 0.0)

        # Boost j2, j3 from R's rest frame to P's rest frame using R's
        # 4-momentum as seen from P's rest frame.
        p4_j2_prest = _boost_to_lab(p4_j2_rrest, p4_R_prest)
        p4_j3_prest = _boost_to_lab(p4_j3_rrest, p4_R_prest)

        # Boost all three jets from P's rest frame to the lab frame.
        for p4_prest in (p4_j1_prest, p4_j2_prest, p4_j3_prest):
            p4_lab = _boost_to_lab(p4_prest, parent_p4)
            E_lab, px_lab, py_lab, pz_lab = p4_lab
            pt_lab = np.sqrt(px_lab**2 + py_lab**2)
            # Apply a small detector-like pT smearing (σ ≈ 5 GeV) and minimum cut
            pt_lab = max(pt_lab + np.random.normal(0, 5.0), 25.0)
            eta_lab = np.arcsinh(pz_lab / max(pt_lab, 1e-8))
            phi_lab = np.arctan2(py_lab, px_lab)
            # Near-massless partons (jet mass is a small detector effect)
            jet_mass = np.random.exponential(0.005)
            jets.append({
                "pt": pt_lab,
                "eta": eta_lab,
                "phi": phi_lab,
                "mass": jet_mass,
            })

    if include_isr:
        isr_pt = max(np.random.exponential(40.0), 25.0)
        jets.append({
            "pt": isr_pt,
            "eta": np.random.uniform(-2.5, 2.5),
            "phi": np.random.uniform(-np.pi, np.pi),
            "mass": np.random.exponential(0.003),
        })

    return _build_event_arrays(jets, include_isr)


def generate_dataset(
    output_path: str,
    n_events: int = 10000,
    parent_masses: list[float] | None = None,
    include_isr: bool = False,
    cascade_fraction: float = 0.0,
):
    """Generate a full HDF5 dataset matching the real data layout.

    Args:
        output_path: Path to write the HDF5 file.
        n_events: Number of events to generate.
        parent_masses: List of parent masses (GeV) sampled uniformly per event.
        include_isr: If True, add a 7th ISR jet to each event.
        cascade_fraction: Fraction of events to generate with the cascade
            topology P→j1+R→j1+(j2+j3) instead of direct 3-body decay.
            0.0 = all direct (default, backward-compatible), 1.0 = all cascade.
            Mixed datasets expose the model to both topologies, which is
            important for robustness when the signal contains intermediate
            on-shell resonances with varying mass splittings.
    """
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
        parent_mass = np.random.choice(parent_masses)
        use_cascade = np.random.random() < cascade_fraction
        if use_cascade:
            event = generate_event_cascade(parent_mass=parent_mass, include_isr=include_isr)
        else:
            event = generate_event(parent_mass=parent_mass, include_isr=include_isr)

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
    print(f"  Topology: {100*(1-cascade_fraction):.0f}% direct 3-body, "
          f"{100*cascade_fraction:.0f}% cascade (P→j+R→j+(jj))")
    print(f"  File size: {Path(output_path).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock HDF5 data")
    parser.add_argument("--output", type=str, default="data/mock_data.h5")
    parser.add_argument("--n-events", type=int, default=10000)
    parser.add_argument("--masses", type=float, nargs="+", default=None)
    parser.add_argument("--include-isr", action="store_true", help="Add ISR jet (7 jets)")
    parser.add_argument(
        "--cascade-fraction", type=float, default=0.0,
        help=(
            "Fraction of events generated with cascade topology P→j+R→j+(jj) "
            "(0.0 = all direct 3-body, 1.0 = all cascade).  Mixed values train "
            "the model to be robust against intermediate on-shell resonances."
        ),
    )
    args = parser.parse_args()
    generate_dataset(
        args.output, args.n_events, args.masses, args.include_isr,
        cascade_fraction=args.cascade_fraction,
    )
