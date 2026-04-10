"""
HDF5 dataset for jet assignment training.

Reads MadGraphMLProducer-format HDF5 files, converts (pt, eta, phi, mass)
to (E, px, py, pz), and constructs truth assignment labels.
"""

import glob as globmod
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .combinatorics import match_truth_to_assignment


def pt_eta_phi_mass_to_epxpypz(pt, eta, phi, mass):
    """Convert (pt, eta, phi, mass) to (E, px, py, pz).

    All inputs are numpy arrays of the same shape.
    """
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.stack([energy, px, py, pz], axis=-1)


class JetAssignmentDataset(Dataset):
    """Dataset that loads jet four-momenta and truth assignment labels from HDF5.

    Args:
        data_paths: Path(s) to HDF5 files — a single path, list, or glob pattern.
        num_jets: Number of leading jets to use (default 7).
        normalize_by_ht: If True, divide 4-vectors by event HT for scale invariance.
    """

    def __init__(
        self,
        data_paths: str | list[str],
        num_jets: int = 7,
        normalize_by_ht: bool = True,
    ):
        self.num_jets = num_jets
        self.normalize_by_ht = normalize_by_ht

        # Resolve file paths
        if isinstance(data_paths, str):
            files = sorted(globmod.glob(data_paths))
            if not files:
                files = [data_paths]
        else:
            files = list(data_paths)

        # Load and concatenate all files
        all_four_momenta = []
        all_labels = []
        all_masses = []  # true parent mass per event (for adversarial training)
        all_ht = []

        for fpath in files:
            four_mom, labels, parent_mass, ht = self._load_file(fpath)
            all_four_momenta.append(four_mom)
            all_labels.append(labels)
            all_masses.append(parent_mass)
            all_ht.append(ht)

        self.four_momenta = torch.cat(all_four_momenta, dim=0)  # (N, num_jets, 4)
        self.labels = torch.cat(all_labels, dim=0)              # (N,)
        self.parent_masses = torch.cat(all_masses, dim=0)       # (N,)
        self.ht = torch.cat(all_ht, dim=0)                     # (N,)

        # Normalize parent masses to TeV scale for stable adversarial training
        self.parent_masses = self.parent_masses / 1000.0

        # Filter out events with invalid labels
        valid = self.labels >= 0
        self.four_momenta = self.four_momenta[valid]
        self.labels = self.labels[valid]
        self.parent_masses = self.parent_masses[valid]
        self.ht = self.ht[valid]

        if self.normalize_by_ht:
            # Divide all 4-vector components by HT (broadcast over jets and components)
            ht_expanded = self.ht.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
            self.four_momenta = self.four_momenta / ht_expanded.clamp(min=1e-6)

    def _load_file(self, fpath: str):
        """Load a single HDF5 file and return processed tensors."""
        with h5py.File(fpath, "r") as f:
            jet_features = f["jet_features"][:]  # (N, 20, 7)
            event_features = f["event_features"][:]  # (N, 7)
            jet_mask = f["jet_mask"][:]  # (N, 20)

        n_events = jet_features.shape[0]

        # Extract leading num_jets jets
        jets = jet_features[:, : self.num_jets, :]  # (N, 7, 7)

        # Kinematics: pt, eta, phi, mass
        pt = jets[:, :, 0]
        eta = jets[:, :, 1]
        phi = jets[:, :, 2]
        mass = jets[:, :, 3]

        # Convert to (E, px, py, pz)
        four_mom = pt_eta_phi_mass_to_epxpypz(pt, eta, phi, mass)  # (N, 7, 4)

        # Truth info
        parent_pdg = jets[:, :, 5]  # (N, 7)
        is_signal = jets[:, :, 6]   # (N, 7)

        # HT from event features
        ht = event_features[:, 4]  # (N,)

        # Filter: require at least num_jets valid jets
        n_valid_jets = jet_mask[:, : self.num_jets].sum(axis=1)
        enough_jets = n_valid_jets >= self.num_jets

        four_mom_t = torch.tensor(four_mom, dtype=torch.float32)
        is_signal_t = torch.tensor(is_signal, dtype=torch.float32)
        parent_pdg_t = torch.tensor(parent_pdg, dtype=torch.float32)
        ht_t = torch.tensor(ht, dtype=torch.float32)

        # Match truth labels to assignment indices
        labels = match_truth_to_assignment(is_signal_t, parent_pdg_t, self.num_jets)

        # Invalidate events without enough jets
        labels[~torch.tensor(enough_jets)] = -1

        # Compute parent mass for adversarial training
        # Use the invariant mass of one truth triplet (from parent_pdg grouping)
        parent_mass = self._compute_truth_parent_mass(four_mom, is_signal, parent_pdg)
        parent_mass_t = torch.tensor(parent_mass, dtype=torch.float32)

        return four_mom_t, labels, parent_mass_t, ht_t

    def _compute_truth_parent_mass(self, four_mom, is_signal, parent_pdg):
        """Compute the true parent invariant mass for each event.

        Uses the truth grouping to sum 4-vectors and compute invariant mass.
        Returns 0 for events where truth is unavailable.
        """
        n_events = four_mom.shape[0]
        masses = np.zeros(n_events, dtype=np.float32)

        for i in range(n_events):
            sig = is_signal[i]
            pids = parent_pdg[i]
            signal_jets = np.where(sig == 1)[0]

            if len(signal_jets) < 3:
                continue

            # Group by parent
            parent_to_jets = {}
            for j in signal_jets:
                if j >= self.num_jets:
                    continue
                pid = int(pids[j])
                if pid not in parent_to_jets:
                    parent_to_jets[pid] = []
                parent_to_jets[pid].append(j)

            for pid, jets_idx in parent_to_jets.items():
                if len(jets_idx) == 3:
                    # Sum the four-momenta (E, px, py, pz)
                    p_sum = four_mom[i, jets_idx].sum(axis=0)
                    m2 = p_sum[0] ** 2 - p_sum[1] ** 2 - p_sum[2] ** 2 - p_sum[3] ** 2
                    masses[i] = np.sqrt(max(m2, 0))
                    break

        return masses

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "four_momenta": self.four_momenta[idx],    # (num_jets, 4)
            "label": self.labels[idx],                  # scalar
            "parent_mass": self.parent_masses[idx],     # scalar (for adversary)
        }
