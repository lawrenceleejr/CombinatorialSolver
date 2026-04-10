"""
Enumerate all possible jet-to-parent assignments and provide truth matching.

Supports two modes:
  - 6 jets (no ISR): C(6,3)/2 = 10 assignments — partition into two groups of 3
  - 7 jets (with ISR): C(7,1) * C(6,3)/2 = 70 assignments — pick ISR + partition
"""

from itertools import combinations

import torch


def enumerate_assignments(num_jets: int = 7) -> list[tuple[int | None, tuple[int, ...], tuple[int, ...]]]:
    """Enumerate all (isr_idx, group1, group2) assignments.

    The two groups are unordered (symmetric parents), so we canonicalize
    by placing the group with the smaller first index first.

    For 6 jets: isr_idx is None, returns 10 assignments.
    For 7 jets: isr_idx is 0-6, returns 70 assignments.

    Returns:
        List of tuples: (isr_index_or_None, group1_indices, group2_indices)
    """
    assignments = []

    if num_jets == 6:
        # No ISR — just partition 6 jets into two groups of 3
        all_jets = list(range(6))
        seen = set()
        for group1 in combinations(all_jets, 3):
            group2 = tuple(j for j in all_jets if j not in group1)
            canon = (min(group1, group2), max(group1, group2))
            if canon not in seen:
                seen.add(canon)
                assignments.append((None, canon[0], canon[1]))
        return assignments

    # 7+ jets: choose ISR then partition remaining 6
    for isr in range(num_jets):
        remaining = [j for j in range(num_jets) if j != isr]
        seen = set()
        for group1 in combinations(remaining, 3):
            group2 = tuple(j for j in remaining if j not in group1)
            canon = (min(group1, group2), max(group1, group2))
            if canon not in seen:
                seen.add(canon)
                assignments.append((isr, canon[0], canon[1]))
    return assignments


def build_assignment_tensors(num_jets: int = 7) -> dict[str, torch.Tensor]:
    """Build index tensors for efficient batched gathering.

    Returns dict with:
        - isr_indices: (N,) int tensor of ISR jet index (-1 if no ISR)
        - group1_indices: (N, 3) int tensor of group1 jet indices
        - group2_indices: (N, 3) int tensor of group2 jet indices
        - num_assignments: int
    """
    assignments = enumerate_assignments(num_jets)
    isr = torch.tensor(
        [a[0] if a[0] is not None else -1 for a in assignments],
        dtype=torch.long,
    )
    g1 = torch.tensor([list(a[1]) for a in assignments], dtype=torch.long)
    g2 = torch.tensor([list(a[2]) for a in assignments], dtype=torch.long)
    return {
        "isr_indices": isr,
        "group1_indices": g1,
        "group2_indices": g2,
        "num_assignments": len(assignments),
    }


def match_truth_groups(
    truth_g1: list[int],
    truth_g2: list[int],
    num_jets: int,
    truth_isr: int | None = None,
) -> int:
    """Find the assignment index matching truth group indices.

    Args:
        truth_g1: List of 3 jet indices for group 1.
        truth_g2: List of 3 jet indices for group 2.
        num_jets: Total number of jets (6 or 7).
        truth_isr: ISR jet index (None for 6-jet events).

    Returns:
        Assignment index, or -1 if no match.
    """
    assignments = enumerate_assignments(num_jets)
    g1 = tuple(sorted(truth_g1))
    g2 = tuple(sorted(truth_g2))
    # Canonical: smaller first element first
    canon = (min(g1, g2), max(g1, g2))

    for idx, (a_isr, a_g1, a_g2) in enumerate(assignments):
        isr_match = (truth_isr is None and a_isr is None) or (a_isr == truth_isr)
        if isr_match and (a_g1, a_g2) == canon:
            return idx

    return -1
