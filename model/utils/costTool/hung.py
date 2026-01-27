import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

def hungarian_assign(
    C_total: np.ndarray,
    cost_max: float = 1e9,  # gating threshold
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Args:
      C_total: [M, N] cost matrix (lower is better)
      cost_max: if assigned cost > cost_max -> treat as unmatched

    Returns:
      matches: List[(i, j)]  track i matched to det j
      unmatched_tracks: List[i]
      unmatched_dets: List[j]
    """
    M, N = C_total.shape
    if M == 0 and N == 0:
        return [], [], []
    if M == 0:
        return [], [], list(range(N))
    if N == 0:
        return [], list(range(M)), []

    # Hungarian (min cost)
    row_ind, col_ind = linear_sum_assignment(C_total)

    matches: List[Tuple[int, int]] = []
    matched_tracks = set()
    matched_dets = set()

    # Apply cost gate
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        c = float(C_total[i, j])
        if c <= float(cost_max):
            matches.append((i, j))
            matched_tracks.add(i)
            matched_dets.add(j)

    unmatched_tracks = [i for i in range(M) if i not in matched_tracks]
    unmatched_dets = [j for j in range(N) if j not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets
