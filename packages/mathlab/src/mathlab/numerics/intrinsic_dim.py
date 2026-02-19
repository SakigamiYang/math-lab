from __future__ import annotations

import numpy as np


def spectral_gap_dim_ignore_padding(
    avg_spectrum: np.ndarray,
    *,
    tol: float = 1e-10,
    eps: float = 1e-12,
) -> dict:
    """
    Estimate intrinsic dimension via log-gap argmax, while ignoring trailing
    zero-padding (or near-zero values).

    avg_spectrum:
      1D array of eigenvalue ratios (should sum to 1 in ideal case).
      May be padded with zeros to a larger length.

    Returns:
      {
        "d_hat": int,
        "effective_dim": int,   # prefix length used for gap computation
        "gaps": list[float],
        "gap_value": float | None,
      }
    """
    avg_spectrum = np.asarray(avg_spectrum, dtype=float)

    positive = np.where(avg_spectrum > tol)[0]
    if len(positive) < 2:
        return {"d_hat": 0, "effective_dim": len(positive), "gaps": [], "gap_value": None}

    L = int(positive[-1] + 1)
    spec = avg_spectrum[:L]

    logv = np.log(spec + eps)
    gaps = logv[:-1] - logv[1:]
    d_hat = int(np.argmax(gaps) + 1)

    return {
        "d_hat": d_hat,
        "effective_dim": L,
        "gaps": gaps.tolist(),
        "gap_value": float(gaps[d_hat - 1]),
    }
