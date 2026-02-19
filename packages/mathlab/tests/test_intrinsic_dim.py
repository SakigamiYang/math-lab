from __future__ import annotations

import numpy as np
from mathlab.numerics.intrinsic_dim import spectral_gap_dim_ignore_padding


def test_ignore_padding_zeros_does_not_force_d_hat_to_effective_dim():
    # Spectrum for a 3D ambient space, padded to 10 dims
    # A clear gap is between component 2 and 3 => d_hat should be 2
    spec3 = np.array([0.70, 0.25, 0.05], dtype=float)
    padded = np.pad(spec3, (0, 7), constant_values=0.0)

    out = spectral_gap_dim_ignore_padding(padded)
    assert out["effective_dim"] == 3
    assert out["d_hat"] == 2


def test_all_zero_or_single_positive_returns_d_hat_0():
    out1 = spectral_gap_dim_ignore_padding(np.zeros(10))
    assert out1["d_hat"] == 0

    out2 = spectral_gap_dim_ignore_padding(np.array([1.0] + [0.0] * 9))
    assert out2["d_hat"] == 0
