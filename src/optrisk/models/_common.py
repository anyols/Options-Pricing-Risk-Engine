"""Internal numerical helpers shared by the pricing models."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

N = norm.cdf
n = norm.pdf

# Floors used instead of branching on expiry == 0 / vol == 0: the BSM/Black-76
# formulas are the correct limiting case as T, sigma -> 0+, so clamping just
# above zero keeps every call site vectorized without special-casing.
MIN_VOL = 1e-12
MIN_T = 1e-12


def as_float_arrays(*args: object) -> list[np.ndarray]:
    """Cast every argument to a float ndarray, ready for elementwise broadcasting."""
    return [np.asarray(a, dtype=float) for a in args]


def is_call(option_type: object) -> np.ndarray:
    """Boolean array, True where ``option_type == "call"`` (scalar or array-like)."""
    return np.asarray(option_type) == "call"


def scalarize(x: np.ndarray):
    """Unwrap a 0-d ndarray back to a plain Python float; pass arrays through."""
    return x.item() if isinstance(x, np.ndarray) and x.ndim == 0 else x
