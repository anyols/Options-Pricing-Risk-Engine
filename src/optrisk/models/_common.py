"""Internal numerical helpers shared by the pricing models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

N = norm.cdf
n = norm.pdf

# Floors used instead of branching on expiry == 0 / vol == 0: the BSM/Black-76
# formulas are the correct limiting case as T, sigma -> 0+, so clamping just
# above zero keeps every call site vectorized without special-casing.
MIN_VOL = 1e-12
MIN_T = 1e-12


def as_float_arrays(*args: object) -> list[NDArray[np.float64]]:
    """Cast every argument to a float ndarray, ready for elementwise broadcasting."""
    return [np.asarray(a, dtype=np.float64) for a in args]


def is_call(option_type: object) -> NDArray[np.bool_]:
    """Boolean array, True where ``option_type == "call"`` (scalar or array-like)."""
    result: NDArray[np.bool_] = np.asarray(option_type) == "call"
    return result


def scalarize(x: NDArray[np.float64] | np.generic) -> float | NDArray[np.float64]:
    """Unwrap a 0-d ndarray or numpy scalar (e.g. np.float64) to a plain Python
    float; pass true arrays through unchanged. Elementwise numpy arithmetic on
    0-d arrays sometimes yields a numpy scalar type rather than a 0-d ndarray
    depending on the exact operation chain, so both cases are handled here to
    keep scalar outputs consistently plain Python floats.
    """
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return float(x.item())
    if isinstance(x, np.generic):
        return float(x.item())
    return x
