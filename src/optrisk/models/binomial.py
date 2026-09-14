"""Cox-Ross-Rubinstein (CRR) binomial tree pricer with American early exercise.

Converges to the Black-Scholes-Merton price as `steps -> inf` for European
options (see tests/test_binomial.py), and produces the early-exercise
premium over BSM for American options -- the reason a binomial tree earns
its place next to closed-form models in this engine.
"""

from __future__ import annotations

import numpy as np

__all__ = ["crr_price"]


def crr_price(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    expiry: float,
    option_type: str = "call",
    style: str = "american",
    steps: int = 200,
) -> float:
    """Price a European or American option via a CRR binomial tree.

    Parameters mirror the BSM model (`spot`, `strike`, `rate`,
    `dividend_yield`, `vol`, `expiry`); `style` is "european" or "american";
    `steps` is the number of time steps in the tree (more steps = closer to
    the continuous-time BSM limit, at O(steps^2) cost).
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if style not in ("european", "american"):
        raise ValueError(f"style must be 'european' or 'american', got {style!r}")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    dt = expiry / steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp((rate - dividend_yield) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"risk-neutral probability {p:.4f} out of (0, 1) for the given "
            "vol/rate/steps combination -- refine the tree (more steps) or check inputs"
        )
    discount = np.exp(-rate * dt)

    j = np.arange(steps + 1)
    terminal_spots = spot * u**j * d ** (steps - j)
    if option_type == "call":
        values = np.maximum(terminal_spots - strike, 0.0)
    else:
        values = np.maximum(strike - terminal_spots, 0.0)

    is_american = style == "american"
    for step in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1 - p) * values[:-1])
        if is_american:
            j = np.arange(step + 1)
            spots_here = spot * u**j * d ** (step - j)
            intrinsic = (
                np.maximum(spots_here - strike, 0.0) if option_type == "call" else np.maximum(strike - spots_here, 0.0)
            )
            values = np.maximum(values, intrinsic)

    return float(values[0])
