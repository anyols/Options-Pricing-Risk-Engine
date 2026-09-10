"""Monte Carlo pricing under geometric Brownian motion.

Variance reduction via antithetic variates (pairing each Z with -Z) and a
control variate on the terminal spot S_T, whose risk-neutral mean is known
exactly (E[S_T] = S0 e^{(r-q)T}), which is the standard textbook control
variate for vanilla payoffs under GBM (Glasserman, "Monte Carlo Methods in
Financial Engineering", Ch. 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["MonteCarloResult", "mc_price"]


@dataclass
class MonteCarloResult:
    price: float
    std_error: float
    n_paths: int


def mc_price(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    expiry: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """Price a European option by simulating terminal GBM spot prices.

    S_T = S0 * exp[(r - q - sigma^2/2) T + sigma sqrt(T) Z],  Z ~ N(0, 1).
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(half)
    if antithetic:
        z = np.concatenate([z, -z])

    drift = (rate - dividend_yield - 0.5 * vol**2) * expiry
    diffusion = vol * np.sqrt(expiry) * z
    terminal = spot * np.exp(drift + diffusion)

    if option_type == "call":
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)

    discounted_payoff = np.exp(-rate * expiry) * payoff

    if control_variate:
        control = terminal
        control_mean = spot * np.exp((rate - dividend_yield) * expiry)
        var = np.var(control)
        beta = np.cov(discounted_payoff, control)[0, 1] / var if var > 0 else 0.0
        estimate = discounted_payoff - beta * (control - control_mean)
    else:
        estimate = discounted_payoff

    price = float(np.mean(estimate))
    std_error = float(np.std(estimate, ddof=1) / np.sqrt(len(estimate)))
    return MonteCarloResult(price=price, std_error=std_error, n_paths=len(estimate))
