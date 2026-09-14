"""Model-agnostic implied volatility solver, plus BSM/Black-76 convenience wrappers.

The core `implied_vol` solver is derivative-free (Brent's method on a
bracketed root), so the exact same function inverts *any* pricer -- BSM,
Black-76, binomial, Monte Carlo, or Heston -- by partially applying every
parameter except volatility. This avoids needing a closed-form vega (which
doesn't exist for a binomial tree or Monte Carlo estimator) and sidesteps
the classic Newton-Raphson blow-up when vega is near zero deep ITM/OTM.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from scipy.optimize import brentq

from optrisk.models.black76 import black76_price
from optrisk.models.black_scholes import bsm_price

__all__ = ["ImpliedVolError", "black76_implied_vol", "bsm_implied_vol", "implied_vol"]


class ImpliedVolError(ValueError):
    """Raised when a target price cannot be matched by any vol in the search bounds."""


def implied_vol(
    target_price: float,
    pricer: Callable[[float], float],
    vol_bounds: tuple[float, float] = (1e-6, 5.0),
    *,
    xtol: float = 1e-8,
) -> float:
    """Solve ``pricer(vol) == target_price`` for ``vol`` via Brent's method.

    Parameters
    ----------
    target_price
        The observed/market price to match.
    pricer
        A callable taking volatility as its only argument and returning a
        price -- typically a `functools.partial` of a pricing function with
        every other parameter already fixed.
    vol_bounds
        Search interval for volatility. Must bracket the root, i.e. the
        target price must lie between the prices at the two bounds.
    """
    lo, hi = vol_bounds
    f_lo, f_hi = pricer(lo) - target_price, pricer(hi) - target_price
    if f_lo * f_hi > 0:
        raise ImpliedVolError(
            f"target_price={target_price!r} is not attainable for vol in {vol_bounds} "
            f"(model prices range [{f_lo + target_price:.6g}, {f_hi + target_price:.6g}]); "
            "check inputs for arbitrage or widen vol_bounds."
        )
    return cast(float, brentq(lambda v: pricer(v) - target_price, lo, hi, xtol=xtol))


def bsm_implied_vol(
    price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    expiry: float,
    option_type: str = "call",
    vol_bounds: tuple[float, float] = (1e-6, 5.0),
    xtol: float = 1e-8,
) -> float:
    """Implied volatility under Black-Scholes-Merton."""

    def pricer(vol: float) -> float:
        return cast(float, bsm_price(spot, strike, rate, dividend_yield, vol, expiry, option_type))

    return implied_vol(price, pricer, vol_bounds, xtol=xtol)


def black76_implied_vol(
    price: float,
    forward: float,
    strike: float,
    rate: float,
    expiry: float,
    option_type: str = "call",
    vol_bounds: tuple[float, float] = (1e-6, 5.0),
    xtol: float = 1e-8,
) -> float:
    """Implied volatility under Black-76."""

    def pricer(vol: float) -> float:
        return cast(float, black76_price(forward, strike, rate, vol, expiry, option_type))

    return implied_vol(price, pricer, vol_bounds, xtol=xtol)
