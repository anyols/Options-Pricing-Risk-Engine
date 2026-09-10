"""Model-agnostic Greeks via central finite differences.

Works with any pricer exposing the keyword signature
``pricer(spot=..., vol=..., expiry=..., rate=...) -> price``, so the exact
same engine computes every Greek -- including third-order ones with no
convenient closed form -- for BSM, Black-76, binomial, Monte Carlo or Heston
alike. This is a deliberate design choice: it is both (a) the *only* Greeks
source for models without analytical formulas, and (b) the cross-validation
oracle for the closed-form Greeks in :mod:`optrisk.greeks.analytical`.

For Black-76 (and any forward-quoted model) pass the forward price as the
``spot`` keyword -- it is simply the label for "the underlying state
variable being bumped".
"""

from __future__ import annotations

from typing import Callable, Optional

from optrisk.greeks.types import Greeks

__all__ = ["numerical_greeks"]


def numerical_greeks(
    pricer: Callable[..., float],
    *,
    spot: float,
    vol: float,
    expiry: float,
    rate: float,
    spot_bump: Optional[float] = None,
    vol_bump: float = 1e-4,
    expiry_bump: float = 1e-4,
    rate_bump: float = 1e-5,
) -> Greeks:
    """Full Greeks set for ``pricer`` via central finite differences.

    Bump sizes default to values that work well for equity/index-scale
    spots (~O(10-1000)) and annualized vol/rate; pass ``spot_bump``
    explicitly for very small or very large underlyings.
    """
    h_s = spot_bump if spot_bump is not None else max(spot * 1e-3, 1e-3)
    h_v = vol_bump
    h_t = min(expiry_bump, expiry / 4) if expiry > 0 else expiry_bump
    h_r = rate_bump

    base = dict(spot=spot, vol=vol, expiry=expiry, rate=rate)

    def p(**overrides: float) -> float:
        kwargs = dict(base)
        kwargs.update(overrides)
        return pricer(**kwargs)

    v0 = p()

    # first order
    delta = (p(spot=spot + h_s) - p(spot=spot - h_s)) / (2 * h_s)
    vega = (p(vol=vol + h_v) - p(vol=vol - h_v)) / (2 * h_v)
    theta = -(p(expiry=expiry + h_t) - p(expiry=expiry - h_t)) / (2 * h_t)
    rho = (p(rate=rate + h_r) - p(rate=rate - h_r)) / (2 * h_r)

    # second order
    gamma = (p(spot=spot + h_s) - 2 * v0 + p(spot=spot - h_s)) / h_s**2
    volga = (p(vol=vol + h_v) - 2 * v0 + p(vol=vol - h_v)) / h_v**2
    vanna = (
        p(spot=spot + h_s, vol=vol + h_v)
        - p(spot=spot + h_s, vol=vol - h_v)
        - p(spot=spot - h_s, vol=vol + h_v)
        + p(spot=spot - h_s, vol=vol - h_v)
    ) / (4 * h_s * h_v)
    charm = -(
        p(spot=spot + h_s, expiry=expiry + h_t)
        - p(spot=spot + h_s, expiry=expiry - h_t)
        - p(spot=spot - h_s, expiry=expiry + h_t)
        + p(spot=spot - h_s, expiry=expiry - h_t)
    ) / (4 * h_s * h_t)

    # third order: pure d^3V/dS^3, and mixed d^3V/dS^2 dY for Y in {vol, expiry}
    speed = (
        p(spot=spot + 2 * h_s)
        - 2 * p(spot=spot + h_s)
        + 2 * p(spot=spot - h_s)
        - p(spot=spot - 2 * h_s)
    ) / (2 * h_s**3)

    zomma = (
        p(spot=spot + h_s, vol=vol + h_v)
        - 2 * p(spot=spot, vol=vol + h_v)
        + p(spot=spot - h_s, vol=vol + h_v)
        - p(spot=spot + h_s, vol=vol - h_v)
        + 2 * p(spot=spot, vol=vol - h_v)
        - p(spot=spot - h_s, vol=vol - h_v)
    ) / (2 * h_s**2 * h_v)

    color = -(
        p(spot=spot + h_s, expiry=expiry + h_t)
        - 2 * p(spot=spot, expiry=expiry + h_t)
        + p(spot=spot - h_s, expiry=expiry + h_t)
        - p(spot=spot + h_s, expiry=expiry - h_t)
        + 2 * p(spot=spot, expiry=expiry - h_t)
        - p(spot=spot - h_s, expiry=expiry - h_t)
    ) / (2 * h_s**2 * h_t)

    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        vanna=vanna,
        volga=volga,
        charm=charm,
        speed=speed,
        zomma=zomma,
        color=color,
    )
