"""Closed-form Greeks for Black-Scholes-Merton and Black-76.

Delta, Gamma, Vega, Theta, Rho, Vanna and Volga all have well-established
closed forms and are computed directly here. Charm, Speed, Zomma and Color
are supplied by the ``*_full_greeks`` variants, which fill them in via the
central finite-difference engine in :mod:`optrisk.greeks.numerical` rather
than relying on easy-to-mis-sign memorized third-order formulas.

Black-76 Greeks are derived from the identity
``Black76(F, K, r, sigma, T) = e^{-rT} * BSM_undiscounted_call(F, K, sigma, T)``
(Black-76 is BSM with the discounting factored out, expressed in the forward),
which gives simple, self-consistent forms -- e.g. Rho = -T * Price exactly,
since the forward carries no direct r-dependence once discounting is
factored out. See tests/test_greeks.py for the put-call-parity and
finite-difference cross-checks that validate this.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from optrisk.greeks.numerical import numerical_greeks
from optrisk.greeks.types import Greeks
from optrisk.models._common import MIN_T, MIN_VOL, N, as_float_arrays, is_call, n, scalarize
from optrisk.models.black76 import black76_d1_d2, black76_price
from optrisk.models.black_scholes import bsm_d1_d2, bsm_price

__all__ = ["black76_full_greeks", "black76_greeks", "bsm_full_greeks", "bsm_greeks"]

ArrayOrFloat = float | np.ndarray


def _scalar(x: NDArray[np.float64] | np.generic) -> float:
    """`scalarize`, narrowed to `float` for the (scalar-input) `Greeks` constructors below.

    `bsm_greeks`/`black76_greeks` are genuinely vectorized internally (see
    `optrisk.risk.hedging`, which calls them with an array of spots to get an
    array-valued `.delta` in one shot), but the `Greeks` dataclass fields are
    typed strictly `float` since every other consumer (Position/Portfolio
    aggregation, the scenario/hedging engines, the dashboard) does scalar
    arithmetic on them. This cast documents that boundary rather than
    silently widening `Greeks` to a float-or-array type everywhere.
    """
    return cast(float, scalarize(x))


def bsm_greeks(
    spot: ArrayOrFloat,
    strike: ArrayOrFloat,
    rate: ArrayOrFloat,
    dividend_yield: ArrayOrFloat,
    vol: ArrayOrFloat,
    expiry: ArrayOrFloat,
    option_type: str | np.ndarray = "call",
) -> Greeks:
    """Delta, Gamma, Vega, Theta, Rho, Vanna, Volga under Black-Scholes-Merton.

    Vectorized: array inputs produce a `Greeks` whose fields are arrays.
    """
    spot, strike, rate, q, vol, expiry = as_float_arrays(spot, strike, rate, dividend_yield, vol, expiry)
    vol = np.maximum(vol, MIN_VOL)
    expiry = np.maximum(expiry, MIN_T)
    d1, d2 = bsm_d1_d2(spot, strike, rate, q, vol, expiry)
    sqrt_t = np.sqrt(expiry)
    disc_q = np.exp(-q * expiry)
    disc_r = np.exp(-rate * expiry)
    pdf_d1 = n(d1)
    call_mask = is_call(option_type)

    delta = np.where(call_mask, disc_q * N(d1), -disc_q * N(-d1))
    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t
    time_decay = -(spot * disc_q * pdf_d1 * vol) / (2 * sqrt_t)
    theta = np.where(
        call_mask,
        time_decay - rate * strike * disc_r * N(d2) + q * spot * disc_q * N(d1),
        time_decay + rate * strike * disc_r * N(-d2) - q * spot * disc_q * N(-d1),
    )
    rho = np.where(call_mask, strike * expiry * disc_r * N(d2), -strike * expiry * disc_r * N(-d2))
    vanna = -disc_q * pdf_d1 * d2 / vol
    volga = vega * d1 * d2 / vol

    return Greeks(
        delta=_scalar(delta),
        gamma=_scalar(gamma),
        vega=_scalar(vega),
        theta=_scalar(theta),
        rho=_scalar(rho),
        vanna=_scalar(vanna),
        volga=_scalar(volga),
    )


def bsm_full_greeks(
    spot: float, strike: float, rate: float, dividend_yield: float, vol: float, expiry: float, option_type: str = "call"
) -> Greeks:
    """`bsm_greeks` plus Charm/Speed/Zomma/Color from the numerical engine. Scalar inputs only."""
    greeks = bsm_greeks(spot, strike, rate, dividend_yield, vol, expiry, option_type)

    def pricer(spot: float, vol: float, expiry: float, rate: float) -> float:
        return cast(float, bsm_price(spot, strike, rate, dividend_yield, vol, expiry, option_type))

    higher = numerical_greeks(pricer, spot=spot, vol=vol, expiry=expiry, rate=rate)
    greeks.charm, greeks.speed, greeks.zomma, greeks.color = (
        higher.charm,
        higher.speed,
        higher.zomma,
        higher.color,
    )
    return greeks


def black76_greeks(
    forward: ArrayOrFloat,
    strike: ArrayOrFloat,
    rate: ArrayOrFloat,
    vol: ArrayOrFloat,
    expiry: ArrayOrFloat,
    option_type: str | np.ndarray = "call",
) -> Greeks:
    """Delta, Gamma, Vega, Theta, Rho, Vanna, Volga under Black-76."""
    forward, strike, rate, vol, expiry = as_float_arrays(forward, strike, rate, vol, expiry)
    vol = np.maximum(vol, MIN_VOL)
    expiry = np.maximum(expiry, MIN_T)
    d1, d2 = black76_d1_d2(forward, strike, vol, expiry)
    sqrt_t = np.sqrt(expiry)
    disc_r = np.exp(-rate * expiry)
    pdf_d1 = n(d1)
    call_mask = is_call(option_type)

    call_price = disc_r * (forward * N(d1) - strike * N(d2))
    put_price = disc_r * (strike * N(-d2) - forward * N(-d1))
    price = np.where(call_mask, call_price, put_price)

    delta = np.where(call_mask, disc_r * N(d1), -disc_r * N(-d1))
    gamma = disc_r * pdf_d1 / (forward * vol * sqrt_t)
    vega = forward * disc_r * pdf_d1 * sqrt_t
    vol_decay = forward * disc_r * pdf_d1 * vol / (2 * sqrt_t)
    theta = rate * price - vol_decay
    rho = -expiry * price
    vanna = -disc_r * pdf_d1 * d2 / vol
    volga = vega * d1 * d2 / vol

    return Greeks(
        delta=_scalar(delta),
        gamma=_scalar(gamma),
        vega=_scalar(vega),
        theta=_scalar(theta),
        rho=_scalar(rho),
        vanna=_scalar(vanna),
        volga=_scalar(volga),
    )


def black76_full_greeks(
    forward: float, strike: float, rate: float, vol: float, expiry: float, option_type: str = "call"
) -> Greeks:
    """`black76_greeks` plus Charm/Speed/Zomma/Color from the numerical engine. Scalar inputs only."""
    greeks = black76_greeks(forward, strike, rate, vol, expiry, option_type)

    def pricer(spot: float, vol: float, expiry: float, rate: float) -> float:
        return cast(float, black76_price(spot, strike, rate, vol, expiry, option_type))

    higher = numerical_greeks(pricer, spot=forward, vol=vol, expiry=expiry, rate=rate)
    greeks.charm, greeks.speed, greeks.zomma, greeks.color = (
        higher.charm,
        higher.speed,
        higher.zomma,
        higher.color,
    )
    return greeks
