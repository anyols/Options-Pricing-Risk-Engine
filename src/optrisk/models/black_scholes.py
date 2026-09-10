"""Black-Scholes-Merton pricing for European options with a continuous carry/dividend yield.

Every function is vectorized over numpy-broadcastable inputs, so a whole
strike ladder, spot grid, or shocked-scenario grid can be priced in a single
call: pass arrays for ``spot`` and/or ``vol`` and get an array of prices back.

Formula (call):    C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
Formula (put):      P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
where                d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
                     d2 = d1 - sigma sqrt(T)
"""

from __future__ import annotations

from typing import Union

import numpy as np

from optrisk.models._common import MIN_T, MIN_VOL, N, as_float_arrays, is_call, scalarize

__all__ = ["bsm_d1_d2", "bsm_price"]

ArrayOrFloat = Union[float, np.ndarray]


def bsm_d1_d2(
    spot: ArrayOrFloat,
    strike: ArrayOrFloat,
    rate: ArrayOrFloat,
    dividend_yield: ArrayOrFloat,
    vol: ArrayOrFloat,
    expiry: ArrayOrFloat,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (d1, d2) pair used throughout the BSM formula and its Greeks."""
    spot, strike, rate, dividend_yield, vol, expiry = as_float_arrays(
        spot, strike, rate, dividend_yield, vol, expiry
    )
    vol = np.maximum(vol, MIN_VOL)
    expiry = np.maximum(expiry, MIN_T)
    vol_sqrt_t = vol * np.sqrt(expiry)
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * vol**2) * expiry) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bsm_price(
    spot: ArrayOrFloat,
    strike: ArrayOrFloat,
    rate: ArrayOrFloat,
    dividend_yield: ArrayOrFloat,
    vol: ArrayOrFloat,
    expiry: ArrayOrFloat,
    option_type: Union[str, np.ndarray] = "call",
) -> ArrayOrFloat:
    """European option price under Black-Scholes-Merton.

    Parameters
    ----------
    spot, strike, rate, dividend_yield, vol, expiry
        S, K, continuously-compounded risk-free rate r, continuous dividend
        yield q, annualized volatility sigma, and time to expiry T (years).
        Any of these may be scalars or broadcastable numpy arrays.
    option_type
        "call", "put", or an array of either for pricing a mixed book in one
        vectorized call.
    """
    spot_a, strike_a, rate_a, div_a, vol_a, exp_a = as_float_arrays(
        spot, strike, rate, dividend_yield, vol, expiry
    )
    d1, d2 = bsm_d1_d2(spot_a, strike_a, rate_a, div_a, vol_a, exp_a)
    disc_q = np.exp(-div_a * exp_a)
    disc_r = np.exp(-rate_a * exp_a)
    call = spot_a * disc_q * N(d1) - strike_a * disc_r * N(d2)
    put = strike_a * disc_r * N(-d2) - spot_a * disc_q * N(-d1)
    price = np.where(is_call(option_type), call, put)
    return scalarize(price)
