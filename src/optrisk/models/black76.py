"""Black-76 pricing for European options on forwards/futures (Black, 1976).

The underlying quoted price is already a forward/futures price F, i.e. a
risk-neutral expectation of the settlement price, so there is no separate
drift term: the whole payoff is discounted at the risk-free rate.

Formula (call):  C = e^{-rT} [F N(d1) - K N(d2)]
Formula (put):    P = e^{-rT} [K N(-d2) - F N(-d1)]
where              d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt(T))
                   d2 = d1 - sigma sqrt(T)

Identity: pricing a BSM option via Black-76 with F = S e^{(r-q)T} reproduces
the BSM price exactly (see tests/test_black76.py) -- Black-76 is what BSM
collapses to once the drift has already been folded into the forward price.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from optrisk.models._common import MIN_T, MIN_VOL, N, as_float_arrays, is_call, scalarize

__all__ = ["black76_d1_d2", "black76_price"]

ArrayOrFloat = Union[float, np.ndarray]


def black76_d1_d2(
    forward: ArrayOrFloat, strike: ArrayOrFloat, vol: ArrayOrFloat, expiry: ArrayOrFloat
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (d1, d2) pair used throughout the Black-76 formula and its Greeks."""
    forward, strike, vol, expiry = as_float_arrays(forward, strike, vol, expiry)
    vol = np.maximum(vol, MIN_VOL)
    expiry = np.maximum(expiry, MIN_T)
    vol_sqrt_t = vol * np.sqrt(expiry)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * expiry) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def black76_price(
    forward: ArrayOrFloat,
    strike: ArrayOrFloat,
    rate: ArrayOrFloat,
    vol: ArrayOrFloat,
    expiry: ArrayOrFloat,
    option_type: Union[str, np.ndarray] = "call",
) -> ArrayOrFloat:
    """European option price on a futures/forward contract under Black-76.

    Parameters
    ----------
    forward, strike, rate, vol, expiry
        Forward/futures price F, strike K, discount rate r, annualized
        volatility sigma, time to expiry T (years). Scalars or
        broadcastable numpy arrays.
    option_type
        "call", "put", or an array of either.
    """
    fwd_a, strike_a, rate_a, vol_a, exp_a = as_float_arrays(forward, strike, rate, vol, expiry)
    d1, d2 = black76_d1_d2(fwd_a, strike_a, vol_a, exp_a)
    disc_r = np.exp(-rate_a * exp_a)
    call = disc_r * (fwd_a * N(d1) - strike_a * N(d2))
    put = disc_r * (strike_a * N(-d2) - fwd_a * N(-d1))
    price = np.where(is_call(option_type), call, put)
    return scalarize(price)
