"""Correctness tests for Black-76, anchored to the BSM model via the exact
identity: pricing with forward F = S * e^{(r-q)T} and discounting at r must
reproduce the BSM price for every option, since Black-76 is BSM re-expressed
in terms of the forward.
"""

from __future__ import annotations

import numpy as np
import pytest

from optrisk.models.black76 import black76_price
from optrisk.models.black_scholes import bsm_price

CASES = [
    # spot, strike, rate, div_yield, vol, expiry
    (100.0, 100.0, 0.05, 0.00, 0.20, 1.0),
    (100.0, 120.0, 0.03, 0.01, 0.35, 0.25),
    (80.0, 70.0, 0.01, 0.00, 0.15, 2.0),
    (50.0, 55.0, 0.07, 0.03, 0.60, 0.05),
]


@pytest.mark.parametrize("spot,strike,rate,q,vol,expiry", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black76_matches_bsm_via_forward(spot, strike, rate, q, vol, expiry, option_type):
    forward = spot * np.exp((rate - q) * expiry)
    b76 = black76_price(forward, strike, rate, vol, expiry, option_type)
    bsm = bsm_price(spot, strike, rate, q, vol, expiry, option_type)
    assert b76 == pytest.approx(bsm, abs=1e-8)


@pytest.mark.parametrize("forward,strike,rate,vol,expiry", [(f, k, r, v, t) for f, k, r, _, v, t in CASES])
def test_put_call_parity(forward, strike, rate, vol, expiry):
    call = black76_price(forward, strike, rate, vol, expiry, "call")
    put = black76_price(forward, strike, rate, vol, expiry, "put")
    lhs = call - put
    rhs = np.exp(-rate * expiry) * (forward - strike)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_atm_call_equals_put():
    call = black76_price(100.0, 100.0, 0.05, 0.25, 1.0, "call")
    put = black76_price(100.0, 100.0, 0.05, 0.25, 1.0, "put")
    assert call == pytest.approx(put, abs=1e-10)


def test_vectorized_over_strikes():
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    calls = black76_price(100.0, strikes, 0.05, 0.2, 1.0, "call")
    assert np.all(np.diff(calls) < 0)  # call value decreases as strike rises
