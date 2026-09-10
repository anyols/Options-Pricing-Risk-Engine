"""Correctness tests for the Black-Scholes-Merton pricer.

Deliberately avoids hardcoding memorized reference prices as the sole
source of truth. Put-call parity is an exact algebraic identity independent
of the pricing formula's derivation, so it is the backbone check; a
well-known textbook value is included as an additional sanity check.
"""

from __future__ import annotations

import numpy as np
import pytest

from optrisk.models.black_scholes import bsm_price

CASES = [
    # spot, strike, rate, div_yield, vol, expiry
    (100.0, 100.0, 0.05, 0.00, 0.20, 1.0),
    (100.0, 120.0, 0.03, 0.01, 0.35, 0.25),
    (80.0, 70.0, 0.01, 0.00, 0.15, 2.0),
    (50.0, 55.0, 0.07, 0.03, 0.60, 0.05),
    (42.0, 40.0, 0.10, 0.00, 0.20, 0.5),
]


@pytest.mark.parametrize("spot,strike,rate,q,vol,expiry", CASES)
def test_put_call_parity(spot, strike, rate, q, vol, expiry):
    call = bsm_price(spot, strike, rate, q, vol, expiry, "call")
    put = bsm_price(spot, strike, rate, q, vol, expiry, "put")
    lhs = call - put
    rhs = spot * np.exp(-q * expiry) - strike * np.exp(-rate * expiry)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_hull_textbook_example():
    # Hull, "Options, Futures and Other Derivatives": S=42, K=40, r=10%,
    # sigma=20%, T=0.5, no dividends -> c ~= 4.76, p ~= 0.81.
    call = bsm_price(42.0, 40.0, 0.10, 0.0, 0.20, 0.5, "call")
    put = bsm_price(42.0, 40.0, 0.10, 0.0, 0.20, 0.5, "put")
    assert call == pytest.approx(4.76, abs=0.01)
    assert put == pytest.approx(0.81, abs=0.01)


def test_deep_itm_call_converges_to_forward_intrinsic():
    spot, strike, rate, q, expiry = 1000.0, 10.0, 0.05, 0.0, 1.0
    call = bsm_price(spot, strike, rate, q, 0.2, expiry, "call")
    forward_intrinsic = spot * np.exp(-q * expiry) - strike * np.exp(-rate * expiry)
    assert call == pytest.approx(forward_intrinsic, abs=1e-6)


def test_deep_otm_call_near_zero():
    call = bsm_price(10.0, 1000.0, 0.05, 0.0, 0.2, 1.0, "call")
    assert call == pytest.approx(0.0, abs=1e-6)


def test_zero_strike_put_is_worthless():
    put = bsm_price(100.0, 1e-6, 0.05, 0.0, 0.2, 1.0, "put")
    assert put == pytest.approx(0.0, abs=1e-6)


def test_price_is_monotonic_in_spot_for_calls():
    spots = np.linspace(50, 150, 25)
    calls = bsm_price(spots, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
    assert np.all(np.diff(calls) > 0)


def test_price_is_monotonic_in_vol():
    vols = np.linspace(0.05, 1.5, 25)
    calls = bsm_price(100.0, 100.0, 0.05, 0.0, vols, 1.0, "call")
    puts = bsm_price(100.0, 100.0, 0.05, 0.0, vols, 1.0, "put")
    assert np.all(np.diff(calls) > 0)
    assert np.all(np.diff(puts) > 0)


def test_vectorized_matches_scalar_loop():
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vec = bsm_price(spots, 100.0, 0.05, 0.02, 0.25, 0.75, "call")
    loop = np.array([bsm_price(float(s), 100.0, 0.05, 0.02, 0.25, 0.75, "call") for s in spots])
    np.testing.assert_allclose(vec, loop, atol=1e-12)


def test_mixed_option_type_array():
    prices = bsm_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, np.array(["call", "put"]))
    call = bsm_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
    put = bsm_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
    np.testing.assert_allclose(prices, [call, put])
