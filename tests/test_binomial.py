"""Correctness tests for the CRR binomial tree."""

from __future__ import annotations

import pytest

from optrisk.models.binomial import crr_price
from optrisk.models.black_scholes import bsm_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_binomial_converges_to_bsm(option_type):
    spot, strike, rate, q, vol, expiry = 100.0, 100.0, 0.05, 0.02, 0.25, 1.0
    bsm = bsm_price(spot, strike, rate, q, vol, expiry, option_type)

    errors = []
    for steps in (25, 100, 400, 1600):
        tree = crr_price(spot, strike, rate, q, vol, expiry, option_type, style="european", steps=steps)
        errors.append(abs(tree - bsm))

    # error should shrink as the tree is refined, and be small at the finest resolution
    assert errors[-1] < 0.01
    assert errors[-1] < errors[0]


def test_american_call_equals_european_when_no_dividends():
    # with no dividends, early exercise of an American call is never optimal
    spot, strike, rate, q, vol, expiry = 100.0, 90.0, 0.05, 0.0, 0.3, 1.0
    american = crr_price(spot, strike, rate, q, vol, expiry, "call", style="american", steps=500)
    european = crr_price(spot, strike, rate, q, vol, expiry, "call", style="european", steps=500)
    assert american == pytest.approx(european, abs=1e-6)


def test_american_put_has_positive_early_exercise_premium():
    # deep ITM put with a high rate: early exercise is valuable
    spot, strike, rate, q, vol, expiry = 60.0, 100.0, 0.10, 0.0, 0.2, 1.0
    american = crr_price(spot, strike, rate, q, vol, expiry, "put", style="american", steps=500)
    european = crr_price(spot, strike, rate, q, vol, expiry, "put", style="european", steps=500)
    assert american > european + 1e-3


def test_american_call_with_dividends_has_early_exercise_premium():
    # a high dividend yield makes early exercise of an American call valuable
    spot, strike, rate, q, vol, expiry = 100.0, 90.0, 0.02, 0.08, 0.2, 1.0
    american = crr_price(spot, strike, rate, q, vol, expiry, "call", style="american", steps=500)
    european = crr_price(spot, strike, rate, q, vol, expiry, "call", style="european", steps=500)
    assert american > european + 1e-3


def test_american_price_at_least_intrinsic():
    spot, strike, rate, q, vol, expiry = 100.0, 110.0, 0.05, 0.0, 0.25, 1.0
    price = crr_price(spot, strike, rate, q, vol, expiry, "put", style="american", steps=200)
    assert price >= max(strike - spot, 0.0) - 1e-9


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        crr_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, option_type="straddle")


def test_invalid_style_raises():
    with pytest.raises(ValueError):
        crr_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, style="bermudan")


def test_single_step_runs():
    price = crr_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, steps=1)
    assert price > 0
