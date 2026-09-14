"""Round-trip tests for the implied volatility solver."""

from __future__ import annotations

from typing import cast

import pytest

from optrisk.models.black76 import black76_price
from optrisk.models.black_scholes import bsm_price
from optrisk.models.implied_vol import (
    ImpliedVolError,
    black76_implied_vol,
    bsm_implied_vol,
    implied_vol,
)

TRUE_VOLS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.90, 1.40]


@pytest.mark.parametrize("true_vol", TRUE_VOLS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bsm_round_trip(true_vol, option_type):
    spot, strike, rate, q, expiry = 100.0, 105.0, 0.04, 0.01, 0.75
    price = bsm_price(spot, strike, rate, q, true_vol, expiry, option_type)
    recovered = bsm_implied_vol(price, spot, strike, rate, q, expiry, option_type)
    assert recovered == pytest.approx(true_vol, abs=1e-6)


@pytest.mark.parametrize("true_vol", TRUE_VOLS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black76_round_trip(true_vol, option_type):
    forward, strike, rate, expiry = 100.0, 95.0, 0.03, 0.5
    price = black76_price(forward, strike, rate, true_vol, expiry, option_type)
    recovered = black76_implied_vol(price, forward, strike, rate, expiry, option_type)
    assert recovered == pytest.approx(true_vol, abs=1e-6)


def test_round_trip_near_lower_bound():
    # zero-rate ATM so a tiny vol still produces a non-degenerate price: with a
    # nonzero rate, drift alone pushes d1/d2 into saturation (N(d1)=N(d2)=1)
    # regardless of vol, making any two small vols indistinguishable in price.
    spot, strike, rate, q, expiry = 100.0, 100.0, 0.0, 0.0, 1.0
    true_vol = 1e-3
    price = bsm_price(spot, strike, rate, q, true_vol, expiry, "call")
    recovered = bsm_implied_vol(price, spot, strike, rate, q, expiry, "call")
    assert recovered == pytest.approx(true_vol, rel=1e-2)


def test_round_trip_near_upper_bound():
    spot, strike, rate, q, expiry = 100.0, 100.0, 0.02, 0.0, 1.0
    true_vol = 4.5
    price = bsm_price(spot, strike, rate, q, true_vol, expiry, "call")
    recovered = bsm_implied_vol(price, spot, strike, rate, q, expiry, "call")
    assert recovered == pytest.approx(true_vol, rel=1e-2)


def test_unreachable_price_raises():
    # a price above the max attainable (vol -> infinity bound) cannot be matched
    with pytest.raises(ImpliedVolError):
        bsm_implied_vol(price=1000.0, spot=100.0, strike=100.0, rate=0.05, dividend_yield=0.0, expiry=1.0)


def test_generic_solver_works_with_arbitrary_pricer():
    # sanity check that `implied_vol` itself is model-agnostic, not just its wrappers
    def toy_pricer(vol: float) -> float:
        return cast(float, bsm_price(100.0, 90.0, 0.05, 0.0, vol, 1.0, "call"))

    target = toy_pricer(0.33)
    recovered = implied_vol(target, toy_pricer)
    assert recovered == pytest.approx(0.33, abs=1e-6)
