"""Tests for instrument model-dispatch, position scaling, and portfolio aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from optrisk.greeks.analytical import black76_full_greeks, bsm_full_greeks
from optrisk.instruments.option import MarketEnv, OptionSpec, Stock
from optrisk.instruments.portfolio import Portfolio, Position
from optrisk.market.sample_data import build_demo_portfolio
from optrisk.models.binomial import crr_price
from optrisk.models.black76 import black76_price
from optrisk.models.black_scholes import bsm_price

EQ = MarketEnv(spot=100.0, rate=0.04, vol=0.25, dividend_yield=0.01)
FUT = MarketEnv(spot=100.0, rate=0.04, vol=0.30, dividend_yield=0.0)


def test_european_equity_auto_resolves_to_bsm():
    spec = OptionSpec(option_type="call", strike=105.0, expiry=0.5, style="european", underlying_type="equity")
    assert spec.price(EQ) == pytest.approx(bsm_price(EQ.spot, 105.0, EQ.rate, EQ.dividend_yield, EQ.vol, 0.5, "call"))


def test_european_future_auto_resolves_to_black76():
    spec = OptionSpec(option_type="put", strike=95.0, expiry=0.5, style="european", underlying_type="future")
    assert spec.price(FUT) == pytest.approx(black76_price(FUT.spot, 95.0, FUT.rate, FUT.vol, 0.5, "put"))


def test_american_auto_resolves_to_binomial():
    spec = OptionSpec(option_type="put", strike=105.0, expiry=0.5, style="american", underlying_type="equity")
    expected = crr_price(EQ.spot, 105.0, EQ.rate, EQ.dividend_yield, EQ.vol, 0.5, "put", style="american", steps=200)
    assert spec.price(EQ, steps=200) == pytest.approx(expected)


def test_greeks_auto_matches_bsm_full_greeks():
    spec = OptionSpec(option_type="call", strike=105.0, expiry=0.5, style="european", underlying_type="equity")
    expected = bsm_full_greeks(EQ.spot, 105.0, EQ.rate, EQ.dividend_yield, EQ.vol, 0.5, "call")
    actual = spec.greeks(EQ)
    assert actual.delta == pytest.approx(expected.delta, rel=1e-6)
    assert actual.gamma == pytest.approx(expected.gamma, rel=1e-6)


def test_greeks_auto_matches_black76_full_greeks():
    spec = OptionSpec(option_type="put", strike=95.0, expiry=0.5, style="european", underlying_type="future")
    expected = black76_full_greeks(FUT.spot, 95.0, FUT.rate, FUT.vol, 0.5, "put")
    actual = spec.greeks(FUT)
    assert actual.delta == pytest.approx(expected.delta, rel=1e-6)
    assert actual.vega == pytest.approx(expected.vega, rel=1e-6)


def test_binomial_override_converges_to_bsm_for_european_option():
    # explicitly forcing model="binomial" on a European-style spec is an
    # end-to-end check that the numerical-Greeks fallback plumbing is wired
    # correctly, not just the binomial tree or the FD engine in isolation.
    spec = OptionSpec(option_type="call", strike=105.0, expiry=0.5, style="european", underlying_type="equity")
    bsm_g = bsm_full_greeks(EQ.spot, 105.0, EQ.rate, EQ.dividend_yield, EQ.vol, 0.5, "call")
    binom_g = spec.greeks(EQ, model="binomial", steps=400)
    assert binom_g.delta == pytest.approx(bsm_g.delta, abs=2e-3)
    assert binom_g.gamma == pytest.approx(bsm_g.gamma, abs=2e-3)
    # vega from bump-and-reprice on a discrete tree is inherently noisier
    # (bumping vol also reshapes the tree spacing itself); 1.5% relative is
    # a fair bar for a 400-step tree rather than sub-percent absolute.
    assert binom_g.vega == pytest.approx(bsm_g.vega, rel=1.5e-2)


def test_monte_carlo_rejects_american():
    spec = OptionSpec(option_type="put", strike=105.0, expiry=0.5, style="american", underlying_type="equity")
    with pytest.raises(ValueError):
        spec.price(EQ, model="monte_carlo")


def test_stock_position_price_and_greeks():
    pos = Position(instrument=Stock(), quantity=50, market=EQ, label="hedge", multiplier=1)
    assert pos.price() == EQ.spot
    g = pos.greeks()
    assert g.delta == 1.0
    assert g.gamma == 0.0
    assert pos.value() == pytest.approx(50 * EQ.spot)
    assert pos.position_greeks().delta == pytest.approx(50.0)


def test_position_scales_price_and_greeks_by_quantity_and_multiplier():
    spec = OptionSpec(option_type="call", strike=100.0, expiry=1.0, style="european", underlying_type="equity")
    pos = Position(instrument=spec, quantity=-3, market=EQ, multiplier=100)
    unit_price = spec.price(EQ)
    unit_greeks = spec.greeks(EQ)
    assert pos.value() == pytest.approx(-3 * 100 * unit_price)
    assert pos.position_greeks().delta == pytest.approx(-3 * 100 * unit_greeks.delta)


def test_portfolio_value_and_greeks_are_sum_of_positions():
    spec1 = OptionSpec(option_type="call", strike=100.0, expiry=1.0)
    spec2 = OptionSpec(option_type="put", strike=100.0, expiry=1.0)
    p1 = Position(instrument=spec1, quantity=1, market=EQ, multiplier=100)
    p2 = Position(instrument=spec2, quantity=2, market=EQ, multiplier=100)
    portfolio = Portfolio(positions=[p1, p2])

    assert portfolio.value() == pytest.approx(p1.value() + p2.value())
    combined_greeks = portfolio.greeks()
    expected_delta = p1.position_greeks().delta + p2.position_greeks().delta
    assert combined_greeks.delta == pytest.approx(expected_delta)


def test_empty_portfolio_has_zero_value_and_greeks():
    portfolio = Portfolio()
    assert portfolio.value() == 0.0
    assert portfolio.greeks().delta == 0.0
    assert len(portfolio) == 0


def test_portfolio_shocked_applies_relative_spot_and_absolute_vol_shock():
    spec = OptionSpec(option_type="call", strike=100.0, expiry=1.0)
    pos = Position(instrument=spec, quantity=1, market=EQ, multiplier=100)
    portfolio = Portfolio(positions=[pos])

    shocked = portfolio.shocked(spot_shock_pct=0.1, vol_shock=0.05)
    assert shocked.positions[0].market.spot == pytest.approx(EQ.spot * 1.1)
    assert shocked.positions[0].market.vol == pytest.approx(EQ.vol + 0.05)
    # original portfolio must be untouched (shocked returns a new Portfolio)
    assert portfolio.positions[0].market.spot == pytest.approx(EQ.spot)


def test_demo_portfolio_builds_and_prices_cleanly():
    portfolio = build_demo_portfolio()
    assert len(portfolio) > 0
    value = portfolio.value()
    greeks = portfolio.greeks()
    assert np.isfinite(value)
    for v in greeks.as_dict().values():
        assert np.isfinite(v)

    frame = portfolio.to_frame()
    assert len(frame) == len(portfolio)
    assert {"label", "value", "delta", "gamma", "vega"}.issubset(frame.columns)
