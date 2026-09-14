"""Instrument specifications and the market data needed to price them.

`OptionSpec.price`/`.greeks` dispatch to the right pricing model
automatically ("auto"): European equity options use closed-form BSM,
European options on a future/forward use closed-form Black-76, and any
American-style option uses the CRR binomial tree. Binomial and Monte Carlo
have no closed-form Greeks, so their `.greeks()` falls back to the shared
finite-difference engine, wrapping the instrument's own `.price()` -- one
consistent numerical path for every model that lacks an analytical formula.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, cast

from optrisk.greeks.analytical import black76_full_greeks, bsm_full_greeks
from optrisk.greeks.numerical import numerical_greeks
from optrisk.greeks.types import Greeks
from optrisk.models.binomial import crr_price
from optrisk.models.black76 import black76_price
from optrisk.models.black_scholes import bsm_price
from optrisk.models.monte_carlo import mc_price

__all__ = ["MarketEnv", "Model", "OptionSpec", "Stock"]

OptionType = Literal["call", "put"]
OptionStyle = Literal["european", "american"]
UnderlyingType = Literal["equity", "future"]
Model = Literal["auto", "bsm", "black76", "binomial", "monte_carlo"]


@dataclass(frozen=True)
class MarketEnv:
    """A market data snapshot: spot (or forward/futures price), rate, vol, dividend yield."""

    spot: float
    rate: float
    vol: float
    dividend_yield: float = 0.0

    def shocked(self, *, spot_shock_pct: float = 0.0, vol_shock: float = 0.0) -> MarketEnv:
        """A new MarketEnv with a relative spot shock and an absolute vol-point shock applied."""
        return replace(self, spot=self.spot * (1.0 + spot_shock_pct), vol=max(self.vol + vol_shock, 1e-6))


@dataclass(frozen=True)
class OptionSpec:
    """A single option contract, independent of any pricing model."""

    option_type: OptionType
    strike: float
    expiry: float
    style: OptionStyle = "european"
    underlying_type: UnderlyingType = "equity"

    def payoff(self, spot_at_expiry: float) -> float:
        if self.option_type == "call":
            return max(spot_at_expiry - self.strike, 0.0)
        return max(self.strike - spot_at_expiry, 0.0)

    def _resolve_model(self, model: Model) -> Model:
        if model != "auto":
            return model
        if self.style == "american":
            return "binomial"
        return "black76" if self.underlying_type == "future" else "bsm"

    def _carry_yield(self, market: MarketEnv) -> float:
        # a futures/forward price has zero risk-neutral drift; the binomial
        # and Monte Carlo engines achieve that by setting the carry
        # (dividend_yield parameter) equal to the discount rate.
        return market.rate if self.underlying_type == "future" else market.dividend_yield

    def price(
        self,
        market: MarketEnv,
        model: Model = "auto",
        *,
        steps: int = 200,
        mc_paths: int = 50_000,
        mc_seed: int = 0,
    ) -> float:
        resolved = self._resolve_model(model)
        if resolved == "bsm":
            return cast(
                float,
                bsm_price(
                    market.spot,
                    self.strike,
                    market.rate,
                    market.dividend_yield,
                    market.vol,
                    self.expiry,
                    self.option_type,
                ),
            )
        if resolved == "black76":
            return cast(
                float, black76_price(market.spot, self.strike, market.rate, market.vol, self.expiry, self.option_type)
            )
        if resolved == "binomial":
            return crr_price(
                market.spot,
                self.strike,
                market.rate,
                self._carry_yield(market),
                market.vol,
                self.expiry,
                self.option_type,
                style=self.style,
                steps=steps,
            )
        if resolved == "monte_carlo":
            if self.style == "american":
                raise ValueError(
                    "Monte Carlo pricing of American options is not supported here "
                    "(would need Longstaff-Schwartz); use model='binomial' instead."
                )
            return mc_price(
                market.spot,
                self.strike,
                market.rate,
                self._carry_yield(market),
                market.vol,
                self.expiry,
                self.option_type,
                n_paths=mc_paths,
                seed=mc_seed,
            ).price
        raise ValueError(f"unknown model {model!r}")

    def greeks(
        self,
        market: MarketEnv,
        model: Model = "auto",
        *,
        steps: int = 200,
        mc_paths: int = 50_000,
        mc_seed: int = 0,
    ) -> Greeks:
        resolved = self._resolve_model(model)
        if resolved == "bsm":
            return bsm_full_greeks(
                market.spot, self.strike, market.rate, market.dividend_yield, market.vol, self.expiry, self.option_type
            )
        if resolved == "black76":
            return black76_full_greeks(market.spot, self.strike, market.rate, market.vol, self.expiry, self.option_type)
        if resolved in ("binomial", "monte_carlo"):

            def pricer(spot: float, vol: float, expiry: float, rate: float) -> float:
                bumped_market = MarketEnv(spot=spot, rate=rate, vol=vol, dividend_yield=market.dividend_yield)
                bumped_instrument = self if expiry == self.expiry else replace(self, expiry=expiry)
                return bumped_instrument.price(
                    bumped_market, model=resolved, steps=steps, mc_paths=mc_paths, mc_seed=mc_seed
                )

            # a binomial tree's price is only piecewise-smooth in spot (nodes
            # are spaced ~sigma*sqrt(T/steps) apart) and MC has its own
            # sampling noise, so bumps need to be coarser than the smooth
            # analytical default to avoid amplifying discretization noise
            # into the derivative estimate.
            return numerical_greeks(
                pricer,
                spot=market.spot,
                vol=market.vol,
                expiry=self.expiry,
                rate=market.rate,
                spot_bump=max(market.spot * 0.01, 1e-2),
                vol_bump=1e-2,
                expiry_bump=max(self.expiry * 0.01, 1e-3),
            )
        raise ValueError(f"unknown model {model!r}")


@dataclass(frozen=True)
class Stock:
    """The underlying itself (e.g. a delta-hedge leg): Delta=1, every other Greek=0."""
