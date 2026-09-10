"""Positions and portfolio-level aggregation of value and Greeks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Union

import pandas as pd

from optrisk.greeks.types import Greeks
from optrisk.instruments.option import MarketEnv, OptionSpec, Stock

Instrument = Union[OptionSpec, Stock]

__all__ = ["Position", "Portfolio"]


@dataclass
class Position:
    """One line item in a book: an instrument, a quantity, and the market it's priced in.

    `quantity` is signed (negative = short) and counted in contracts/shares;
    `multiplier` converts a contract to underlying units (e.g. 100 shares
    per equity option contract).
    """

    instrument: Instrument
    quantity: float
    market: MarketEnv
    label: str = ""
    multiplier: float = 1.0

    def price(self, model: str = "auto", **model_kwargs) -> float:
        if isinstance(self.instrument, Stock):
            return self.market.spot
        return self.instrument.price(self.market, model=model, **model_kwargs)

    def greeks(self, model: str = "auto", **model_kwargs) -> Greeks:
        if isinstance(self.instrument, Stock):
            return Greeks(delta=1.0)
        return self.instrument.greeks(self.market, model=model, **model_kwargs)

    def value(self, model: str = "auto", **model_kwargs) -> float:
        return self.quantity * self.multiplier * self.price(model=model, **model_kwargs)

    def position_greeks(self, model: str = "auto", **model_kwargs) -> Greeks:
        return (self.quantity * self.multiplier) * self.greeks(model=model, **model_kwargs)

    def with_market(self, market: MarketEnv) -> "Position":
        return replace(self, market=market)


@dataclass
class Portfolio:
    """A collection of positions, possibly across multiple underlyings/models."""

    positions: List[Position] = field(default_factory=list)
    name: str = "Portfolio"

    def add(self, position: Position) -> "Portfolio":
        self.positions.append(position)
        return self

    def value(self, model: str = "auto", **model_kwargs) -> float:
        return sum(p.value(model=model, **model_kwargs) for p in self.positions)

    def greeks(self, model: str = "auto", **model_kwargs) -> Greeks:
        return sum((p.position_greeks(model=model, **model_kwargs) for p in self.positions), Greeks())

    def shocked(self, *, spot_shock_pct: float = 0.0, vol_shock: float = 0.0) -> "Portfolio":
        """A new Portfolio with every position's market shocked (relative spot, absolute vol)."""
        shocked_positions = [
            p.with_market(p.market.shocked(spot_shock_pct=spot_shock_pct, vol_shock=vol_shock))
            for p in self.positions
        ]
        return Portfolio(positions=shocked_positions, name=self.name)

    def to_frame(self, model: str = "auto", **model_kwargs) -> pd.DataFrame:
        """One row per position: identifying info, price/value, and full Greeks."""
        rows = []
        for p in self.positions:
            g = p.position_greeks(model=model, **model_kwargs)
            row = {
                "label": p.label or type(p.instrument).__name__,
                "quantity": p.quantity,
                "multiplier": p.multiplier,
                "spot": p.market.spot,
                "vol": p.market.vol,
                "price": p.price(model=model, **model_kwargs),
                "value": p.value(model=model, **model_kwargs),
            }
            row.update(g.as_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.positions)
