"""Shared Greeks container used by every pricing model and the portfolio layer."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class Greeks:
    """A full first/second/third-order option Greeks set.

    Vega and Volga are quoted per 1.0 (100 vol points) change in volatility;
    Rho per 1.0 (100%) change in rate; Theta/Charm/Color per 1.0 year of
    calendar time. Divide by 100 (vega/volga/rho) or 365 (theta/charm/color)
    for the conventional "per vol point / per bp / per day" trader quoting
    convention -- kept in raw per-unit form here so the scenario/hedging
    engines can multiply directly by shocks expressed in the same units.

    Supports `+`, `-`, scalar `*`, and Python's built-in `sum()` so portfolio
    Greeks are just ``sum(qty * position_greeks for ...)``.
    """

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0
    speed: float = 0.0
    zomma: float = 0.0
    color: float = 0.0

    def __add__(self, other: Greeks) -> Greeks:
        if not isinstance(other, Greeks):
            return NotImplemented
        return Greeks(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def __radd__(self, other: object) -> Greeks:
        # makes sum([g1, g2, ...]) work, since sum() seeds the accumulator with 0
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: Greeks) -> Greeks:
        return self + (-other)

    def __neg__(self) -> Greeks:
        return self * -1.0

    def __mul__(self, scalar: float) -> Greeks:
        return Greeks(**{f.name: getattr(self, f.name) * scalar for f in fields(self)})

    __rmul__ = __mul__

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
