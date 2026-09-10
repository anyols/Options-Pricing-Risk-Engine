"""Chart library: static (matplotlib/seaborn) charts for the README/report/
notebook, and interactive (Plotly) charts for the Streamlit dashboard.

Every function returns a figure object (never calls `.show()`/renders
itself) so the caller decides whether to save, embed, or display it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import CenteredNorm
from matplotlib.figure import Figure
from scipy.signal import savgol_filter

from optrisk.greeks.types import Greeks
from optrisk.instruments.option import Stock
from optrisk.instruments.portfolio import Portfolio
from optrisk.risk.hedging import HedgeSimulationResult
from optrisk.risk.scenarios import TAYLOR_ORDER_LABELS, TAYLOR_ORDERS, ScenarioResult
from optrisk.viz.theme import (
    AMBER,
    BLUE,
    ERROR_CMAP,
    GRAY,
    GREEN,
    NAVY,
    PALETTE,
    PNL_CMAP,
    PURPLE,
    RED,
    TEAL,
    TEXT,
    apply_matplotlib_theme,
)

apply_matplotlib_theme()

__all__ = [
    "plot_payoff_diagram",
    "plot_greek_curves",
    "plot_pnl_heatmap",
    "plot_taylor_error_heatmaps",
    "plot_taylor_slice",
    "plot_hedge_path",
    "plot_hedge_pnl_distribution",
    "plot_vol_smile",
    "plot_greeks_bar",
    "plotly_pnl_surface",
    "plotly_hedge_path",
    "plotly_vol_smile",
    "plotly_pnl_distribution",
]


# ============================================================ static (matplotlib) ===


def _position_expiry_pnl(position, spot_at_expiry: np.ndarray) -> np.ndarray:
    if isinstance(position.instrument, Stock):
        payoff = spot_at_expiry
        cost_basis = position.market.spot
    else:
        payoff = np.array([position.instrument.payoff(float(s)) for s in spot_at_expiry])
        cost_basis = position.price()
    return position.quantity * position.multiplier * (payoff - cost_basis)


def plot_payoff_diagram(portfolio: Portfolio, spot_range: Optional[np.ndarray] = None) -> Figure:
    """Classic "hockey stick" P&L-at-expiry diagram, as a function of the
    *primary* underlying's terminal spot (the first position's market).

    A 2D diagram can only vary one risk factor, so positions on a different
    underlying (a different `market` snapshot -- e.g. the CRUDE futures
    leg in the demo book) are held fixed at their own current spot rather
    than being stretched across the primary underlying's range.
    """
    reference_market = portfolio.positions[0].market
    if spot_range is None:
        spot_range = np.linspace(0.5 * reference_market.spot, 1.5 * reference_market.spot, 200)

    total = np.zeros_like(spot_range)
    other_underlyings = False
    for position in portfolio.positions:
        if position.market == reference_market:
            total += _position_expiry_pnl(position, spot_range)
        else:
            other_underlyings = True
            total += _position_expiry_pnl(position, np.full_like(spot_range, position.market.spot))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.fill_between(spot_range, total, 0, where=(total >= 0), color=GREEN, alpha=0.15, interpolate=True)
    ax.fill_between(spot_range, total, 0, where=(total < 0), color=RED, alpha=0.15, interpolate=True)
    ax.plot(spot_range, total, color=NAVY, linewidth=2.2, label="Portfolio P&L at expiry")
    subtitle = " (other underlyings held at current spot)" if other_underlyings else ""
    ax.set_xlabel("Primary underlying spot at expiry" + subtitle)
    ax.set_ylabel("P&L ($)")
    ax.set_title(f"{portfolio.name}: P&L at Expiry")
    ax.legend()
    fig.tight_layout()
    return fig


def _smooth_for_display(series: np.ndarray) -> np.ndarray:
    """Light Savitzky-Golay smoothing for charting only (never applied to
    engine output used in calculations/tests). A book containing an
    American-style position prices Greeks via bump-and-reprice on a
    binomial tree (see `OptionSpec.greeks`), which is only piecewise-smooth
    in spot -- nearby trees have slightly different node alignment relative
    to the strike, producing small high-frequency "sawtooth" noise on top
    of an otherwise smooth true curve. That is a textbook case for this
    filter, which fits a local low-order polynomial rather than just
    averaging, so it cleans up the noise without flattening real curvature.
    """
    window = min(31, len(series) - (1 - len(series) % 2))
    if window < 5:
        return series
    return savgol_filter(series, window_length=window, polyorder=3)


def plot_greek_curves(
    portfolio: Portfolio,
    spot_shocks_pct: Optional[np.ndarray] = None,
    greek_names: Sequence[str] = ("delta", "gamma", "vega", "theta"),
    **model_kwargs,
) -> Figure:
    """Small multiples of portfolio Greeks as the underlying moves."""
    if spot_shocks_pct is None:
        spot_shocks_pct = np.linspace(-0.3, 0.3, 121)
    ref_spot = portfolio.positions[0].market.spot
    spots = ref_spot * (1 + spot_shocks_pct)

    rows = [portfolio.shocked(spot_shock_pct=float(s)).greeks(**model_kwargs).as_dict() for s in spot_shocks_pct]
    frame = pd.DataFrame(rows, index=spots)

    n = len(greek_names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name, color in zip(axes, greek_names, PALETTE):
        ax.plot(frame.index, _smooth_for_display(frame[name].to_numpy()), color=color, linewidth=2.2)
        ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.axvline(ref_spot, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(name.capitalize())
        ax.set_xlabel("Spot")
    axes[0].set_ylabel("Portfolio Greek")
    fig.suptitle(f"{portfolio.name}: Greeks vs Spot", fontweight="bold", color=NAVY)
    fig.tight_layout()
    return fig


def plot_pnl_heatmap(result: ScenarioResult, order: Optional[str] = None) -> Figure:
    """Heatmap of P&L over the (spot shock, vol shock) grid: full reprice, or a Taylor order."""
    data = result.full_pnl if order is None else result.taylor_pnl[order]
    title = "Full Reprice P&L" if order is None else f"Taylor Approx P&L ({TAYLOR_ORDER_LABELS[order]})"

    fig, ax = plt.subplots(figsize=(8, 6))
    norm = CenteredNorm(vcenter=0)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=PNL_CMAP,
        norm=norm,
        extent=[result.spot_shocks[0] * 100, result.spot_shocks[-1] * 100, result.vol_shocks[0] * 100, result.vol_shocks[-1] * 100],
    )
    fig.colorbar(im, ax=ax, label="P&L ($)")
    ax.set_xlabel("Spot shock (%)")
    ax.set_ylabel("Vol shock (vol pts)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_taylor_error_heatmaps(result: ScenarioResult) -> Figure:
    """Small multiples: (Taylor approx - full reprice) error, one panel per order."""
    fig, axes = plt.subplots(1, len(TAYLOR_ORDERS), figsize=(4.6 * len(TAYLOR_ORDERS), 4.2), sharey=True)
    errors = [result.error(order) for order in TAYLOR_ORDERS]
    vmax = max(np.max(np.abs(e)) for e in errors)
    extent = [result.spot_shocks[0] * 100, result.spot_shocks[-1] * 100, result.vol_shocks[0] * 100, result.vol_shocks[-1] * 100]

    im = None
    for ax, order, err in zip(axes, TAYLOR_ORDERS, errors):
        im = ax.imshow(err, aspect="auto", origin="lower", cmap=ERROR_CMAP, vmin=0, vmax=vmax, extent=extent)
        # ERROR_CMAP is sequential, so feed |error| for a clean magnitude read
        im.set_data(np.abs(err))
        ax.set_title(TAYLOR_ORDER_LABELS[order], fontsize=10.5)
        ax.set_xlabel("Spot shock (%)")
    axes[0].set_ylabel("Vol shock (vol pts)")
    fig.colorbar(im, ax=axes, label="|Approximation error| ($)", shrink=0.85)
    fig.suptitle("Where Greeks-Only Risk Misses Reality", fontweight="bold", color=NAVY)
    return fig


def plot_taylor_slice(result: ScenarioResult, vol_shock_index: Optional[int] = None) -> Figure:
    """The single most direct nonlinearity chart: full reprice vs each Taylor order, spot sliced at fixed vol shock."""
    if vol_shock_index is None:
        vol_shock_index = len(result.vol_shocks) // 2  # default to vol_shock = 0 if grid is symmetric

    spot_pct = result.spot_shocks * 100
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(spot_pct, result.full_pnl[vol_shock_index], color=NAVY, linewidth=2.8, label="Full reprice (exact)", zorder=5)
    for order, color, style in zip(TAYLOR_ORDERS, [RED, AMBER, TEAL, PURPLE], ["--", "-.", ":", "--"]):
        ax.plot(
            spot_pct,
            result.taylor_pnl[order][vol_shock_index],
            color=color,
            linewidth=1.8,
            linestyle=style,
            label=TAYLOR_ORDER_LABELS[order],
        )
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    vol_shock_pts = result.vol_shocks[vol_shock_index] * 100
    ax.set_xlabel("Spot shock (%)")
    ax.set_ylabel("Portfolio P&L ($)")
    ax.set_title(f"Full Repricing vs Taylor Approximation (vol shock = {vol_shock_pts:+.1f} pts)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_hedge_path(result: HedgeSimulationResult) -> Figure:
    """Underlying path + delta, and the resulting hedge P&L, for one simulated path."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True, height_ratios=[1.2, 1])

    ax1.plot(result.times, result.spot_path, color=NAVY, linewidth=1.8, label="Spot")
    ax1.set_ylabel("Spot", color=NAVY)
    ax1b = ax1.twinx()
    ax1b.plot(result.times, result.delta_path, color=TEAL, linewidth=1.6, linestyle="--", label="Option delta")
    ax1b.set_ylabel("Delta", color=TEAL)
    ax1b.grid(False)
    ax1.set_title("Simulated Underlying Path and Hedge Delta")

    ax2.axhline(0, color=GRAY, linewidth=0.8)
    ax2.plot(result.times, result.portfolio_value_path, color=AMBER, linewidth=2.2)
    ax2.fill_between(result.times, result.portfolio_value_path, 0, color=AMBER, alpha=0.15)
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Mark-to-model hedge P&L ($)")
    ax2.set_title(f"Hedging P&L (final = {result.final_pnl:+.3f})")

    fig.tight_layout()
    return fig


def plot_hedge_pnl_distribution(frequency_frame: pd.DataFrame) -> Figure:
    """Final hedging P&L distribution, faceted by rebalancing frequency."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = frequency_frame.groupby("frequency")["n_rebalances"].mean().sort_values().index.tolist()
    for freq, color in zip(order, PALETTE):
        subset = frequency_frame.loc[frequency_frame["frequency"] == freq, "final_pnl"]
        sns.kdeplot(subset, ax=ax, color=color, linewidth=2.2, fill=True, alpha=0.12, label=f"{freq} (std={subset.std():.3f})")
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("Final hedging P&L ($)")
    ax.set_ylabel("Density")
    ax.set_title("Hedging P&L Distribution by Rebalancing Frequency")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_vol_smile(strikes: np.ndarray, iv_series: Dict[str, np.ndarray], spot: Optional[float] = None) -> Figure:
    """One or more implied-vol curves vs strike (e.g. Heston-implied vs a flat BSM assumption)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (label, ivs), color in zip(iv_series.items(), PALETTE):
        ax.plot(strikes, np.asarray(ivs) * 100, marker="o", markersize=4, color=color, linewidth=2.0, label=label)
    if spot is not None:
        ax.axvline(spot, color=GRAY, linewidth=0.8, linestyle="--", label="Spot (ATM)")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Implied Volatility Smile")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_greeks_bar(greeks: Greeks, greek_names: Sequence[str] = ("delta", "gamma", "vega", "theta", "rho")) -> Figure:
    """Horizontal bar chart summarizing a portfolio's net Greeks."""
    values = [getattr(greeks, name) for name in greek_names]
    colors = [GREEN if v >= 0 else RED for v in values]

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(greek_names) + 1.5))
    bars = ax.barh([n.capitalize() for n in greek_names], values, color=colors)
    ax.axvline(0, color=GRAY, linewidth=0.8)

    # Greeks routinely span several orders of magnitude (e.g. Gamma ~ 10s,
    # Vega ~ 10,000s), so a fixed "label just past the bar's tip" placement
    # either collides with the y-axis category labels for the longest bars,
    # or is invisible against a near-zero-width bar for the shortest ones.
    # Long bars get an inward white label; short bars get an outward dark one.
    span = max(abs(v) for v in values) or 1.0
    pad = span * 0.02
    for bar, value in zip(bars, values):
        y = bar.get_y() + bar.get_height() / 2
        if abs(value) > 0.15 * span:
            x, ha, color = (value - pad if value >= 0 else value + pad), ("right" if value >= 0 else "left"), "white"
        else:
            x, ha, color = (value + pad if value >= 0 else value - pad), ("left" if value >= 0 else "right"), TEXT
        ax.text(x, y, f"{value:,.2f}", va="center", ha=ha, fontsize=9.5, color=color, fontweight="bold")

    ax.set_xlabel("Portfolio Greek (dollar terms)")
    ax.set_title("Net Portfolio Greeks")
    ax.margins(x=0.18, y=0.15)
    fig.tight_layout()
    return fig


# ============================================================ interactive (plotly) ===


def plotly_pnl_surface(result: ScenarioResult, order: Optional[str] = None):
    """Interactive 3D P&L surface over (spot shock, vol shock): full reprice, or a Taylor order."""
    import plotly.graph_objects as go

    from optrisk.viz.theme import plotly_template

    data = result.full_pnl if order is None else result.taylor_pnl[order]
    title = "Full Reprice P&L Surface" if order is None else f"Taylor Approx P&L Surface ({TAYLOR_ORDER_LABELS[order]})"

    fig = go.Figure(
        data=go.Surface(
            x=result.spot_shocks * 100,
            y=result.vol_shocks * 100,
            z=data,
            colorscale="RdYlGn",
            cmid=0,
            colorbar=dict(title="P&L ($)"),
        )
    )
    fig.update_layout(
        template=plotly_template(),
        title=title,
        scene=dict(
            xaxis_title="Spot shock (%)",
            yaxis_title="Vol shock (pts)",
            zaxis_title="P&L ($)",
        ),
        height=560,
    )
    return fig


def plotly_hedge_path(result: HedgeSimulationResult):
    """Interactive spot path + cumulative hedge P&L, with hover detail."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from optrisk.viz.theme import AMBER as _AMBER
    from optrisk.viz.theme import NAVY as _NAVY
    from optrisk.viz.theme import plotly_template

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=result.times, y=result.spot_path, name="Spot", line=dict(color=_NAVY, width=2)), secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=result.times, y=result.portfolio_value_path, name="Hedge P&L", line=dict(color=_AMBER, width=2)),
        secondary_y=True,
    )
    fig.update_layout(template=plotly_template(), title="Delta-Hedging Simulation", height=460)
    fig.update_xaxes(title_text="Time (years)")
    fig.update_yaxes(title_text="Spot", secondary_y=False)
    fig.update_yaxes(title_text="Mark-to-model hedge P&L ($)", secondary_y=True)
    return fig


def plotly_vol_smile(strikes: np.ndarray, iv_series: Dict[str, np.ndarray], spot: Optional[float] = None):
    """Interactive implied-vol smile with one trace per series."""
    import plotly.graph_objects as go

    from optrisk.viz.theme import plotly_template

    fig = go.Figure()
    for label, ivs in iv_series.items():
        fig.add_trace(go.Scatter(x=strikes, y=np.asarray(ivs) * 100, mode="lines+markers", name=label))
    if spot is not None:
        fig.add_vline(x=spot, line_dash="dash", line_color="#8B96A5", annotation_text="Spot")
    fig.update_layout(
        template=plotly_template(),
        title="Implied Volatility Smile",
        xaxis_title="Strike",
        yaxis_title="Implied volatility (%)",
        height=440,
    )
    return fig


def plotly_pnl_distribution(frequency_frame: pd.DataFrame):
    """Interactive overlapping histogram of hedging P&L by rebalancing frequency."""
    import plotly.express as px

    from optrisk.viz.theme import PALETTE as _PALETTE
    from optrisk.viz.theme import plotly_template

    order = frequency_frame.groupby("frequency")["n_rebalances"].mean().sort_values().index.tolist()
    fig = px.histogram(
        frequency_frame,
        x="final_pnl",
        color="frequency",
        category_orders={"frequency": order},
        color_discrete_sequence=_PALETTE,
        barmode="overlay",
        opacity=0.6,
        histnorm="probability density",
        marginal="box",
    )
    fig.update_layout(
        template=plotly_template(),
        title="Hedging P&L Distribution by Rebalancing Frequency",
        xaxis_title="Final hedging P&L ($)",
        yaxis_title="Density",
        height=480,
    )
    return fig
