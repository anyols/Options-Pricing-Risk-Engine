"""Options Pricing & Risk Engine -- interactive Streamlit dashboard.

Run with: streamlit run app/dashboard.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from optrisk.instruments.option import MarketEnv, OptionSpec, Stock
from optrisk.instruments.portfolio import Portfolio, Position
from optrisk.market.sample_data import build_demo_portfolio
from optrisk.models.heston import heston_implied_vol_smile
from optrisk.risk.hedging import run_hedge_frequency_comparison, simulate_delta_hedge
from optrisk.risk.scenarios import TAYLOR_ORDER_LABELS, TAYLOR_ORDERS, default_shock_grid, run_scenario_analysis
from optrisk.viz.plots import (
    plot_greeks_bar,
    plotly_hedge_path,
    plotly_pnl_distribution,
    plotly_pnl_surface,
    plotly_vol_smile,
)
from optrisk.viz.theme import NAVY

st.set_page_config(page_title="Options Pricing & Risk Engine", page_icon="\U0001f4c8", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FAFBFC; }}
    h1, h2, h3 {{ color: {NAVY}; }}
    [data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; }}
    [data-testid="stMetricLabel"] {{ color: #5B6472; }}
    div[data-testid="stSidebarUserContent"] {{ padding-top: 1rem; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================ portfolio construction ===


def _init_state() -> None:
    if "custom_positions" not in st.session_state:
        st.session_state.custom_positions = []
    if "custom_market" not in st.session_state:
        st.session_state.custom_market = {"spot": 100.0, "rate": 0.04, "vol": 0.25, "dividend_yield": 0.01}


def build_custom_portfolio() -> Portfolio:
    market = MarketEnv(**st.session_state.custom_market)
    portfolio = Portfolio(name="Custom Book")
    for leg in st.session_state.custom_positions:
        instrument: OptionSpec | Stock
        if leg["kind"] == "Stock":
            instrument = Stock()
        else:
            instrument = OptionSpec(
                option_type=leg["option_type"],
                strike=leg["strike"],
                expiry=leg["expiry"],
                style=leg["style"],
                underlying_type="equity",
            )
        portfolio.add(
            Position(
                instrument=instrument,
                quantity=leg["quantity"],
                market=market,
                label=leg["label"],
                multiplier=leg.get("multiplier", 100),
            )
        )
    return portfolio


def sidebar_portfolio() -> Portfolio:
    st.sidebar.title("\U0001f4c8 optrisk")
    st.sidebar.caption("Options Pricing & Risk Engine")
    mode = st.sidebar.radio("Portfolio", ["Demo multi-underlying book", "Build your own"], label_visibility="collapsed")

    if mode == "Demo multi-underlying book":
        st.sidebar.info(
            "A 6-leg book across a fictitious equity (TECH) and a futures option (CRUDE), with a partial delta hedge.",
            icon="\U0001f4bc",
        )
        return build_demo_portfolio()

    _init_state()
    st.sidebar.subheader("Market")
    m = st.session_state.custom_market
    m["spot"] = st.sidebar.number_input("Spot", value=m["spot"], min_value=0.01, step=1.0)
    m["rate"] = st.sidebar.number_input("Risk-free rate", value=m["rate"], format="%.4f", step=0.005)
    m["vol"] = st.sidebar.number_input("Volatility", value=m["vol"], min_value=0.001, format="%.4f", step=0.01)
    m["dividend_yield"] = st.sidebar.number_input(
        "Dividend yield", value=m["dividend_yield"], format="%.4f", step=0.005
    )

    st.sidebar.subheader("Add a position")
    with st.sidebar.form("add_leg", clear_on_submit=True):
        kind = st.selectbox("Instrument", ["Call", "Put", "Stock"])
        col1, col2 = st.columns(2)
        strike = col1.number_input("Strike", value=float(round(m["spot"])), min_value=0.01)
        expiry = col2.number_input("Expiry (yrs)", value=0.5, min_value=0.001, step=0.25)
        style = st.selectbox("Style", ["european", "american"])
        quantity = st.number_input("Quantity (+long / -short)", value=1.0, step=1.0)
        submitted = st.form_submit_button("Add to book", width="stretch")
        if submitted:
            if kind == "Stock":
                leg = {"kind": "Stock", "quantity": quantity, "label": f"Stock x{quantity:g}", "multiplier": 1}
            else:
                opt_type = "call" if kind == "Call" else "put"
                leg = {
                    "kind": "Option",
                    "option_type": opt_type,
                    "strike": strike,
                    "expiry": expiry,
                    "style": style,
                    "quantity": quantity,
                    "multiplier": 100,
                    "label": f"{kind} K={strike:g} T={expiry:g}y",
                }
            st.session_state.custom_positions.append(leg)

    if st.session_state.custom_positions:
        st.sidebar.subheader("Current book")
        for i, leg in enumerate(st.session_state.custom_positions):
            c1, c2 = st.sidebar.columns([4, 1])
            c1.write(f"{leg['quantity']:+g}x {leg['label']}")
            if c2.button("✕", key=f"remove_{i}"):
                st.session_state.custom_positions.pop(i)
                st.rerun()
        if st.sidebar.button("Clear book", width="stretch"):
            st.session_state.custom_positions = []
            st.rerun()

    if not st.session_state.custom_positions:
        st.sidebar.warning("Add at least one position to explore the book.")
        return Portfolio(name="Custom Book")
    return build_custom_portfolio()


portfolio = sidebar_portfolio()

st.title("Options Pricing & Risk Engine")
st.caption(
    "Portfolio pricing, Greeks, scenario analysis and delta-hedging on "
    "Black-Scholes / Black-76 / binomial / Monte Carlo / Heston"
)

if len(portfolio) == 0:
    st.stop()

tab_summary, tab_greeks, tab_scenario, tab_hedge, tab_smile = st.tabs(
    [
        "\U0001f4cb Portfolio Summary",
        "\U0001f4d0 Greeks Explorer",
        "\U0001f30b Scenario Analysis",
        "⚖️ Delta-Hedging Simulator",
        "\U0001f30a Vol Smile (Heston)",
    ]
)


# ============================================================ tab: portfolio summary ===

with tab_summary:
    value = portfolio.value()
    greeks = portfolio.greeks()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio Value", f"${value:,.0f}")
    c2.metric("Net Delta ($)", f"{greeks.delta:,.0f}")
    c3.metric("Net Gamma ($)", f"{greeks.gamma:,.1f}")
    c4.metric("Net Vega ($)", f"{greeks.vega:,.0f}")
    c5.metric("Net Theta ($/yr)", f"{greeks.theta:,.0f}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Positions")
        frame = portfolio.to_frame()
        display_cols = [
            "label",
            "quantity",
            "multiplier",
            "spot",
            "vol",
            "price",
            "value",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
        ]
        st.dataframe(
            frame[display_cols].style.format({c: "{:,.2f}" for c in display_cols if c not in ("label", "multiplier")}),
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("Net Greeks")
        st.pyplot(plot_greeks_bar(greeks), width="stretch")


# ============================================================ tab: greeks explorer ===

with tab_greeks:
    st.subheader("Portfolio Greeks vs Spot")
    ref_spot = portfolio.positions[0].market.spot
    range_pct = st.slider("Spot range (%)", 5, 60, 30, key="greeks_range") / 100
    n_points = st.slider("Resolution", 21, 161, 81, step=20, key="greeks_res")

    spot_shocks_pct = np.linspace(-range_pct, range_pct, n_points)
    spots = ref_spot * (1 + spot_shocks_pct)
    rows = [portfolio.shocked(spot_shock_pct=float(s)).greeks().as_dict() for s in spot_shocks_pct]
    greek_frame = pd.DataFrame(rows, index=spots)

    greek_choice = st.multiselect(
        "Greeks to plot", list(greek_frame.columns), default=["delta", "gamma", "vega", "theta"]
    )
    if greek_choice:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from optrisk.viz.theme import PALETTE, plotly_template

        fig = make_subplots(rows=1, cols=len(greek_choice), subplot_titles=[g.capitalize() for g in greek_choice])
        # PALETTE is deliberately longer than any greek_choice selection; strict=False caps at the shorter one.
        for i, (g, color) in enumerate(zip(greek_choice, PALETTE, strict=False), start=1):
            fig.add_trace(
                go.Scatter(x=greek_frame.index, y=greek_frame[g], name=g, line={"color": color, "width": 2.5}),
                row=1,
                col=i,
            )
            fig.add_vline(x=ref_spot, line_dash="dash", line_color="#8B96A5", row=1, col=i)
        fig.update_layout(template=plotly_template(), showlegend=False, height=380, title="Portfolio Greeks vs Spot")
        st.plotly_chart(fig, width="stretch")


# ============================================================ tab: scenario analysis ===

with tab_scenario:
    st.subheader("Full Repricing vs Taylor-Series Greek Approximation")
    st.caption(
        "The gap between the exact repriced P&L and each Taylor order is the direct measure of nonlinear "
        "portfolio risk."
    )

    c1, c2, c3, c4 = st.columns(4)
    spot_range_pct = c1.slider("Spot shock range (%)", 5, 50, 25) / 100
    vol_range_pts = c2.slider("Vol shock range (pts)", 2, 30, 15) / 100
    n_spot = c3.slider("Spot grid points", 11, 61, 31, step=10)
    n_vol = c4.slider("Vol grid points", 11, 41, 21, step=10)

    spot_shocks, vol_shocks = default_shock_grid(spot_range_pct, n_spot, vol_range_pts, n_vol)
    with st.spinner("Repricing the full grid..."):
        result = run_scenario_analysis(portfolio, spot_shocks, vol_shocks)

    surf_col, slice_col = st.columns([3, 2])
    with surf_col:
        surface_choice = st.selectbox("Surface", ["Full reprice"] + [TAYLOR_ORDER_LABELS[o] for o in TAYLOR_ORDERS])
        order = (
            None
            if surface_choice == "Full reprice"
            else TAYLOR_ORDERS[[TAYLOR_ORDER_LABELS[o] for o in TAYLOR_ORDERS].index(surface_choice)]
        )
        st.plotly_chart(plotly_pnl_surface(result, order=order), width="stretch")

    with slice_col:
        vol_idx = st.slider("Slice at vol shock", 0, len(vol_shocks) - 1, len(vol_shocks) // 2, format="")
        st.caption(f"Vol shock = {vol_shocks[vol_idx] * 100:+.1f} pts")
        import plotly.graph_objects as go

        from optrisk.viz.theme import AMBER as _A
        from optrisk.viz.theme import PURPLE as _P
        from optrisk.viz.theme import RED as _R
        from optrisk.viz.theme import TEAL as _T
        from optrisk.viz.theme import plotly_template

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=spot_shocks * 100, y=result.full_pnl[vol_idx], name="Full reprice", line={"color": NAVY, "width": 3}
            )
        )
        for o, color, dash in zip(TAYLOR_ORDERS, [_R, _A, _T, _P], ["dash", "dashdot", "dot", "dash"], strict=True):
            fig.add_trace(
                go.Scatter(
                    x=spot_shocks * 100,
                    y=result.taylor_pnl[o][vol_idx],
                    name=TAYLOR_ORDER_LABELS[o],
                    line={"color": color, "width": 1.8, "dash": dash},
                )
            )
        fig.update_layout(
            template=plotly_template(),
            height=420,
            xaxis_title="Spot shock (%)",
            yaxis_title="P&L ($)",
            legend={"orientation": "h", "y": -0.25},
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Approximation Error by Taylor Order")
    error_cols = st.columns(len(TAYLOR_ORDERS))
    for col, order_name in zip(error_cols, TAYLOR_ORDERS, strict=True):
        col.metric(
            TAYLOR_ORDER_LABELS[order_name],
            f"${result.max_abs_error(order_name):,.0f}",
            help="Max |Taylor approx - full reprice| over the grid",
        )


# ============================================================ tab: delta hedging ===

with tab_hedge:
    st.subheader("Delta-Hedging Simulator")
    st.caption(
        "Delta-hedge an option through expiry under a *realized* vol that can differ from the *implied* "
        "vol used to hedge."
    )

    c1, c2, c3, c4 = st.columns(4)
    h_spot = c1.number_input("Spot", value=float(round(portfolio.positions[0].market.spot)), min_value=0.01)
    h_strike = c1.number_input("Strike", value=float(round(portfolio.positions[0].market.spot)), min_value=0.01)
    h_expiry = c2.number_input("Expiry (yrs)", value=0.5, min_value=0.02, step=0.25)
    h_rate = c2.number_input("Rate", value=0.03, format="%.4f")
    h_iv = c3.number_input("Implied (hedging) vol", value=0.22, min_value=0.01, format="%.3f")
    h_rv = c3.number_input("Realized vol", value=0.34, min_value=0.01, format="%.3f")
    h_type = c4.selectbox("Option type", ["call", "put"])
    h_qty = c4.selectbox("Position", ["Long (+1)", "Short (-1)"])
    qty_val = 1.0 if h_qty.startswith("Long") else -1.0

    run_col, freq_col = st.columns([1, 3])
    n_paths = freq_col.slider("Monte Carlo paths (for the P&L distribution)", 100, 3000, 600, step=100)

    seed = 7
    single_path = simulate_delta_hedge(
        spot0=h_spot,
        strike=h_strike,
        rate=h_rate,
        dividend_yield=0.0,
        implied_vol=h_iv,
        realized_vol=h_rv,
        expiry=h_expiry,
        option_type=h_type,
        option_quantity=qty_val,
        n_steps=252,
        seed=seed,
    )
    st.plotly_chart(plotly_hedge_path(single_path), width="stretch")

    with st.spinner("Running hedge simulations across rebalancing frequencies..."):
        freq_frame = run_hedge_frequency_comparison(
            spot0=h_spot,
            strike=h_strike,
            rate=h_rate,
            dividend_yield=0.0,
            implied_vol=h_iv,
            realized_vol=h_rv,
            expiry=h_expiry,
            option_type=h_type,
            option_quantity=qty_val,
            frequencies={"Daily": 252, "Weekly": 36, "Monthly": 12},
            n_paths=n_paths,
            seed=seed,
        )
    stat_col, chart_col = st.columns([1, 2])
    with stat_col:
        st.write("**Final P&L by frequency**")
        st.dataframe(freq_frame.groupby("frequency")["final_pnl"].agg(mean="mean", std="std").round(3), width="stretch")
        gamma_sign = "positive (long gamma)" if qty_val > 0 else "negative (short gamma)"
        vol_gap = "exceeds" if h_rv > h_iv else "falls short of"
        st.info(
            f"Position gamma is **{gamma_sign}**, and realized vol **{vol_gap}** implied vol -- theory "
            "predicts the sign of the average hedging P&L above.",
            icon="\U0001f4a1",
        )
    with chart_col:
        st.plotly_chart(plotly_pnl_distribution(freq_frame), width="stretch")


# ============================================================ tab: vol smile ===

with tab_smile:
    st.subheader("Heston-Implied Volatility Smile vs Flat BSM")
    st.caption("A single flat Black-Scholes volatility cannot reproduce a skew; Heston's stochastic volatility can.")

    c1, c2, c3 = st.columns(3)
    s_spot = c1.number_input("Spot", value=100.0, min_value=0.01, key="smile_spot")
    s_rate = c1.number_input("Rate", value=0.03, format="%.4f", key="smile_rate")
    s_expiry = c2.number_input("Expiry (yrs)", value=0.5, min_value=0.02, step=0.25, key="smile_expiry")
    s_v0 = c2.number_input("v0 (initial variance)", value=0.045, min_value=0.001, format="%.4f")
    s_kappa = c3.number_input("kappa (mean reversion)", value=1.8, min_value=0.01)
    s_theta = c1.number_input("theta (long-run variance)", value=0.045, min_value=0.001, format="%.4f")
    s_xi = c2.number_input("xi (vol of vol)", value=0.55, min_value=0.001)
    s_rho = c3.number_input("rho (correlation)", value=-0.75, min_value=-0.999, max_value=0.999)

    strikes = np.linspace(0.7 * s_spot, 1.3 * s_spot, 25)
    with st.spinner("Pricing across strikes..."):
        heston_ivs = heston_implied_vol_smile(
            spot=s_spot,
            rate=s_rate,
            dividend_yield=0.0,
            expiry=s_expiry,
            v0=s_v0,
            kappa=s_kappa,
            theta=s_theta,
            xi=s_xi,
            rho=s_rho,
            strikes=strikes,
            option_type="call",
        )
    flat = np.full_like(strikes, np.sqrt(s_theta))
    st.plotly_chart(
        plotly_vol_smile(strikes, {"Heston (stochastic vol)": heston_ivs, "Flat BSM assumption": flat}, spot=s_spot),
        width="stretch",
    )
