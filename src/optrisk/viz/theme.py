"""Shared visual identity for every chart in this project.

One palette, applied consistently to the static matplotlib/seaborn charts
(used for the README/report assets and the notebook) and the interactive
Plotly charts (used in the Streamlit dashboard), so a screenshot of either
looks like part of the same product.
"""

from __future__ import annotations

import matplotlib as mpl
import plotly.graph_objects as go
import seaborn as sns
from cycler import cycler

# -- palette -------------------------------------------------------------
NAVY = "#0B2545"
BLUE = "#1B6CA8"
TEAL = "#0F9D8C"
AMBER = "#E8A33D"
PURPLE = "#7C5CBF"
GREEN = "#2FA84F"
RED = "#D64545"
GRAY = "#8B96A5"
LIGHT_GRAY = "#E4E9F0"
BACKGROUND = "#FFFFFF"
PANEL = "#F7F9FC"
TEXT = "#1F2937"

PALETTE = [NAVY, TEAL, AMBER, PURPLE, BLUE, RED, GREEN, GRAY]

#: diverging colormap for signed P&L heatmaps (loss -> breakeven -> gain)
PNL_CMAP = "RdYlGn"
#: sequential colormap for |error| / magnitude heatmaps
ERROR_CMAP = "OrRd"

FONT_STACK = "Segoe UI, Helvetica Neue, Arial, sans-serif"

_THEME_APPLIED = False


def apply_matplotlib_theme() -> None:
    """Apply the shared theme globally. Idempotent; safe to call from every module."""
    global _THEME_APPLIED
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": LIGHT_GRAY,
            "axes.labelcolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.titlecolor": NAVY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": LIGHT_GRAY,
            "grid.linewidth": 0.7,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "legend.frameon": False,
            "savefig.facecolor": BACKGROUND,
            "savefig.dpi": 160,
            "figure.dpi": 110,
            "axes.prop_cycle": cycler(color=PALETTE),
        }
    )
    _THEME_APPLIED = True


def plotly_template() -> go.layout.Template:
    """A Plotly layout template matching the matplotlib theme."""
    return go.layout.Template(
        layout=go.Layout(
            font={"family": FONT_STACK, "color": TEXT, "size": 13},
            paper_bgcolor=BACKGROUND,
            plot_bgcolor=PANEL,
            colorway=PALETTE,
            title={"font": {"color": NAVY, "size": 18}},
            xaxis={"gridcolor": LIGHT_GRAY, "zerolinecolor": LIGHT_GRAY, "linecolor": LIGHT_GRAY},
            yaxis={"gridcolor": LIGHT_GRAY, "zerolinecolor": LIGHT_GRAY, "linecolor": LIGHT_GRAY},
            legend={"bgcolor": "rgba(0,0,0,0)"},
            margin={"l": 60, "r": 30, "t": 60, "b": 50},
        )
    )
