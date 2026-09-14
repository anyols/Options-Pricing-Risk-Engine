"""Heston (1993) stochastic-volatility pricing via the Fang-Oosterlee COS method.

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2_t,      dW1 dW2 = rho dt

Unlike BSM, Heston lets volatility itself be random and mean-reverting, and
`rho < 0` reproduces the equity volatility skew (spot down -> vol up). There
is no closed-form price, but the characteristic function of ln(S_T) is known
in closed form, and the Fang & Oosterlee (2008) COS method inverts it into a
price via a Fourier-cosine series -- typically accurate to machine precision
with only ~100-200 series terms, much faster than Monte Carlo.

The characteristic function below uses the Albrecher et al. (2007) "Little
Heston Trap" branch (`-d` instead of the original `+d`), which is
mathematically equivalent to Heston's original formula but avoids a complex
-logarithm discontinuity that the naive formula hits for long maturities or
high vol-of-vol.

Validation deliberately avoids trusting a memorized closed-form cumulant
formula for the COS truncation range (a common source of subtle bugs in
Heston implementations): `_heston_cumulants` differentiates the *already
independently-verified* characteristic function numerically instead. See
tests/test_heston.py for the two validation routes used: the analytic
degenerate-case limit (xi -> 0, v0 = theta collapses Heston to BSM), and
agreement with an independent Heston Monte Carlo simulation.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from optrisk.models.monte_carlo import MonteCarloResult

__all__ = ["heston_implied_vol_smile", "heston_mc_price", "heston_price"]


def _heston_char_func(
    u: NDArray[np.float64],
    x: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    rate: float,
    dividend_yield: float,
    tau: float,
) -> NDArray[np.complex128]:
    """Characteristic function of Y = ln(S_T / K) under Heston, at complex frequency `u`."""
    iu = 1j * u
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi**2 * (iu + u**2))
    g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    exp_dtau = np.exp(-d * tau)

    C = (rate - dividend_yield) * iu * tau + (kappa * theta / xi**2) * (
        (kappa - rho * xi * iu - d) * tau - 2.0 * np.log((1 - g * exp_dtau) / (1 - g))
    )
    D = ((kappa - rho * xi * iu - d) / xi**2) * ((1 - exp_dtau) / (1 - g * exp_dtau))
    return cast(NDArray[np.complex128], np.exp(C + D * v0 + iu * x))


def _heston_cumulants(
    x: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    rate: float,
    dividend_yield: float,
    tau: float,
) -> tuple[float, float]:
    """First two cumulants of Y = ln(S_T/K), via central differences of ln(phi(u)) at u=0.

    Used only to size the COS truncation range [a, b]; a numerical estimate
    of the mean/variance from the (independently validated) characteristic
    function is a lower-risk source of truth here than re-deriving Heston's
    closed-form cumulant expressions, which run to dozens of terms.
    """

    def psi(u: float) -> complex:
        cf = _heston_char_func(np.array([u]), x, v0, kappa, theta, xi, rho, rate, dividend_yield, tau)
        return cast(complex, np.log(cf)[0])

    h = 1e-3
    psi_plus, psi_minus, psi_mid = psi(h), psi(-h), psi(0.0)
    c1 = (-1j * (psi_plus - psi_minus) / (2 * h)).real
    c2 = (-(psi_plus - 2 * psi_mid + psi_minus) / h**2).real
    return c1, c2


def _psi_k(k: NDArray[np.float64], a: float, b: float, c: float, d: float) -> NDArray[np.float64]:
    """Integral of cos(k*pi*(y-a)/(b-a)) over [c, d]."""
    k_safe = np.where(k == 0, 1.0, k)
    omega = k_safe * np.pi / (b - a)
    raw = (np.sin(omega * (d - a)) - np.sin(omega * (c - a))) * (b - a) / (k_safe * np.pi)
    return cast(NDArray[np.float64], np.where(k == 0, d - c, raw))


def _chi_k(k: NDArray[np.float64], a: float, b: float, c: float, d: float) -> NDArray[np.float64]:
    """Integral of e^y * cos(k*pi*(y-a)/(b-a)) over [c, d]."""
    omega = k * np.pi / (b - a)
    denom = 1.0 + omega**2
    at_d = np.exp(d) * (np.cos(omega * (d - a)) + omega * np.sin(omega * (d - a)))
    at_c = np.exp(c) * (np.cos(omega * (c - a)) + omega * np.sin(omega * (c - a)))
    return cast(NDArray[np.float64], (at_d - at_c) / denom)


def _cos_payoff_coefficients(
    k: NDArray[np.float64], a: float, b: float, is_call: bool, strike: float
) -> NDArray[np.float64]:
    """Fourier-cosine coefficients of a vanilla payoff on [a, b] (Fang & Oosterlee 2008, Table 1)."""
    if is_call:
        lower = max(a, 0.0)
        values = _chi_k(k, a, b, lower, b) - _psi_k(k, a, b, lower, b) if lower < b else np.zeros_like(k, dtype=float)
    else:
        upper = min(b, 0.0)
        values = -_chi_k(k, a, b, a, upper) + _psi_k(k, a, b, a, upper) if upper > a else np.zeros_like(k, dtype=float)
    return (2.0 / (b - a)) * strike * values


def heston_price(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    expiry: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: str = "call",
    n_terms: int = 160,
    l_bound: float = 10.0,
) -> float:
    """European option price under Heston via the Fang-Oosterlee COS method.

    Parameters
    ----------
    v0, kappa, theta, xi, rho
        Initial variance, mean-reversion speed, long-run variance,
        vol-of-vol, and spot/vol correlation of the Heston SDE.
    n_terms
        Number of COS series terms (~100-200 is typically enough for
        machine-precision accuracy).
    l_bound
        Truncation-range width in standard deviations of ln(S_T/K)
        (Fang & Oosterlee recommend 8-12).
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    x = np.log(spot / strike)
    c1, c2 = _heston_cumulants(x, v0, kappa, theta, xi, rho, rate, dividend_yield, expiry)
    width = l_bound * np.sqrt(max(c2, 1e-12))
    a, b = c1 - width, c1 + width

    k = np.arange(n_terms, dtype=np.float64)
    u = k * np.pi / (b - a)
    cf = _heston_char_func(u, x, v0, kappa, theta, xi, rho, rate, dividend_yield, expiry)
    weights = np.real(cf * np.exp(-1j * u * a))
    weights[0] *= 0.5

    coeffs = _cos_payoff_coefficients(k, a, b, option_type == "call", strike)
    price = np.exp(-rate * expiry) * np.sum(weights * coeffs)
    return float(price)


def heston_mc_price(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    expiry: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int | None = None,
) -> MonteCarloResult:
    """Heston price via Euler-Maruyama simulation with the full-truncation scheme
    (Lord, Koekkoek & Van Dijk, 2010): the variance process is floored at zero
    wherever it feeds the drift/diffusion, avoiding both negative variance and
    the bias of reflecting/absorbing schemes. Used only for independent
    cross-validation of `heston_price`, since it is far slower than the COS method.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    rng = np.random.default_rng(seed)
    dt = expiry / n_steps
    sqrt_dt = np.sqrt(dt)

    s = np.full(n_paths, spot, dtype=float)
    v = np.full(n_paths, v0, dtype=float)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w2 = rho * z1 + np.sqrt(1 - rho**2) * z2

        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        s = s * np.exp((rate - dividend_yield - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1)
        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * w2

    payoff = np.maximum(s - strike, 0.0) if option_type == "call" else np.maximum(strike - s, 0.0)
    discounted = np.exp(-rate * expiry) * payoff
    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    return MonteCarloResult(price=price, std_error=std_error, n_paths=n_paths)


def heston_implied_vol_smile(
    spot: float,
    rate: float,
    dividend_yield: float,
    expiry: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    strikes: np.ndarray,
    option_type: str = "call",
) -> np.ndarray:
    """Black-Scholes implied vols recovered from Heston prices across `strikes`.

    This is the smile/skew that a single flat BSM volatility cannot
    represent -- the concrete payoff of layering a stochastic-vol model on
    top of the BSM/Black-76 core.
    """
    from optrisk.models.implied_vol import bsm_implied_vol

    ivs = np.empty(len(strikes))
    for i, strike in enumerate(strikes):
        price = heston_price(spot, strike, rate, dividend_yield, expiry, v0, kappa, theta, xi, rho, option_type)
        ivs[i] = bsm_implied_vol(price, spot, strike, rate, dividend_yield, expiry, option_type)
    return ivs
