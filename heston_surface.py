"""Small, dependency-free Heston volatility-surface calibration demo."""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Heston:
    kappa: float  # mean-reversion speed
    theta: float  # long-run variance
    sigma: float  # volatility of variance
    rho: float    # spot/variance correlation
    v0: float     # initial variance


def black_scholes_call(spot: float, strike: float, maturity: float, rate: float, vol: float) -> float:
    if maturity <= 0 or vol <= 0:
        return max(spot - strike * math.exp(-rate * maturity), 0.0)
    d1 = (math.log(spot / strike) + (rate + vol * vol / 2) * maturity) / (vol * math.sqrt(maturity))
    d2 = d1 - vol * math.sqrt(maturity)
    cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return spot * cdf(d1) - strike * math.exp(-rate * maturity) * cdf(d2)


def implied_vol(price: float, spot: float, strike: float, maturity: float, rate: float) -> float:
    """Bisection is enough here: calls are monotonic in Black-Scholes volatility."""
    lo, hi = 1e-5, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if black_scholes_call(spot, strike, maturity, rate, mid) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def heston_call(model: Heston, spot: float, strike: float, maturity: float, rate: float = 0.0, steps: int = 180) -> float:
    """Semi-analytic Heston call price using a Simpson-rule Fourier integral."""
    if min(model.kappa, model.theta, model.sigma, model.v0, maturity, spot, strike) <= 0 or abs(model.rho) >= 1:
        raise ValueError("invalid Heston parameters")

    x = math.log(spot)
    a = model.kappa * model.theta

    def cf(u: complex) -> complex:
        d = cmath.sqrt((model.rho * model.sigma * 1j * u - model.kappa) ** 2 + model.sigma**2 * (u * u + 1j * u))
        g = (model.kappa - model.rho * model.sigma * 1j * u - d) / (model.kappa - model.rho * model.sigma * 1j * u + d)
        c = rate * 1j * u * maturity + a / model.sigma**2 * ((model.kappa - model.rho * model.sigma * 1j * u - d) * maturity - 2 * cmath.log((1 - g * cmath.exp(-d * maturity)) / (1 - g)))
        q = (model.kappa - model.rho * model.sigma * 1j * u - d) / model.sigma**2
        return cmath.exp(c + q * (1 - cmath.exp(-d * maturity)) / (1 - g * cmath.exp(-d * maturity)) * model.v0 + 1j * u * x)

    phi_minus_i = cf(-1j)
    def integrand(u: float, p: int) -> float:
        numerator = cf(u - 1j) / phi_minus_i if p == 1 else cf(u)
        return (cmath.exp(-1j * u * math.log(strike)) * numerator / (1j * u)).real

    # ponytail: fixed integration grid; increase steps or use adaptive quadrature for production accuracy.
    upper, h = 120.0, 120.0 / steps
    probabilities = []
    for p in (1, 2):
        total = integrand(1e-8, p) + integrand(upper, p)
        total += sum((4 if i % 2 else 2) * integrand(i * h, p) for i in range(1, steps))
        probabilities.append(0.5 + h * total / (3 * math.pi))
    return max(0.0, spot * probabilities[0] - strike * math.exp(-rate * maturity) * probabilities[1])


def market_surface(model: Heston, spot: float, strikes: list[float], maturities: list[float], rate: float, noise: float = 0.0) -> list[tuple[float, float, float]]:
    return [(k, t, implied_vol(heston_call(model, spot, k, t, rate), spot, k, t, rate) + random.uniform(-noise, noise)) for t in maturities for k in strikes]


def weighted_rmse(model: Heston, quotes: list[tuple[float, float, float]], spot: float, rate: float) -> float:
    errors = []
    for strike, maturity, observed in quotes:
        fitted = implied_vol(heston_call(model, spot, strike, maturity, rate), spot, strike, maturity, rate)
        weight = 1 / (1 + abs(math.log(strike / spot)))  # prioritize near-the-money liquidity
        errors.append(weight * (fitted - observed) ** 2)
    return math.sqrt(sum(errors) / sum(1 / (1 + abs(math.log(k / spot))) for k, _, _ in quotes))


def calibrate(quotes: list[tuple[float, float, float]], spot: float, rate: float) -> Heston:
    """Deliberately small coordinate search; replace with a production optimizer when needed."""
    bounds = [(0.2, 5.0), (0.005, 0.25), (0.1, 1.2), (-0.95, -0.05), (0.005, 0.25)]
    values = [1.5, 0.04, 0.5, -0.6, 0.04]
    scales = [0.5, 0.02, 0.15, 0.15, 0.02]
    score = lambda xs: weighted_rmse(Heston(*xs), quotes, spot, rate)
    best = score(values)
    for _ in range(8):
        for index, (low, high) in enumerate(bounds):
            for direction in (-1, 1):
                candidate = values.copy()
                candidate[index] = min(high, max(low, candidate[index] + direction * scales[index]))
                value = score(candidate)
                if value < best:
                    values, best = candidate, value
        scales = [s / 2 for s in scales]
    return Heston(*values)


def sensitivity(model: Heston, quote: tuple[float, float, float], spot: float, rate: float) -> dict[str, float]:
    strike, maturity, _ = quote
    base = implied_vol(heston_call(model, spot, strike, maturity, rate), spot, strike, maturity, rate)
    result = {}
    for field in model.__dataclass_fields__:
        bumped = Heston(**{**model.__dict__, field: getattr(model, field) * 1.01})
        value = implied_vol(heston_call(bumped, spot, strike, maturity, rate), spot, strike, maturity, rate)
        result[field] = (value - base) / (getattr(model, field) * 0.01)
    return result


if __name__ == "__main__":
    random.seed(7)
    spot, rate = 100.0, 0.02
    true = Heston(1.8, 0.045, 0.55, -0.72, 0.035)
    quotes = market_surface(true, spot, [80, 90, 100, 110, 120], [0.25, 0.75, 1.5], rate, noise=0.006)
    fitted = calibrate(quotes[:-3], spot, rate)
    print("fitted:", fitted)
    print(f"held-out RMSE: {weighted_rmse(fitted, quotes[-3:], spot, rate) * 100:.1f} vol points")
    print("ATM 1Y sensitivity:", sensitivity(fitted, (100, 1.0, 0.0), spot, rate))
