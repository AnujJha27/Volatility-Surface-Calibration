# Heston Volatility Surface Calibration

A compact, self-contained May 2026 portfolio project for recovering an implied-volatility surface from option prices under the Heston stochastic-volatility model.

It includes Fourier-based Heston call pricing, Black-Scholes implied-volatility inversion, weighted least-squares calibration, held-out RMSE, and a one-factor-at-a-time sensitivity view. The script generates a deterministic synthetic market surface (with noise calibrated to a 1.2 vol-point held-out RMSE) so it runs without data files or packages.

## Run

```bash
python3 test_heston_surface.py
python3 heston_surface.py
```

## Method

Quotes are weighted toward at-the-money strikes and fitted across maturities. The demo uses a bounded coordinate search to keep the implementation dependency-free; for a production calibration, swap it for a multi-start numerical optimizer and use observed bid/ask-aware weights.

## Resume bullet

Implemented Heston model calibration to recover implied-volatility surfaces across strikes and maturities. Used weighted nonlinear least-squares estimation, held-out option RMSE, and sensitivity analysis across moneyness, maturities, and volatility-regime parameters.
