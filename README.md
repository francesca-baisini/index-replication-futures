# Portfolio Replica Strategy

**Can you replicate a synthetic cross-asset benchmark using only liquid futures — without knowing the replication weights?**

Group project — Politecnico di Milano, A.Y. 2025–2026.  

## Problem

The target is a synthetic **Monster Index**: a fixed-weight, long-only blend of
50% HFRX Global Hedge Fund Index (HFRXGL), 25% Bloomberg Global Aggregate (LEGATRUU)
and 25% MSCI World (MXWO).

The replica can only trade **10 liquid futures** across equity (ES1, NQ1, VG1, TP1),
rates (RX1, TY1, DU1, TU2) and commodities (GC1, CO1). Replication weights are not
given and must be inferred from return time series. Correlations between the target
and the futures are time-varying and regime-dependent, so a single static fit does not work.

Data: weekly, Oct 2007 – Apr 2021. Training: 2007–2017. Out-of-sample test: Jan 2018 – Apr 2021
(includes the 2018 volatility spike and the COVID-19 crash).

## Methodology

1. **EDA & diagnostics** — distribution analysis (fat tails, negative skew), ADF/KPSS stationarity tests,
   static and 52-week rolling correlations.
2. **Volatility modelling** — rolling GARCH / GJR-GARCH with Normal and Student-t innovations;
   residual diagnostics on standardized squared residuals.
3. **Weight estimation** — OLS, Lasso, Ridge and Elastic Net in a rolling walk-forward backtest
   (52/104/156-week windows), with hyperparameters chosen at each step by TimeSeriesSplit CV
   that minimizes tracking error.
4. **State-space benchmark** — Kalman filter with weights as latent state, process noise and
   forgetting factor tuned by maximum likelihood.
5. **Risk overlay** — 1-month 99% VaR via Weighted Historical Simulation, with an 8% cap.
6. **Rebalancing** — fixed weekly / monthly / quarterly vs. a volatility-regime adaptive schedule
   (weekly in high-vol regimes, every 8 weeks in calm ones), with 5 bps one-way costs and a 2% turnover
   threshold. A GARCH-forecast trigger was also tested.
7. **Out-of-sample test** of the selected configuration on 2018–2021.

## Results

**Model selection (training, walk-forward, 156-week window)** — the four linear models are within
8 bps of tracking error of each other (2.80%–2.88%), with correlations of 0.82–0.83. Elastic Net was selected
for the lowest drawdown (5.30% vs 10.16% for the target) and the highest Sharpe (0.74). The Kalman filter reached
the lowest training TE (2.40%).

**Rebalancing (training)** — the adaptive schedule cut cumulative transaction costs by ~38% vs.
weekly rebalancing (0.147% vs 0.239%, 56 vs 104 rebalances) with the best information ratio among the
policies tested. Replacing the realized-vol trigger with a GARCH forecast did **not** help: it caused more
rebalances (69–70) and higher costs.

**Out-of-sample (Jan 2018 – Apr 2021)**

|                    | Replica | Target  |
|--------------------|---------|---------|
| Ann. return        | 3.13%   | 5.15%   |
| Ann. volatility    | 3.11%   | 6.83%   |
| Sharpe ratio       | 1.01    | 0.77    |
| Max drawdown       | 4.89%   | 13.39%  |
| Tracking error     | 5.25%   | —       |
| Correlation        | 0.68    | —       |
| Beta to target     | 0.31    | —       |
| Information ratio  | −0.40   | —       |

The replica behaves as a **low-beta tracker** rather than a full clone: gross exposure is
~0.2x and beta is 0.3–0.5. Most of the drawdown reduction and the higher Sharpe come from this lower exposure,
and the replica gives up return relative to the target (negative IR). Out-of-sample tracking error
(5.25%) is materially higher than in training (2.97%).

## Limitations

- The hedge-fund component (50% of the target) is only partially replicable with linear futures exposures.
- The replica runs at low gross exposure. A natural extension is to rescale or constrain the weights toward
  unit beta and compare tracking quality.
- Transaction costs are a flat 5 bps. Futures roll costs, margin and cash management are not modelled.
- Rebalancing policies were compared on the training period. Only the selected policy was run out-of-sample.

## Repository structure

    notebooks/Main_Group13.ipynb   full analysis with outputs
    src/backtest.py                portfolio simulation
    src/rebalancing.py             fixed and adaptive rebalancing
    src/bt_metrics.py              performance metrics
    src/var_functions.py           VaR (parametric, HS, WHS)
    src/utilities.py               statistical tests and plots
    data/                          see data/README.md

## Quick start

    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    cd notebooks && jupyter notebook Main_Group13.ipynb

The notebook must be launched from `notebooks/`, because it adds the parent folder to `sys.path`.

## Tech stack

Python · pandas · NumPy · scikit-learn · statsmodels · arch · matplotlib · seaborn
