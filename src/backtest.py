"""
backtest.py
==================
Main backtest loop for the portfolio replica strategy.

Receives pre-computed portfolio weights (one row per rolling-window date)
and simulates the live trading strategy including:
  - Fixed or adaptive rebalancing
  - Transaction costs
  - VaR-triggered rebalancing (optional)
  - Full performance reporting

Designed to plug into the group's unified notebook:
  - Input  : weights DataFrame from the regression module
  - Output : returns Series + full metrics summary

Usage example
-------------
    from backtest_engine import run_backtest
    from metrics import performance_summary

    results = run_backtest(
        weights=my_elastic_net_weights,   # DataFrame: dates × assets
        factor_returns=futures_returns,   # DataFrame: dates × assets
        target_returns=hfrx_returns,      # Series: dates
        rebalance_mode="adaptive",
        transaction_costs=5e-4,
    )
    print(performance_summary(results["replica_returns"], target_returns))
"""

import numpy as np
import pandas as pd

from . import bt_metrics
from .bt_metrics import (
    performance_summary,
    rolling_tracking_error,
    rolling_information_ratio,
    rolling_sharpe,
    drawdown_series,
    annualised_return,
    annualised_volatility,
    max_drawdown,
    tracking_error,
    information_ratio,
)
from . import rebalancing
from .rebalancing import (
    fixed_rebalance_schedule,
    adaptive_rebalance_schedule,
    apply_transaction_threshold,
    compute_turnover,
    classify_vol_regime,
)


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def run_backtest(
    weights: pd.DataFrame,
    factor_returns: pd.DataFrame,
    target_returns: pd.Series,
    rebalance_mode: str = "adaptive",
    transaction_costs: float = 5e-4,
    fixed_rebalance_freq: int = 4,
    vol_window: int = 4,
    low_vol_freq: int = 8,
    high_vol_freq: int = 1,
    vol_low_pct: float = 33.0,
    vol_high_pct: float = 67.0,
    turnover_threshold: float = 0.02,
    verbose: bool = True,
) -> dict:
    """
    Run the full replica backtest.

    Parameters
    ----------
    weights : pd.DataFrame
        Estimated portfolio weights. Index = rolling window end dates,
        columns = asset names matching factor_returns columns.
        One row per rolling window (e.g. weekly). These are the "clean"
        weights already VaR-adjusted by the VaR module.
    factor_returns : pd.DataFrame
        Weekly returns for all futures/factors. Index = trading dates,
        columns = asset names.
    target_returns : pd.Series
        Weekly returns of the target index (e.g. HFRX).
    rebalance_mode : str
        "fixed"    → rebalance every `fixed_rebalance_freq` periods
        "adaptive" → rebalance frequency driven by realised volatility
    transaction_costs : float
        One-way transaction cost rate (e.g. 5e-4 = 5 bps).
    fixed_rebalance_freq : int
        Periods between rebalances for "fixed" and "var" modes.
    vol_window : int
        Rolling window for volatility estimation in "adaptive" mode.
    low_vol_freq : int
        Rebalance frequency in calm regimes (adaptive mode).
    high_vol_freq : int
        Rebalance frequency in turbulent regimes (adaptive mode).
    vol_low_pct : float
        Percentile threshold for low-vol regime (adaptive mode).
    vol_high_pct : float
        Percentile threshold for high-vol regime (adaptive mode).
    turnover_threshold : float
        Skip rebalance if total weight change (L1) is below this level.
    verbose : bool
        Print summary stats.

    Returns
    -------
    dict with keys:
        "replica_returns"     : pd.Series  – period-by-period net returns
        "replica_cumulative"  : pd.Series  – cumulative wealth index
        "target_cumulative"   : pd.Series  – target wealth index (aligned)
        "active_weights"      : pd.DataFrame – weights held each period
        "rebalance_dates"     : pd.DatetimeIndex
        "turnover_series"     : pd.Series  – one-way turnover per period
        "cost_series"         : pd.Series  – transaction costs per period
        "vol_regime"          : pd.Series  – regime label per date (adaptive)
        "metrics"             : pd.DataFrame – performance_summary table
        "rolling_te"          : pd.Series  – rolling tracking error
        "rolling_ir"          : pd.Series  – rolling information ratio
    """
    
    # edit 

    '''weights, factor_returns =_align_portfolios_and_returns(weights, factor_returns)
    common_assets = weights.columns
    if common_assets.empty:
        raise ValueError("weights and factor_returns share no common assets.")
    
    trade_dates = factor_returns.index.sort_values()
    target_aligned = target_returns[trade_dates]'''

    # ── 1. Align dates ──────────────────────────────────────────────────────
    common_assets = weights.columns.intersection(factor_returns.columns)
    if common_assets.empty:
        raise ValueError("weights and factor_returns share no common assets.")

    factor_returns = factor_returns[common_assets]
    weights = weights[common_assets]

    # We only run the backtest on dates where we have factor returns
    trade_dates = factor_returns.index.sort_values()

    # Align target
    target_aligned = target_returns.reindex(trade_dates)

    # ── 2. Build rebalance schedule ─────────────────────────────────────────
    if rebalance_mode == "fixed":
        rebalance_dates = fixed_rebalance_schedule(trade_dates, frequency=fixed_rebalance_freq)

    elif rebalance_mode == "adaptive":
        rebalance_dates = adaptive_rebalance_schedule(
            returns=target_aligned.dropna(),
            vol_window=vol_window,
            low_vol_freq=low_vol_freq,
            high_vol_freq=high_vol_freq,
            vol_low_pct=vol_low_pct,
            vol_high_pct=vol_high_pct,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown rebalance_mode: {rebalance_mode!r}. Choose 'fixed' or 'adaptive'.")

    if verbose:
        print(f"\nRebalance mode    : {rebalance_mode}")
        print(f"Total trade dates : {len(trade_dates)}")
        print(f"Rebalance events  : {len(rebalance_dates)}")
        print(f"Avg holding period: {len(trade_dates) / max(1, len(rebalance_dates)):.1f} periods")

    # ── 3. Main backtest loop ───────────────────────────────────────────────
    period_returns = []
    period_dates = []
    period_costs = []
    period_turnover = []
    active_weights_rows = []

    current_weights = pd.Series(0.0, index=common_assets)

    for date in trade_dates:
        # Check if we should rebalance today
        if date in rebalance_dates:
            # Get the latest available weights (look up the most recent weight row ≤ date)
            available = weights.loc[weights.index <= date]
            if available.empty:
                new_weights = current_weights.copy()
            else:
                new_weights = available.iloc[-1]

            # Apply turnover threshold: skip if change is tiny
            weights_to_use, did_rebalance = apply_transaction_threshold(
                new_weights=new_weights,
                old_weights=current_weights,
                threshold=turnover_threshold,
            )

            if did_rebalance:
                to = compute_turnover(weights_to_use, current_weights)
                cost = to * transaction_costs
                current_weights = weights_to_use.copy()
            else:
                to = 0.0
                cost = 0.0
        else:
            to = 0.0
            cost = 0.0

        # Compute period return
        ret_today = factor_returns.loc[date, common_assets]
        gross_ret = float((current_weights * ret_today).sum())
        net_ret = gross_ret - cost

        period_returns.append(net_ret)
        period_dates.append(date)
        period_costs.append(cost)
        period_turnover.append(to)
        active_weights_rows.append(current_weights.copy())

    # ── 4. Assemble results ─────────────────────────────────────────────────
    replica_returns = pd.Series(period_returns, index=pd.DatetimeIndex(period_dates), name="replica")
    cost_series = pd.Series(period_costs, index=pd.DatetimeIndex(period_dates), name="transaction_costs")
    turnover_series = pd.Series(period_turnover, index=pd.DatetimeIndex(period_dates), name="turnover")
    active_weights = pd.DataFrame(active_weights_rows, index=pd.DatetimeIndex(period_dates))

    replica_cumulative = (1 + replica_returns).cumprod()
    target_cumulative = (1 + target_aligned.dropna()).cumprod()

    # ── 5. Metrics ──────────────────────────────────────────────────────────
    aligned_target = target_aligned.reindex(replica_returns.index).dropna()
    aligned_replica = replica_returns.reindex(aligned_target.index)

    metrics = performance_summary(
        replica_returns=aligned_replica,
        target_returns=aligned_target,
    )

    roll_te = rolling_tracking_error(aligned_replica, aligned_target, window=52)
    roll_ir = rolling_information_ratio(aligned_replica, aligned_target, window=52)

    # Volatility regime classification for analysis
    vol_regime = classify_vol_regime(
        target_aligned.dropna(),
        vol_window=vol_window,
        vol_low_pct=vol_low_pct,
        vol_high_pct=vol_high_pct,
    )

    if verbose:
        print("\n── Performance Summary ──────────────────────────────────")
        print(metrics.to_string())
        print(f"\nTotal transaction costs (cumulative): {cost_series.sum()*100:.4f}%")
        print(f"Avg one-way turnover per period     : {turnover_series[turnover_series > 0].mean()*100:.2f}%")
        print(f"Number of actual rebalances         : {(turnover_series > 0).sum()}")

    return {
        "replica_returns": replica_returns,
        "replica_cumulative": replica_cumulative,
        "target_cumulative": target_cumulative,
        "active_weights": active_weights,
        "rebalance_dates": rebalance_dates,
        "turnover_series": turnover_series,
        "cost_series": cost_series,
        "vol_regime": vol_regime,
        "metrics": metrics,
        "rolling_te": roll_te,
        "rolling_ir": roll_ir,
    }


def run_backtest_edit(
    weights: pd.DataFrame,
    factor_returns: pd.DataFrame,
    target_returns: pd.Series,
    rebalance_mode: str = "adaptive",
    transaction_costs: float = 5e-4,
    fixed_rebalance_freq: int = 4,
    vol_window: int = 4,
    low_vol_freq: int = 8,
    high_vol_freq: int = 1,
    vol_low_pct: float = 33.0,
    vol_high_pct: float = 67.0,
    turnover_threshold: float = 0.02,
    verbose: bool = True,
    garch_vol : pd.Series = None,
) -> dict:
    """
    Run the full replica backtest.

    Parameters
    ----------
    weights : pd.DataFrame
        Estimated portfolio weights. Index = rolling window end dates,
        columns = asset names matching factor_returns columns.
        One row per rolling window (e.g. weekly). These are the "clean"
        weights already VaR-adjusted by the VaR module.
    factor_returns : pd.DataFrame
        Weekly returns for all futures/factors. Index = trading dates,
        columns = asset names.
    target_returns : pd.Series
        Weekly returns of the target index (e.g. HFRX).
    rebalance_mode : str
        "fixed"    → rebalance every `fixed_rebalance_freq` periods
        "adaptive" → rebalance frequency driven by realised volatility
    transaction_costs : float
        One-way transaction cost rate (e.g. 5e-4 = 5 bps).
    fixed_rebalance_freq : int
        Periods between rebalances for "fixed" and "var" modes.
    vol_window : int
        Rolling window for volatility estimation in "adaptive" mode.
    low_vol_freq : int
        Rebalance frequency in calm regimes (adaptive mode).
    high_vol_freq : int
        Rebalance frequency in turbulent regimes (adaptive mode).
    vol_low_pct : float
        Percentile threshold for low-vol regime (adaptive mode).
    vol_high_pct : float
        Percentile threshold for high-vol regime (adaptive mode).
    turnover_threshold : float
        Skip rebalance if total weight change (L1) is below this level.
    verbose : bool
        Print summary stats.

    Returns
    -------
    dict with keys:
        "replica_returns"     : pd.Series  – period-by-period net returns
        "replica_cumulative"  : pd.Series  – cumulative wealth index
        "target_cumulative"   : pd.Series  – target wealth index (aligned)
        "active_weights"      : pd.DataFrame – weights held each period
        "rebalance_dates"     : pd.DatetimeIndex
        "turnover_series"     : pd.Series  – one-way turnover per period
        "cost_series"         : pd.Series  – transaction costs per period
        "vol_regime"          : pd.Series  – regime label per date (adaptive)
        "metrics"             : pd.DataFrame – performance_summary table
        "rolling_te"          : pd.Series  – rolling tracking error
        "rolling_ir"          : pd.Series  – rolling information ratio
    """
    
    # edit 

    weights, factor_returns =_align_portfolios_and_returns(weights, factor_returns)
    common_assets = weights.columns
    if common_assets.empty:
        raise ValueError("weights and factor_returns share no common assets.")
    
    trade_dates = factor_returns.index.sort_values()
    target_aligned = target_returns[trade_dates]

    '''# ── 1. Align dates ──────────────────────────────────────────────────────
    common_assets = weights.columns.intersection(factor_returns.columns)
    if common_assets.empty:
        raise ValueError("weights and factor_returns share no common assets.")

    factor_returns = factor_returns[common_assets]
    weights = weights[common_assets]

    # We only run the backtest on dates where we have factor returns
    trade_dates = factor_returns.index.sort_values()

    # Align target
    target_aligned = target_returns.reindex(trade_dates)'''

    # ── 2. Build rebalance schedule ─────────────────────────────────────────
    if rebalance_mode == "fixed":
        rebalance_dates = fixed_rebalance_schedule(trade_dates, frequency=fixed_rebalance_freq)

    elif rebalance_mode == "adaptive":
        rebalance_dates = adaptive_rebalance_schedule(
            returns=target_aligned.dropna(),
            vol_window=vol_window,
            low_vol_freq=low_vol_freq,
            high_vol_freq=high_vol_freq,
            vol_low_pct=vol_low_pct,
            vol_high_pct=vol_high_pct,
            verbose=verbose,
            garch_vol=garch_vol
        )

    else:
        raise ValueError(f"Unknown rebalance_mode: {rebalance_mode!r}. Choose 'fixed' or 'adaptive'.")

    if verbose:
        print(f"\nRebalance mode    : {rebalance_mode}")
        print(f"Total trade dates : {len(trade_dates)}")
        print(f"Rebalance events  : {len(rebalance_dates)}")
        print(f"Avg holding period: {len(trade_dates) / max(1, len(rebalance_dates)):.1f} periods")

    # ── 3. Main backtest loop ───────────────────────────────────────────────
    period_returns = []
    period_dates = []
    period_costs = []
    period_turnover = []
    active_weights_rows = []

    current_weights = pd.Series(0.0, index=common_assets)

    for date in trade_dates:
        # Check if we should rebalance today
        if date in rebalance_dates:
            # Get the latest available weights (look up the most recent weight row ≤ date)
            available = weights.loc[weights.index <= date]
            if available.empty:
                new_weights = current_weights.copy()
            else:
                new_weights = available.iloc[-1]

            # Apply turnover threshold: skip if change is tiny
            weights_to_use, did_rebalance = apply_transaction_threshold(
                new_weights=new_weights,
                old_weights=current_weights,
                threshold=turnover_threshold,
            )

            if did_rebalance:
                to = compute_turnover(weights_to_use, current_weights)
                cost = to * transaction_costs
                current_weights = weights_to_use.copy()
            else:
                to = 0.0
                cost = 0.0
        else:
            to = 0.0
            cost = 0.0

        # Compute period return
        ret_today = factor_returns.loc[date, common_assets]
        gross_ret = float((current_weights * ret_today).sum())
        net_ret = gross_ret - cost

        period_returns.append(net_ret)
        period_dates.append(date)
        period_costs.append(cost)
        period_turnover.append(to)
        active_weights_rows.append(current_weights.copy())

    # ── 4. Assemble results ─────────────────────────────────────────────────
    replica_returns = pd.Series(period_returns, index=pd.DatetimeIndex(period_dates), name="replica")
    cost_series = pd.Series(period_costs, index=pd.DatetimeIndex(period_dates), name="transaction_costs")
    turnover_series = pd.Series(period_turnover, index=pd.DatetimeIndex(period_dates), name="turnover")
    active_weights = pd.DataFrame(active_weights_rows, index=pd.DatetimeIndex(period_dates))

    replica_cumulative = (1 + replica_returns).cumprod()
    target_cumulative = (1 + target_aligned.dropna()).cumprod()

    # ── 5. Metrics ──────────────────────────────────────────────────────────
    aligned_target = target_aligned.reindex(replica_returns.index).dropna()
    aligned_replica = replica_returns.reindex(aligned_target.index)

    metrics = performance_summary(
        replica_returns=aligned_replica,
        target_returns=aligned_target,
    )

    roll_te = rolling_tracking_error(aligned_replica, aligned_target, window=52)
    roll_ir = rolling_information_ratio(aligned_replica, aligned_target, window=52)

    # Volatility regime classification for analysis
    vol_regime = classify_vol_regime(
        target_aligned.dropna(),
        vol_window=vol_window,
        vol_low_pct=vol_low_pct,
        vol_high_pct=vol_high_pct,
    )

    if verbose:
        print("\n── Performance Summary ──────────────────────────────────")
        print(metrics.to_string())
        print(f"\nTotal transaction costs (cumulative): {cost_series.sum()*100:.4f}%")
        print(f"Avg one-way turnover per period     : {turnover_series[turnover_series > 0].mean()*100:.2f}%")
        print(f"Number of actual rebalances         : {(turnover_series > 0).sum()}")

    return {
        "replica_returns": replica_returns,
        "replica_cumulative": replica_cumulative,
        "target_cumulative": target_cumulative,
        "active_weights": active_weights,
        "rebalance_dates": rebalance_dates,
        "turnover_series": turnover_series,
        "cost_series": cost_series,
        "vol_regime": vol_regime,
        "metrics": metrics,
        "rolling_te": roll_te,
        "rolling_ir": roll_ir,
    }


# ---------------------------------------------------------------------------
# Comparison helper: run multiple modes and compare
# ---------------------------------------------------------------------------

def compare_rebalance_modes(
    weights: pd.DataFrame,
    factor_returns: pd.DataFrame,
    target_returns: pd.Series,
    transaction_costs: float = 5e-4,
    fixed_freq: int = 4,
    verbose: bool = False,
    garch_vol: pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Run the backtest under multiple rebalance modes and compare metrics.

    Includes:
        - Fixed weekly
        - Fixed monthly
        - Fixed quarterly
        - Standard adaptive (realised volatility)
        - Adaptive using each supplied GARCH volatility model

    Parameters
    ----------
    garch_vol : pd.DataFrame | pd.Series | None
        If DataFrame:
            columns = different GARCH model forecasts
        If Series:
            treated as a single GARCH model

    Returns
    -------
    pd.DataFrame
        Summary comparison table.
    """

    # ------------------------------------------------------------
    # Standard benchmark modes
    # ------------------------------------------------------------
    modes = {
        "Fixed (weekly)": {
            "rebalance_mode": "fixed",
            "fixed_rebalance_freq": 1,
            "garch_vol": None,
        },
        "Fixed (monthly)": {
            "rebalance_mode": "fixed",
            "fixed_rebalance_freq": 4,
            "garch_vol": None,
        },
        "Fixed (quarterly)": {
            "rebalance_mode": "fixed",
            "fixed_rebalance_freq": 13,
            "garch_vol": None,
        },
        "Adaptive": {
            "rebalance_mode": "adaptive",
            "garch_vol": None,
        },
    }

    # ------------------------------------------------------------
    # Add GARCH adaptive models automatically
    # ------------------------------------------------------------
    if garch_vol is not None:

        # Convert Series → DataFrame
        if isinstance(garch_vol, pd.Series):
            garch_vol = garch_vol.to_frame(name="GARCH")

        # Add one adaptive configuration per column
        for col in garch_vol.columns:

            modes[f"Adaptive garch model {col}"] = {
                "rebalance_mode": "adaptive",
                "garch_vol": garch_vol[col],
            }

    # ------------------------------------------------------------
    # Run all backtests
    # ------------------------------------------------------------
    summary_rows = []

    for name, kwargs in modes.items():

        res = run_backtest_edit(
            weights=weights,
            factor_returns=factor_returns,
            target_returns=target_returns,
            transaction_costs=transaction_costs,
            verbose=verbose,
            rebalance_mode=kwargs["rebalance_mode"],
            fixed_rebalance_freq=kwargs.get("fixed_rebalance_freq", fixed_freq),
            garch_vol=kwargs.get("garch_vol", None),
        )

        aligned_target = target_returns.reindex(
            res["replica_returns"].index
        ).dropna()

        aligned_replica = res["replica_returns"].reindex(
            aligned_target.index
        )

        summary_rows.append({
            "Mode": name,
            "N Rebalances": int((res["turnover_series"] > 0).sum()),
            "Ann. Return (%)": round(
                annualised_return(aligned_replica) * 100, 3
            ),
            "Ann. Vol (%)": round(
                annualised_volatility(aligned_replica) * 100, 3
            ),
            "Max DD (%)": round(
                max_drawdown(aligned_replica) * 100, 3
            ),
            "Tracking Error (%)": round(
                tracking_error(aligned_replica, aligned_target) * 100, 3
            ),
            "Information Ratio": round(
                information_ratio(aligned_replica, aligned_target), 3
            ),
            "Total Costs (%)": round(
                res["cost_series"].sum() * 100, 4
            ),
        })

    return pd.DataFrame(summary_rows).set_index("Mode")


#
#--------- Helper function to align portfolios and weights
# 
def _align_portfolios_and_returns(
    portfolios: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align a portfolio schedule with the return matrix on the common investable universe.
    Fixes data leakage by shifting the end-of-period portfolio weights forward 
    before reindexing and forward-filling on the trading calendar.

    Parameters:
        portfolios (pd.DataFrame): Portfolio weights indexed by rebalance date.
        returns (pd.DataFrame): Asset returns indexed by trading date.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The aligned portfolio and return matrices.
    """
    if portfolios.empty or portfolios.shape[1] == 0:
        raise ValueError("portfolios must contain at least one asset")

    overlapping_assets = portfolios.columns.intersection(returns.columns)
    if overlapping_assets.empty:
        raise ValueError("portfolios and returns must share at least one asset")

    portfolios_shared = portfolios.loc[:, overlapping_assets]


    shifted_portfolios = portfolios_shared.shift(1)


    aligned_portfolios = (
        shifted_portfolios
        .reindex(returns.index)
        .ffill()
        .dropna(axis=0, how="all")
    )
    
    if aligned_portfolios.empty:
        raise ValueError(
            "portfolios and returns must overlap on at least one investable date"
        )

    aligned_returns = returns.loc[aligned_portfolios.index, overlapping_assets]

    return aligned_portfolios.fillna(0.0), aligned_returns