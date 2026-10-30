"""
rebalancing.py
==============
Rebalancing logic for the portfolio replica backtest.

Two modes:
  1. Fixed window  – rebalance every N periods regardless of market conditions.
  2. Adaptive window – rebalance frequency varies with realised volatility
     (or with the VaR signal passed in from the VaR module).

The core idea (from the lecture notes):
  - Short window → more frequent rebalancing → better tracking but higher costs
  - Long window  → less frequent → cheaper but potentially stale weights
  - Adaptive: shorten when vol is high, lengthen when vol is low
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fixed-window rebalancing
# ---------------------------------------------------------------------------

def fixed_rebalance_schedule(
    index: pd.DatetimeIndex,
    frequency: int = 4,
    first_rebalance: int = 0,
) -> pd.DatetimeIndex:
    """
    Generate a fixed-frequency rebalance schedule.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Full trading date index of the backtest.
    frequency : int
        Rebalance every `frequency` periods (e.g. 4 = monthly for weekly data).
    first_rebalance : int
        Index position of the first rebalance date.

    Returns
    -------
    pd.DatetimeIndex
        Dates on which to rebalance.
    """
    rebalance_positions = range(first_rebalance, len(index), frequency)
    return index[list(rebalance_positions)]


# ---------------------------------------------------------------------------
# Adaptive rebalancing
# ---------------------------------------------------------------------------

def _compute_realised_vol(
    returns: pd.Series,
    vol_window: int = 4,
) -> pd.Series:
    """Rolling realised volatility (std) over `vol_window` periods."""
    return returns.rolling(vol_window).std()


def adaptive_rebalance_schedule(
    returns: pd.Series,
    vol_window: int = 4,
    low_vol_freq: int = 8,
    high_vol_freq: int = 1,
    vol_low_pct: float = 33.0,
    vol_high_pct: float = 67.0,
    verbose: bool = False,
    garch_vol : pd.Series = None,
) -> pd.DatetimeIndex:
    """
    Generate a volatility-adaptive rebalance schedule.

    The idea:
      - Compute rolling realised volatility of the target (or replica) returns.
      - Define three regimes via percentile thresholds:
          * Low vol  (below vol_low_pct-th pct)  → rebalance every `low_vol_freq` periods
          * Mid vol  (between thresholds)         → rebalance every mid frequency
          * High vol (above vol_high_pct-th pct)  → rebalance every `high_vol_freq` periods

    Parameters
    ----------
    returns : pd.Series
        Return series used to compute volatility (typically target returns).
    vol_window : int
        Rolling window for volatility estimation (in periods). Default 4 = 1 month weekly.
    low_vol_freq : int
        Periods between rebalances in calm regimes. Default 8 = ~2 months weekly.
    high_vol_freq : int
        Periods between rebalances in turbulent regimes. Default 1 = every week.
    vol_low_pct : float
        Percentile below which vol is considered "low". Default 33rd.
    vol_high_pct : float
        Percentile above which vol is considered "high". Default 67th.
    verbose : bool
        If True, print regime distribution.

    Returns
    -------
    pd.DatetimeIndex
        Dates on which to rebalance.
    """
    rolling_vol = _compute_realised_vol(returns, vol_window).dropna()

    low_thresh = np.percentile(rolling_vol, vol_low_pct)
    high_thresh = np.percentile(rolling_vol, vol_high_pct)
    mid_freq = max(1, (low_vol_freq + high_vol_freq) // 2)

    if verbose:
        low_pct = (rolling_vol < low_thresh).mean() * 100
        mid_pct = ((rolling_vol >= low_thresh) & (rolling_vol <= high_thresh)).mean() * 100
        high_pct = (rolling_vol > high_thresh).mean() * 100
        print(f"Vol thresholds → Low: {low_thresh:.4f}, High: {high_thresh:.4f}")
        print(f"Regime distribution → Low vol: {low_pct:.1f}%, Mid: {mid_pct:.1f}%, High: {high_pct:.1f}%")
        print(f"Rebalance freq → Low: {low_vol_freq}p, Mid: {mid_freq}p, High: {high_vol_freq}p")

    rebalance_dates = []
    i = 0
    dates = rolling_vol.index.tolist()

    while i < len(dates):
        date = dates[i]
        rebalance_dates.append(date)

        if garch_vol is not None:
            vol = garch_vol.loc[date]
        else:
            vol = rolling_vol.loc[date]
        

        if vol < low_thresh:
            step = low_vol_freq
        elif vol > high_thresh:
            step = high_vol_freq
        else:
            step = mid_freq

        i += step

    return pd.DatetimeIndex(rebalance_dates)


# ---------------------------------------------------------------------------
# Weight update helpers
# ---------------------------------------------------------------------------

def should_rebalance(
    date: pd.Timestamp,
    rebalance_dates: pd.DatetimeIndex,
) -> bool:
    """Return True if `date` is in the rebalance schedule."""
    return date in rebalance_dates


def apply_transaction_threshold(
    new_weights: pd.Series,
    old_weights: pd.Series,
    threshold: float = 0.02,
) -> tuple[pd.Series, bool]:
    """
    Avoid rebalancing if the weight change is negligible (saves transaction costs).

    Parameters
    ----------
    new_weights : pd.Series
    old_weights : pd.Series
    threshold : float
        L1-norm threshold below which we skip the rebalance.
        E.g. 0.02 = skip if total weight change is less than 2%.

    Returns
    -------
    (weights_to_use, rebalanced)
        weights_to_use : old_weights if skipped, new_weights otherwise.
        rebalanced : True if we actually rebalanced.
    """
    if old_weights is None or old_weights.empty:
        return new_weights, True

    diff = (new_weights.reindex(old_weights.index).fillna(0) - old_weights).abs().sum()
    if diff < threshold:
        return old_weights, False
    return new_weights, True


def compute_turnover(
    new_weights: pd.Series,
    old_weights: pd.Series,
) -> float:
    """
    One-way portfolio turnover (sum of absolute weight changes / 2).

    Convention: one-way turnover, so you pay costs on each buy or sell
    but not both (they net out for a balanced trade).
    """
    if old_weights is None or old_weights.empty:
        return new_weights.abs().sum()

    all_assets = new_weights.index.union(old_weights.index)
    w_new = new_weights.reindex(all_assets).fillna(0)
    w_old = old_weights.reindex(all_assets).fillna(0)
    return float((w_new - w_old).abs().sum())


# ---------------------------------------------------------------------------
# Volatility regime classification (utility for analysis/plotting)
# ---------------------------------------------------------------------------

def classify_vol_regime(
    returns: pd.Series,
    vol_window: int = 4,
    vol_low_pct: float = 33.0,
    vol_high_pct: float = 67.0,
) -> pd.Series:
    """
    Return a string series classifying each date into 'low', 'mid', or 'high' vol regime.
    Useful for plotting and analysis.
    """
    rolling_vol = _compute_realised_vol(returns, vol_window)
    low_thresh = np.percentile(rolling_vol.dropna(), vol_low_pct)
    high_thresh = np.percentile(rolling_vol.dropna(), vol_high_pct)

    def _classify(v):
        if pd.isna(v):
            return np.nan
        if v < low_thresh:
            return "low"
        elif v > high_thresh:
            return "high"
        else:
            return "mid"

    return rolling_vol.map(_classify)