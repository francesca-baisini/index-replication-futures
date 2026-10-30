"""
metrics.py
==========
Performance and risk metrics for portfolio replica backtesting.

All functions accept pd.Series of returns (not cumulative) unless stated otherwise.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_cumulative(returns: pd.Series) -> pd.Series:
    """Convert a return series to a cumulative (wealth) index starting at 1."""
    return (1 + returns).cumprod()


def _annualisation_factor(returns: pd.Series, periods_per_year: int | None = None) -> float:
    """
    Infer the annualisation factor from the return series frequency.
    Falls back to periods_per_year if provided, otherwise tries to infer from the index.
    """
    if periods_per_year is not None:
        return float(periods_per_year)

    if isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 1:
        median_days = pd.Series(returns.index).diff().dt.days.median()
        if median_days <= 2:
            return 252.0   # daily
        elif median_days <= 8:
            return 52.0    # weekly
        elif median_days <= 16:
            return 26.0    # bi-weekly
        else:
            return 12.0    # monthly

    return 52.0  # default: weekly (our dataset)


# ---------------------------------------------------------------------------
# Standalone return / risk metrics
# ---------------------------------------------------------------------------

def annualised_return(returns: pd.Series, periods_per_year: int | None = None) -> float:
    """Geometric annualised return."""
    ann = _annualisation_factor(returns, periods_per_year)
    cumret = _to_cumulative(returns).iloc[-1]
    n = len(returns)
    return float(cumret ** (ann / n) - 1)


def annualised_volatility(returns: pd.Series, periods_per_year: int | None = None) -> float:
    """Annualised standard deviation of returns."""
    ann = _annualisation_factor(returns, periods_per_year)
    return float(returns.std() * np.sqrt(ann))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Periodic return series.
    risk_free_rate : float
        Annualised risk-free rate (e.g. 0.02 = 2%). Defaults to 0.
    periods_per_year : int | None
        Override the automatic frequency detection.
    """
    ann = _annualisation_factor(returns, periods_per_year)
    rf_periodic = (1 + risk_free_rate) ** (1 / ann) - 1
    excess = returns - rf_periodic
    if excess.std() == 0:
        return np.nan
    return float(excess.mean() / excess.std() * np.sqrt(ann))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown (as a positive number, e.g. 0.30 = 30% drawdown).

    Computed on the cumulative wealth index.
    """
    cum = _to_cumulative(returns)
    rolling_max = cum.cummax()
    drawdowns = (cum - rolling_max) / rolling_max
    return float(-drawdowns.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series (negative values = underwater)."""
    cum = _to_cumulative(returns)
    rolling_max = cum.cummax()
    return (cum - rolling_max) / rolling_max


def calmar_ratio(returns: pd.Series, periods_per_year: int | None = None) -> float:
    """Annualised return divided by maximum drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return float(annualised_return(returns, periods_per_year) / mdd)


# ---------------------------------------------------------------------------
# Replica-specific metrics
# ---------------------------------------------------------------------------

def tracking_error(
    replica_returns: pd.Series,
    target_returns: pd.Series,
    periods_per_year: int | None = None,
) -> float:
    """
    Annualised tracking error (std of the return difference).

    Parameters
    ----------
    replica_returns : pd.Series
        Returns of the replica portfolio.
    target_returns : pd.Series
        Returns of the target (e.g. HFRX index).
    """
    diff = replica_returns - target_returns
    aligned = diff.dropna()
    ann = _annualisation_factor(aligned, periods_per_year)
    return float(aligned.std() * np.sqrt(ann))


def information_ratio(
    replica_returns: pd.Series,
    target_returns: pd.Series,
    periods_per_year: int | None = None,
) -> float:
    """
    Information ratio: annualised mean of return difference / tracking error.

    A positive IR means the replica outperforms the target on average.
    In a pure replication context, we want IR ≈ 0 with low TE.
    """
    diff = replica_returns - target_returns
    aligned = diff.dropna()
    ann = _annualisation_factor(aligned, periods_per_year)

    te = aligned.std()
    if te == 0:
        return np.nan
    return float(aligned.mean() / te * np.sqrt(ann))


def correlation_with_target(
    replica_returns: pd.Series,
    target_returns: pd.Series,
) -> float:
    """Pearson correlation between replica and target returns."""
    aligned = pd.concat([replica_returns, target_returns], axis=1).dropna()
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def beta_to_target(
    replica_returns: pd.Series,
    target_returns: pd.Series,
) -> float:
    """OLS beta of replica on target."""
    aligned = pd.concat([replica_returns, target_returns], axis=1).dropna()
    r, t = aligned.iloc[:, 0], aligned.iloc[:, 1]
    cov = np.cov(r, t)
    var_t = t.var()
    if var_t == 0:
        return np.nan
    return float(cov[0, 1] / var_t)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def performance_summary(
    replica_returns: pd.Series,
    target_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
    label_replica: str = "Replica",
    label_target: str = "Target",
) -> pd.DataFrame:
    """
    One-stop summary table comparing replica vs target.

    Returns a DataFrame with one column per portfolio and rows for each metric.

    Parameters
    ----------
    replica_returns : pd.Series
    target_returns : pd.Series
    risk_free_rate : float
        Annualised risk-free rate.
    periods_per_year : int | None
        Override frequency detection.
    label_replica : str
    label_target : str
    """
    metrics: dict[str, dict] = {}

    for label, ret in [(label_replica, replica_returns), (label_target, target_returns)]:
        metrics[label] = {
            "Ann. Return (%)": annualised_return(ret, periods_per_year) * 100,
            "Ann. Volatility (%)": annualised_volatility(ret, periods_per_year) * 100,
            "Sharpe Ratio": sharpe_ratio(ret, risk_free_rate, periods_per_year),
            "Max Drawdown (%)": max_drawdown(ret) * 100,
            "Calmar Ratio": calmar_ratio(ret, periods_per_year),
        }

    df = pd.DataFrame(metrics)

    # Add cross-metrics
    df.loc["Tracking Error (%)"] = {
        label_replica: tracking_error(replica_returns, target_returns, periods_per_year) * 100,
        label_target: np.nan,
    }
    df.loc["Information Ratio"] = {
        label_replica: information_ratio(replica_returns, target_returns, periods_per_year),
        label_target: np.nan,
    }
    df.loc["Correlation w/ Target"] = {
        label_replica: correlation_with_target(replica_returns, target_returns),
        label_target: np.nan,
    }
    df.loc["Beta to Target"] = {
        label_replica: beta_to_target(replica_returns, target_returns),
        label_target: np.nan,
    }

    return df.round(4)


# ---------------------------------------------------------------------------
# Rolling metrics (useful for time-varying analysis)
# ---------------------------------------------------------------------------

def rolling_tracking_error(
    replica_returns: pd.Series,
    target_returns: pd.Series,
    window: int = 52,
    periods_per_year: int | None = None,
) -> pd.Series:
    """
    Rolling annualised tracking error.

    Parameters
    ----------
    window : int
        Rolling window length in periods (default 52 = 1 year of weekly data).
    """
    diff = replica_returns - target_returns
    ann = _annualisation_factor(diff.dropna(), periods_per_year)
    return diff.rolling(window).std() * np.sqrt(ann)


def rolling_information_ratio(
    replica_returns: pd.Series,
    target_returns: pd.Series,
    window: int = 52,
    periods_per_year: int | None = None,
) -> pd.Series:
    """Rolling annualised information ratio."""
    diff = replica_returns - target_returns
    ann = _annualisation_factor(diff.dropna(), periods_per_year)
    rolling_mean = diff.rolling(window).mean()
    rolling_std = diff.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(ann)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 52,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    ann = _annualisation_factor(returns, periods_per_year)
    rf_periodic = (1 + risk_free_rate) ** (1 / ann) - 1
    excess = returns - rf_periodic
    return (excess.rolling(window).mean() / excess.rolling(window).std()) * np.sqrt(ann)