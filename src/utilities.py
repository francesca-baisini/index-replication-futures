import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox



def calculate_main_metrics(data, annual_factor=52, plot=False):
    """
    Computes key financial performance metrics for each column in the provided dataset.
    
    Theoretical Support:
    - Annualized Return: Arithmetic mean scaled by time factor, assuming return linearity.
    - Annualized Volatility: Standard deviation scaled by the square root of time, assuming IID returns.
    - Sharpe Ratio: Normalizes excess return by the unit of risk to compare different risk profiles.
    - Max Drawdown: Captures the peak-to-trough decline to assess worst-case loss scenarios.
    - Skewness & Kurtosis: Measure non-normality and tail risk (Leptokurtosis), essential for non-linear Hedge Fund strategies.

    Parameters:
    data (pd.DataFrame): Time series of asset prices.
    annual_factor (int): Observations per year (default 52 for weekly data).
    plot (bool): If True, generates subplots (Density, CumReturns, Drawdown) for each column.

    Returns:
    pd.DataFrame: A summary table with metrics as rows and assets as columns.
    """

    if isinstance(data, pd.Series):
        data = data.to_frame()
        
    # Calculate percentage returns
    returns = data
    
    # Calculate components
    ann_return = returns.mean() * annual_factor
    ann_volatility = returns.std() * np.sqrt(annual_factor)
    sharpe_ratio = ann_return / ann_volatility
    
    # Calculate Maximum Drawdown components
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    # Aggregate stats
    stats = pd.DataFrame({
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    }).T

    if plot:
        for col in data.columns:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Visual Analysis: {col}', fontsize=16, fontweight='bold')

            # 1. Density Plot of pct_change
            returns[col].plot(kind='density', ax=axes[0], color='blue')
            axes[0].set_title('Returns Density (Distribution)')
            axes[0].grid(True, alpha=0.3)

            # 2. Cumulative Returns Plot
            cum_returns[col].plot(ax=axes[1], color='green')
            axes[1].set_title('Cumulative Returns (Growth of 1$)')
            axes[1].grid(True, alpha=0.3)

            # 3. Drawdown Plot
            drawdown[col].plot(ax=axes[2], kind='area', color='red', alpha=0.3)
            axes[2].set_title('Drawdown Analysis')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

    return stats


def plot_qq(data):
    """
    Plots QQ-plots for all assets in the dataset using a subplot grid.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    num_assets = len(data.columns)
    cols = 3
    rows = (num_assets // cols) + (num_assets % cols > 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(data.columns):
        stats.probplot(data[col].dropna(), dist="norm", plot=axes[i])
        axes[i].set_title(f'QQ-Plot: {col}')
        axes[i].grid(True, alpha=0.3)
        
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()
    return



def plot_return_dist(series, bins=50):
    """
    Plots a professional histogram of asset returns superimposed with a 
    Normal Distribution curve.
    
    This diagnostic plot is used to visually identify stylized facts of financial 
    time series, such as skewness (asymmetry) and kurtosis (heavy tails / leptokurtosis), 
    by comparing the empirical data against a standard theoretical Gaussian distribution.
    
    Parameters:
    - series: pd.Series containing the asset returns.
    - bins: int, the number of histogram bins (default is 50).
    """
    # 1. Data Cleaning (removal of NaNs)
    clean_data = series.dropna()
    
    # 2. Style Configuration
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # 3. Plot Empirical Histogram and KDE (Kernel Density Estimate)
    sns.histplot(clean_data, bins=bins, kde=True, stat="density", 
                 color='skyblue', label='Empirical Returns Density', alpha=0.6)
    
    # 4. Superimpose Theoretical Normal Distribution
    # Generated using the empirical mean (mu) and standard deviation (std) of the series
    mu, std = clean_data.mean(), clean_data.std()
    x = np.linspace(clean_data.min(), clean_data.max(), 100)
    p = norm.pdf(x, mu, std)
    
    plt.plot(x, p, 'r--', linewidth=2, label=rf'Normal Dist. ($\mu$={mu:.4f}, $\sigma$={std:.4f})')
    
    # 5. Aesthetics and Labels
    plt.title(f'Distribution Analysis: {series.name if series.name else "Returns"}', fontsize=15, fontweight='bold')
    plt.xlabel('Returns', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    
    # Add reference lines for zero and the empirical mean
    plt.axvline(0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(mu, color='red', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    return


def stationarity_test(data, max_lags=52, window=52, doPlot=False):
    """
    Performs a comprehensive stationarity analysis: Visual, Statistical, and Spectral.
    """
    # Ensure input is a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    results_list = []

    for col in data.columns:
        series = data[col].dropna()
        if doPlot:
            # --- 1. Visual Analysis: Rolling Statistics ---
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'Stationarity Analysis: {col}', fontsize=16, fontweight='bold')

            # Rolling Mean & Std
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            
            axes[0, 0].plot(series, label='Actual', alpha=0.5)
            axes[0, 0].plot(rolling_mean, label='Rolling Mean', color='red')
            axes[0, 0].plot(rolling_std, label='Rolling Std', color='black')
            axes[0, 0].set_title('Rolling Mean & Standard Deviation')
            axes[0, 0].legend()

            # --- 2. Spectral Analysis (Periodogram) ---
            # We subtract the mean to focus on frequencies rather than the DC component
            f, Pxx_den = periodogram(series - series.mean())
            axes[0, 1].semilogy(f, Pxx_den)
            axes[0, 1].set_title('Power Spectral Density (Periodogram)')
            axes[0, 1].set_xlabel('Frequency')
            axes[0, 1].set_ylabel('PSD [V**2/Hz]')

            # --- 3. Autocorrelation (ACF & PACF) ---
            plot_acf(series, lags=max_lags, ax=axes[1, 0])
            plot_pacf(series, lags=max_lags, ax=axes[1, 1])
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # --- 4. Statistical Tests ---
        # Augmented Dickey-Fuller (ADF)
        adf_res = adfuller(series)
        # KPSS Test (Complementary to ADF)
        kpss_res = kpss(series, regression='c', nlags="auto")
        
        results_list.append({
            'Asset': col,
            'ADF p-value': adf_res[1],
            'KPSS p-value': kpss_res[1],
            'Stationary (ADF)': 'Yes' if adf_res[1] < 0.05 else 'No',
            'Stationary (KPSS)': 'Yes' if kpss_res[1] > 0.05 else 'No'
        })

    return pd.DataFrame(results_list)


def volatility_autocorrelation_test(data, max_lags=52):
    """
    Analyzes volatility clustering by plotting ACF and PACF of squared log-returns.
    
    Theoretical Support:
    - Squared Returns Proxy: Squaring returns removes the direction and isolates magnitude, 
      acting as a proxy for latent volatility.
    - Volatility Clustering: Significant spikes in these plots indicate that variance is not 
      Independent and Identically Distributed (I.I.D.), proving conditional heteroskedasticity.
    - ARCH Effect: A sharp cutoff in PACF after 'p' lags suggests an ARCH(p) process, 
      where current volatility is directly driven by its own past shocks.
    """
    # Ensure input is a DataFrame to handle multiple columns
    if isinstance(data, pd.Series):
        data = data.to_frame()

    for col in data.columns:
        # Calculate squared log-returns
        # We drop NaNs to avoid errors in the plotting functions
        squared_series = (data[col]**2).dropna()
        
        # Create a side-by-side visualization for each asset
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f'Volatility Memory Analysis: {col}', fontsize=14, fontweight='bold')

        # 1. ACF of squared returns (identifies persistence/MA components)
        plot_acf(squared_series, lags=max_lags, ax=axes[0])
        axes[0].set_title('ACF of Squared Returns')
        axes[0].grid(True, alpha=0.3)

        # 2. PACF of squared returns (identifies direct AR components/ARCH order)
        plot_pacf(squared_series, lags=max_lags, ax=axes[1])
        axes[1].set_title('PACF of Squared Returns')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    return
    


def rolling_window_pipeline(data, window_size=104, step=1):
    """
    Crea un generatore di finestre mobili per l'analisi out-of-sample.
    
    Input necessari:
    - data: DataFrame con le serie storiche (log-returns).
    - window_size: Dimensione della finestra di training (default 2 anni settimanali).
    - step: Di quanto spostare la finestra a ogni iterazione (solitamente 1).
    
    Ritorna:
    - Una lista di dizionari, dove ogni dizionario rappresenta uno 'step' temporale 
      con i relativi segmenti di train e test.
    """
    pipeline = []
    total_obs = len(data)
    
    # Il numero di finestre possibili
    n_windows = total_obs - window_size
    
    for i in range(0, n_windows, step):
        # Definizione dei limiti della finestra
        train_start = i
        train_end = i + window_size
        
        # Estrazione dei segmenti
        # Usiamo .iloc per garantire l'ordine cronologico senza shuffling
        train_df = data.iloc[train_start:train_end]
        test_df = data.iloc[train_end : train_end + step]
        
        # Creazione del dizionario per lo step corrente
        step_context = {
            'step_id': i,
            'forecast_date': test_df.index[0],
            'train': train_df,
            'test': test_df
        }
        
        pipeline.append(step_context)
        
    return pipeline


def plot_garch_performance(actual_returns, predicted_vol_series, model_name="Best Model"):
    """
    Plots the out-of-sample GARCH predicted volatility against realized proxies.
    
    Inputs:
    - actual_returns: pd.Series of the true log-returns.
    - predicted_vol_series: pd.Series of the forecasted volatility from our grid search.
    - model_name: String for the plot title.
    """
    
    # 1. Align the dates (Out-of-sample alignment)
    # We only take the returns that match the dates we forecasted
    aligned_returns = actual_returns.loc[predicted_vol_series.index]
    
    # 2. Calculate proxies for realized volatility
    # Proxy A: Absolute Returns (very noisy, but shows exact weekly shocks)
    absolute_returns = aligned_returns.abs()
    
    # Proxy B: Short-term Rolling Volatility (e.g., 4-week realizing volatility)
    # Annualized to match standard reporting, but scaled down to weekly for comparison
    rolling_realized_vol = aligned_returns.rolling(window=4).std()
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot absolute returns as light gray bars in the background
    ax.bar(absolute_returns.index, absolute_returns, width=5, color='gray', alpha=0.3, label='|Actual Returns| (Proxy)')
    
    # Plot the short-term realized volatility
    ax.plot(rolling_realized_vol.index, rolling_realized_vol, color='blue', alpha=0.6, linestyle='--', label='4-Week Realized Vol')
    
    # Plot the GARCH prediction (The Star of the Show)
    ax.plot(predicted_vol_series.index, predicted_vol_series, color='red', linewidth=2, label=f'GARCH Forecast ({model_name})')
    
    # Formatting
    ax.set_title(f'Volatility Forecast vs Realization: {model_name}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Weekly Volatility (Standard Deviation)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return



def check_residual_autocorrelation(actual_returns, predicted_vol_series, model_name, max_lags=40):
    """
    Calculates standardized squared residuals and plots ACF and PACF 
    to verify if the GARCH model successfully eliminated volatility clustering.
    """
    
    # 1. Align the actual returns with the out-of-sample forecast dates
    aligned_returns = actual_returns.loc[predicted_vol_series.index]
    
    # 2. Calculate Squared Standardized Residuals: z_t^2 = (r_t / sigma_t)^2
    # This represents the "remaining" variance that the model couldn't explain
    std_residuals = aligned_returns / predicted_vol_series
    std_residuals_sq = std_residuals**2
    
    # Drop any potential NaNs generated by missing forecasts
    std_residuals_sq = std_residuals_sq.dropna()
    
    # 3. Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'Diagnostic Check (Squared Std. Residuals): {model_name}', fontsize=15, fontweight='bold')
    
    # ACF Plot
    plot_acf(std_residuals_sq, lags=max_lags, ax=axes[0], alpha=0.05, color='darkblue')
    axes[0].set_title('ACF (Autocorrelation)')
    axes[0].set_xlabel('Lags (Weeks)')
    
    # PACF Plot
    plot_pacf(std_residuals_sq, lags=max_lags, ax=axes[1], alpha=0.05, color='darkred')
    axes[1].set_title('PACF (Partial Autocorrelation)')
    axes[1].set_xlabel('Lags (Weeks)')
    
    plt.tight_layout()
    plt.show()
    return



def compute_metrics(replica_series, aligned_target):
    """Standard annualized performance metrics."""
    annual = 52
    rr = replica_series.mean() * annual
    tr = aligned_target.mean()  * annual
    rv = replica_series.std()   * np.sqrt(annual)
    tv = aligned_target.std()   * np.sqrt(annual)
    te = (replica_series - aligned_target).std() * np.sqrt(annual)
    return {
        'replica_return':    rr,
        'target_return':     tr,
        'replica_vol':       rv,
        'target_vol':        tv,
        'replica_sharpe':    rr / rv if rv > 0 else 0,
        'target_sharpe':     tr / tv if tv > 0 else 0,
        'tracking_error':    te,
        'information_ratio': (rr - tr) / te if te > 0 else 0,
        'correlation':       replica_series.corr(aligned_target),
        'max_drawdown':      (1 - (1 + replica_series).cumprod() /
                              (1 + replica_series).cumprod().cummax()).max(),
    }

def ljung_box(replica_series, aligned_target):
    """Ljung-Box test on residuals at lags 1, 4, 8."""
    resid = replica_series.values - aligned_target.values
    lb    = acorr_ljungbox(resid, lags=[1, 4, 8], return_df=True)
    return lb['lb_pvalue'].iloc[0], lb['lb_pvalue'].iloc[1], lb['lb_pvalue'].iloc[2]




# ── Shared: metrics summary table ────────────────────────────────────────────
def print_metrics_table(result, model_name, target_name, window, extra_rows=None):
    tgt_cum = (1 + result['aligned_target']).cumprod()
    tgt_dd  = (1 - tgt_cum / tgt_cum.cummax()).max()
    rows = [
        ('Annualized return',     f"{result['target_return']*100:.2f}%",  f"{result['replica_return']*100:.2f}%"),
        ('Annualized volatility', f"{result['target_vol']*100:.2f}%",     f"{result['replica_vol']*100:.2f}%"),
        ('Sharpe ratio',          f"{result['target_sharpe']:.2f}",        f"{result['replica_sharpe']:.2f}"),
        ('Max Drawdown',          f"{tgt_dd*100:.2f}%",                    f"{result['max_drawdown']*100:.2f}%"),
        ('Tracking Error',        'N/A',                                    f"{result['tracking_error']*100:.2f}%"),
        ('Information Ratio',     'N/A',                                    f"{result['information_ratio']:.3f}"),
        ('Correlation',           'N/A',                                    f"{result['correlation']:.4f}"),
        ('Avg Gross Exposure',    'N/A',                                    f"{result['avg_gross_exposure']:.4f}"),
        ('Avg VaR (1%, 1M)',      'N/A',                                    f"{result['avg_var']*100:.2f}%"),
        ('VaR method',            'N/A',                                    f"{result['var_method']}"),
        ('LB p-value lag 1',      'N/A',                                    f"{result['lb_pval_lag1']:.3f}"),
        ('LB p-value lag 4',      'N/A',                                    f"{result['lb_pval_lag4']:.3f}"),
        ('LB p-value lag 8',      'N/A',                                    f"{result['lb_pval_lag8']:.3f}"),
    ]
    if extra_rows:
        for name, val in extra_rows:
            rows.append((name, 'N/A', val))
    df_out = pd.DataFrame(rows, columns=['Metric', 'Target', 'Replica'])
    print(f"\n{'='*60}")
    print(f"{model_name} — {target_name} | Best window: {window}w")
    print(f"{'='*60}")
    print(df_out.to_string(index=False))


# ── Shared: 2 core plots (cumulative returns + scatter) ───────────────────────
def plot_core(result, model_name, target_name, window, color):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} — {target_name} | Window: {window}w',
                 fontsize=13, fontweight='bold')

    axes[0].plot(result['cumulative_target'],  label='Target',   color='steelblue', lw=1.8)
    axes[0].plot(result['cumulative_replica'], label=model_name, color=color,       lw=1.8)
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Cumulative return')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    rep = result['replica_returns']; tgt = result['aligned_target']
    lim = max(abs(tgt).max(), abs(rep).max()) * 1.1
    axes[1].scatter(tgt, rep, alpha=0.4, s=12, color=color)
    axes[1].plot([-lim, lim], [-lim, lim], 'r--', lw=1.2, label='Perfect replica')
    axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
    axes[1].set_title('Weekly Returns: Target vs Replica')
    axes[1].set_xlabel('Target'); axes[1].set_ylabel('Replica')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()


def compute_metrics_kalman(result, name='Kalman'):
    rep = result['replica']
    tgt = result['target']
    diff = rep - tgt
    af = 52

    rv = rep.std() * np.sqrt(af)
    tv = tgt.std() * np.sqrt(af)
    te = diff.std() * np.sqrt(af)
    excess = (rep.mean() - tgt.mean()) * af

    cum = (1 + rep).cumprod()
    mdd = (1 - cum / cum.cummax()).max()

    return pd.Series({
        'replica_return':    rep.mean() * af,
        'target_return':     tgt.mean() * af,
        'replica_vol':       rv,
        'target_vol':        tv,
        'replica_sharpe':    (rep.mean() * af) / rv if rv > 0 else 0,
        'target_sharpe':     (tgt.mean() * af) / tv if tv > 0 else 0,
        'tracking_error':    te,
        'information_ratio': excess / te if te > 0 else 0,
        'correlation':       rep.corr(tgt),
        'max_drawdown':      mdd,
        'avg_gross_exposure': result['gross_exposure'].mean(),
        'avg_var':           result['var'].mean(),
    }, name=name)


def print_metrics_table_kalman(metrics, model_name, target_name):
    print(f"\n{'='*55}")
    print(f"  {model_name} — {target_name}")
    print(f"{'='*55}")
    fmt = {}
    for k, v in metrics.items():
        if 'sharpe' in k or 'ratio' in k or 'correlation' in k or 'exposure' in k:
            fmt[k] = f"{v:.3f}"
        else:
            fmt[k] = f"{v*100:.2f}%"
    for k, v in fmt.items():
        print(f"  {k:<22s} {v:>10s}")