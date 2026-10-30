import numpy as np
import warnings

def calculate_var_hs(returns, confidence=0.01, horizon=4):
    """
    Historical Simulation Value at Risk (HS-VaR).

    Methodology

    The steps are:
      1. Take the vector of observed portfolio returns r_1, ..., r_T.
      2. Scale each observation to the desired horizon h using the square-root-of-time
         rule:  r_scaled = r * sqrt(h).
      3. Sort the scaled returns in ascending order.
      4. The VaR at confidence level alpha is the alpha-th empirical quantile of
         the LOSS distribution, i.e. the negative of the alpha-th return quantile.

    Parameters
    ----------
    returns    : array-like
        Vector of observed portfolio returns (weekly, same currency).
        Must contain at least 2 non-NaN observations.
    confidence : float, default 0.01
        Left-tail probability level. 0.01 = 99% VaR, 0.05 = 95% VaR.
    horizon    : int, default 4
        Number of periods (weeks) to scale the VaR to.
        horizon=4 gives a 1-month VaR from weekly observations.

    Returns
    -------
    float
        VaR expressed as a positive number representing a loss.
        Example: 0.08 means "we estimate a loss of at most 8% with 99% probability
        over 1 month" — i.e. there is a 1% chance of losing MORE than 8%.

    
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    # 1. Validation: For 99% VaR, 1/0.01 = 100. We need at least 100 points.
    if len(returns) < (1 / confidence):
        
        warnings.warn(f"Sample size {len(returns)} is too small for {1-confidence}% confidence level.")
    
    if len(returns) < 2:
        return 0.0

    # 2. Scaling (Keeping your logic, but adding a note)
    # Theoretically, it is better to scale the RESULT, not the distribution.
    # var_1period = -np.percentile(returns, confidence * 100)
    # var_h = var_1period * np.sqrt(horizon)
    
    # Using 'nearest' or 'lower' method for np.percentile is often safer in Risk Mgmt
    var_1period = -np.percentile(returns, confidence * 100, method='linear')
    
    # Scale the resulting VaR
    var_h = var_1period * np.sqrt(horizon)

    return max(float(var_h), 0.0)



def calculate_var_whs(returns, confidence=0.01, lambda_val=0.97, horizon=4):
    """
    Weighted Historical Simulation (WHS) for VaR and Expected Shortfall.

    Methodology:
    1. Calculate losses from returns (Loss = -Return).
    2. Assign weights to each observation based on recency: w_s = C * lambda^(t-s).
       Recent observations get higher weights; distant ones decay exponentially.
    3. Scale losses to the target horizon using the square-root-of-time rule.
    4. Sort losses in descending order (worst losses first) while keeping 
       their assigned weights attached.
    5. Find the index i* such that the cumulative sum of weights of the 
       sorted losses reaches the target tail probability (alpha).
    6. VaR is the loss at that threshold; ES is the weighted average of 
       all losses exceeding that threshold.

    Parameters
    ----------
    returns    : array-like
        Vector of observed portfolio returns (ordered chronologically, 
        oldest first, most recent last).
    alpha      : float, default 0.01
        Tail probability (0.01 for 99% confidence, 0.05 for 95% confidence).
    lambda_val : float, default 0.97
        Decay factor. Typically between 0.95 and 0.99. 
        Lower values make the model more responsive to recent events.
    horizon    : int, default 4
        Number of periods to scale the metrics to (e.g., 4 for monthly 
        VaR from weekly data).

    Returns
    -------
    float
        Value at Risk as a positive number.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    n = len(returns)

    if n < 2:
        return {'VaR': 0.0, 'ES': 0.0}

    # Step 1: Convert returns to losses and scale by horizon
    # Note: Losses are (-returns)
    losses = -returns * np.sqrt(horizon)

    # Step 2: Calculate weights w_s = C * lambda^(t-s)
    # t is current time, s is observation time.
    # The most recent observation (index n-1) gets weight C * lambda^0
    # The oldest observation (index 0) gets weight C * lambda^(n-1)
    
    # Generate powers: [n-1, n-2, ..., 0]
    powers = np.arange(n - 1, -1, -1)
    raw_weights = lambda_val ** powers
    
    # Normalization factor C = (1-lambda) / (1-lambda^n)
    # This ensures sum(weights) == 1
    weights = raw_weights / np.sum(raw_weights)

    # Step 3: Sort losses in DESCENDING order (worst losses first)
    # and reorder weights to match the sorted losses
    idx = np.argsort(losses)[::-1]
    sorted_losses = losses[idx]
    sorted_weights = weights[idx]

    # Step 4: Calculate Cumulative Weights
    cum_weights = np.cumsum(sorted_weights)

    # Step 5: Find VaR (L^i*) 
    # Per formula: i* is the largest value s.t. sum(weights) <= alpha
    # We find the first index where the cumulative weight exceeds alpha
    indices_below_alpha = np.where(cum_weights <= confidence)[0]
    
    if len(indices_below_alpha) == 0:
        # If the single worst loss has a weight > alpha, VaR is that loss
        i_star_idx = 0
    else:
        i_star_idx = indices_below_alpha[-1]

    var_val = sorted_losses[i_star_idx]

    # Step 6: Calculate Expected Shortfall (ES)
    # Weighted average of losses from the worst up to i*
    tail_losses = sorted_losses[:i_star_idx + 1]
    tail_weights = sorted_weights[:i_star_idx + 1]
    

    # Final floor at 0 for reporting consistency
    return max(float(var_val), 0.0)
    