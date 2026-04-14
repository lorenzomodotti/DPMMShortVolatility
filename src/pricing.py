import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

def bs_price(F, K, T, r, sigma, flag_call):
    """
    Compute option price according to Black-Scholes formula (Black-76 with forward price)
    """
    d1 = (np.log(F/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag_call:
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def bs_iv(row):
    """
    Invert the Black-Scholes formula to get the implied volatility with boundary checks
    """
    price = row['mid']
    F = row['forward']
    K = row['strike']
    T = row['time_to_expiry']
    r = row['risk_free_rate']
    is_call = (row['call_put'] == 'Call')
    df = np.exp(-r * T)
    
    # The price must be greater than the discounted payoff
    if is_call:
        intrinsic = df * max(0, F - K)
    else:
        intrinsic = df * max(0, K - F)
        
    # A call cannot be worth more than the discounted forward
    # A put cannot be worth more than the discounted strike
    upper_bound = df * F if is_call else df * K

    if price <= intrinsic + 1e-7 or price >= upper_bound - 1e-7:
        return np.nan

    def objective(sigma):
        return bs_price(F, K, T, r, sigma, is_call) - price
    
    try:
        return brentq(objective, 0.0001, 5.0, xtol=1e-5)
    except (ValueError, RuntimeError):
        return np.nan
    
def compute_sharpe_ratio(returns, risk_free_rate = 0.0):
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def compute_sortino_ratio(returns, risk_free_rate = 0.0):
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 2 or downside_returns.std() == 0:
        return 0
    return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)