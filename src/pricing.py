import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

def black_scholes_pricing(S, K, T, r, sigma, flag_call):
    """
    Compute option price according to Black-Scholes formula
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if flag_call:
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_iv(row, r_fallback=None):
    """
    Invert the Black-Scholes formula to get the implied volatility with strict arbitrage boundary check
    """
    price = row['option_price']
    S = row['underlying_price']
    K = row['strike']
    T = row['time_to_expiry'] / 365.0
    is_call = (row['call_put'] == 'Call')
    
    # Use historical rate if available
    r = row['r'] if 'r' in row else r_fallback

    discount_factor = np.exp(-r * T)
    
    if is_call:
        # Call Floor: S - PV(K)
        theoretical_floor = S - (K * discount_factor)
    else:
        # Put Floor: PV(K) - S
        theoretical_floor = (K * discount_factor) - S
    
    # If price is below the theoretical floor the model cannot solve it
    if price <= max(0, theoretical_floor) + 1e-6:
        return np.nan

    def objective(sigma):
        return black_scholes_pricing(S, K, T, r, sigma, is_call) - price
    
    try:
        return brentq(objective, 0.001, 5.0, xtol=1e-4)
    except:
        return np.nan

def update_iv(df, error_tolerance, r_fallback=None):
    """
    Substitute implied volatility where pricing errors occurred.
    """
    # Pricing errors
    bad_mask = df['pricing_error'].abs() > error_tolerance
    
    # Recompute IV
    recomputed_ivs = df[bad_mask].apply(lambda row: bs_iv(row, r_fallback), axis=1)
    
    # Update data
    valid_recompute = recomputed_ivs.notna()
    indices_to_update = valid_recompute.index[valid_recompute]
    df.loc[indices_to_update, 'vol'] = recomputed_ivs[indices_to_update]
    return df

def validate_pricing_data(df, r):
    """
    Compute difference in theoretical (Black-Scholes) and observed price
    """
    sigma = df['vol'] 
    S = df['underlying_price']
    K = df['strike']
    T = df['time_to_expiry'] / 365.0
    
    if 'r' in df.columns:
        r_calc = df['r']
    else:
        r_calc = r
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r_calc + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    n_d1 = norm.cdf(-d1)
    n_d2 = norm.cdf(-d2)
    
    discount_factor = np.exp(-r_calc * T)
    
    call_prices = S * nd1 - K * discount_factor * nd2
    put_prices = K * discount_factor * n_d2 - S * n_d1
    
    is_call = (df['call_put'] == 'Call')
    df['theoretical_price'] = np.where(is_call, call_prices, put_prices)
    df['pricing_error'] = df['theoretical_price'] - df['option_price']
    return df