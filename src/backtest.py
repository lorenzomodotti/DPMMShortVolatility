import numpy as np

def simulate_trading_strategy(df, cluster_index, edge_threshold=1.0, prob_threshold=0.75):
    """
    Backtest trading strategies
    """
    
    # Volatility edge
    df['edge'] = df['atm_iv'] - df['vol_hat']
    
    # Position
    df['position'] = 0.0 
    
    # High probability of being in normal regime
    mask_safe = df.iloc[:,cluster_index] > (prob_threshold)
    # High edge
    mask_value = df['edge'] > edge_threshold
    
    # Trading signal when both conditions are met
    df.loc[mask_safe & mask_value, 'position'] = 1.0
    
    # Daily realized variance
    realized_var_annual = (df['close'].pct_change() ** 2) * 252
    # Daily implied variance
    implied_var_annual = (df['atm_iv']/100) ** 2
    
    # Theoretical PnL of delta-hedged short vol position (naive strategy)
    df['naive_daily_return'] = 0.5 * (implied_var_annual.shift(1) - realized_var_annual)
    
    # Apply trading signal (conditional strategy)
    df['conditional_daily_return'] = df['naive_daily_return'] * df['position'].shift(1)

    # Cumulative PnL of naive strategy
    df['naive_curve'] = (1 + df['naive_daily_return']).cumprod()
    
    # Cumulative PnL of conditional strategy
    df['conditional_curve'] = (1 + df['conditional_daily_return']).cumprod()
    
    return df

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def compute_total_returns(curve):
    curve = curve.dropna()
    return 100 * (curve.iloc[-1] - curve.iloc[0]) / curve.iloc[0]