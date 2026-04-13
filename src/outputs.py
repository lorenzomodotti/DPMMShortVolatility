"""
Logic to plot and print data
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib import colormaps
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from src.config import *

def plot_volatility(vol_forecast, atm_iv, vol, path = None):
    """
    Plot forecasted volatility by GARCH, ATM implied volatility, and realized volatility
    """

    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(vol.index, vol, label='21-day Average Realized Volatility', color='darkgray')
    ax.fill_between(vol.index, vol, 0, color='darkgray', alpha=0.25)
    ax.plot(vol_forecast.index, vol_forecast, label='21-day Forecasted Volatility', color='black')
    ax.plot(atm_iv.index, atm_iv, label='30-day ATM Implied Volatility', color='slateblue')
    
    ax.set_title("Volatility Comparison", fontsize=14)
    ax.set_ylabel("", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.2f}'))
    ax.grid(True, alpha=0.2)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_cluster_centroids(dpmm, spline_transformer, labels=None, path = None):
    """
    Plot the K cluster centroids (volatility smiles) learned by the DPMM
    """
    dpmm.eval()
    K = dpmm.K
    
    # Grid for plotting
    plot_grid = np.linspace(spline_transformer.lower_bound, spline_transformer.upper_bound, 25)
    
    # BSpline basis for plotting grid
    basis_grid = torch.tensor(spline_transformer(plot_grid), dtype=torch.float32)
    
    # Compute implied volatility curve from learned coefficients
    with torch.no_grad():
        coeffs = dpmm.q_mu.cpu()
        curves = torch.matmul(basis_grid.cpu(), coeffs.T).numpy()

    color_palette = colormaps['Pastel2']

    if labels is None:
        labels = [f"Cluster {k}" for k in range(K)]
        colors = [color_palette(k) for k in range(K)]
    else:
        colors = []
        remaining_indices = [i for i in range(K) if i != 1]
        other_idx_iter = iter(remaining_indices)
        for label in labels:
            if label == 'Panic Regime':
                colors.append(color_palette(1))
            else:
                colors.append(color_palette(next(other_idx_iter)))

        
    plt.figure(figsize=(9, 5))
    for k in range(K):
        plt.plot(plot_grid, curves[:, k], linewidth=2.5, color=colors[k], label=labels[k])
    plt.title(f"Market Regimes (Posterior Centroids)", fontsize=14)
    plt.xlabel("Log-Moneyness", fontsize=12)
    plt.ylabel("30-day ATM Implied Volatility", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3, frameon=False)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_cluster_probabilities(df_regimes, vol_series, price_series, rolling_window=1, labels=None, path = None):
    """
    Plot the evolution of cluster probabilities over time using a stacked area chart
    """

    # Smoothing to improve readability
    if rolling_window > 1:
        df_regimes = df_regimes.rolling(window=rolling_window).mean().dropna()
        # Normalization 
        df_regimes = df_regimes.div(df_regimes.sum(axis=1), axis=0)

    K = df_regimes.shape[1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), sharex=True)

    color_palette = colormaps['Pastel2']

    if labels is None:
        labels = [f"Cluster {k}" for k in range(K)]
        colors = [color_palette(k) for k in range(K)]
    else:
        colors = []
        remaining_indices = [i for i in range(K) if i != 1]
        other_idx_iter = iter(remaining_indices)
        for label in labels:
            if label == 'Panic Regime':
                colors.append(color_palette(1))
            else:
                colors.append(color_palette(next(other_idx_iter)))

    # Upper plot: regimes vs vol

    ax1.stackplot(df_regimes.index, df_regimes.T, colors=colors, alpha=0.85)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Posterior Probability", fontsize=12)

    ax11 = ax1.twinx()
    ax11.plot(vol_series.index, vol_series, color='black', label='SPY Volatility')
    ax11.set_ylabel("Volatility", fontsize=12, rotation=270, labelpad=20)

    # Lower plot: regimes vs price

    ax2.stackplot(df_regimes.index, df_regimes.T, labels=labels, colors=colors, alpha=0.85)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Posterior Probability", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)

    ax22 = ax2.twinx()
    ax22.plot(price_series.index, price_series, color='black', linestyle = '--', label='SPY Price')
    ax22.set_ylabel("Price (Log Scale)", fontsize=12, rotation=270, labelpad=20)
    ax22.set_yscale('log')
    
    # Format Date Axis
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle("Market Regimes (Posterior Centroids)", fontsize=14)
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=(K), frameon=False)
    
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_performance(results_df, df_daily):
    """
    Plot the performance of the trading strategies
    """
    
    # SPY Buy & Hold (normalized to start at 1)
    spy_curve = df_daily.loc[results_df.index, 'close']
    spy_curve = 100 * spy_curve / spy_curve.iloc[0]
    
    # Strategy volatility shorting with Bayesian rule
    strategy_curve = 100 * results_df['conditional_curve']

    naive_curve = (100 * results_df['naive_curve']).clip(lower=0)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(strategy_curve.index, strategy_curve, label='Conditional Short Volatility', color='black', linewidth=2)
    ax.plot(naive_curve.index, naive_curve, label='Naive Short Volatility', color="black", linestyle='--')
    ax.plot(spy_curve.index, spy_curve, label='Long SPY', color='grey', alpha=0.4)
    
    ax.set_title("Performance Comparison", fontsize=14)
    ax.set_ylabel("Returns (%)", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.2f}'))
    ax.grid(True, alpha=0.3)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.tight_layout()
    plt.show()

def plot_drawdown(curve, label, path=None):
    """
    Calculate and plot drawdown
    """
    # Calculate Drawdown
    rolling_max = curve.cummax()
    drawdown = (curve - rolling_max) / rolling_max
    
    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown, 0, color='gray', alpha=0.3)
    plt.plot(drawdown.index, drawdown, color='gray', linewidth=1)
    
    plt.title(f"{label} Risk Profile", fontsize=12)
    plt.ylabel("Drawdown (%)")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(True, alpha=0.3)
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_monthly_returns(daily_returns, label):
    """
    Transforms daily returns into a Month vs Year grid and plots a heatmap
    """
    # Compute monthly returns
    monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1).to_frame('Return')
    
    # Extract year and month
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    
    # Pivot table
    returns_grid = monthly_returns.pivot(index='Year', columns='Month', values='Return')
    
    # Map integers to months
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    returns_grid.columns = [month_map[c] for c in returns_grid.columns]
    
    # Compute annual total returns
    annual_rets = daily_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    returns_grid['Total'] = annual_rets.values
    
    # Heatmap plot
    plt.figure(figsize=(12, len(returns_grid) * 0.8)) # Dynamic height
    ax = sns.heatmap(returns_grid, annot=True, fmt='.1%', 
                     cmap='RdYlGn', center=0, cbar=False,
                     linewidths=1, linecolor='white')
    
    plt.title(f'{label} Monthly Returns', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('')
    plt.xlabel('')
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.show()
    return returns_grid

def plot_rolling_correlation(results_df, df_daily, window=60):
    """
    Plot the rolling correlation between the strategy and SPY
    """
    
    # Align dates
    spy_returns = df_daily['return']
    strategy_returns = results_df['conditional_daily_return']
    
    # Combine returns
    comparison = pd.DataFrame({
        'Strategy': strategy_returns,
        'SPY': spy_returns
    }).dropna()
    
    # Compute rolling correlation
    rolling_corr = comparison['Strategy'].rolling(window=window).corr(comparison['SPY'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(rolling_corr.index, rolling_corr, color='black', linewidth=1.5)

    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(-0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    avg_corr = rolling_corr.mean()
    ax.set_title(f"Rolling Correlation: Conditional Short Volatility vs Long SPY\nAverage Correlation: {avg_corr:.2f}", fontsize=14)
    ax.set_ylabel(f'{window}-day Correlation', fontsize=12)
    ax.set_ylim(-1.0, 1.0) # Correlation is bounded [-1, 1]
    
    ax.fill_between(rolling_corr.index, rolling_corr, 0, where=(rolling_corr >= 0), color='black', alpha=0.1)
    ax.fill_between(rolling_corr.index, rolling_corr, 0, where=(rolling_corr < 0), color='red', alpha=0.1)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_returns(returns_dict):
    print("\nReturns:")
    print(f" > Long SPY: {returns_dict['long']:.3f}%")
    print(f" > Naive Short Volatility: {returns_dict['naive']:.3f}%")
    print(f" > Conditional Short Volatility: {returns_dict['conditional']:.3f}%")

def print_sharpes(sharpes_dict):
    print("\nSharpe Ratio:")
    print(f" > Long SPY: {sharpes_dict['long']:.3f}")
    print(f" > Naive Short Volatility: {sharpes_dict['naive']:.3f}")
    print(f" > Conditional Short Volatility: {sharpes_dict['conditional']:.3f}")

def print_backtest_metrics(backtest_metrics):
    print("\n----- Backtest Results -----")
    print(f"Total Return: {backtest_metrics['total_return']:.4f}")
    print(f"Sharpe Ratio (Alpha): {backtest_metrics['alpha_sharpe_ratio']:.4f}")
    print(f"Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio: {backtest_metrics['sortino_ratio']:.4f}")
    print(f"Maximum Drawdown: {backtest_metrics['max_drawdown']:.4f}")
    print(f"Number of Trades: {backtest_metrics['trade_count']:.0f}")
    print(f"Turnover: {backtest_metrics['turnover']:.0f}")
    print("----- ---------------- -----\n")

def plot_equity_curve(equity_curve, initial_capital, path = None):
    """
    Plot the performance of the trading strategies
    """

    equity_curve = equity_curve / initial_capital
    
    plt.figure(figsize=(12, 4))
    
    plt.plot(equity_curve.index, equity_curve, label='Conditional Short Volatility', color='black', linewidth=2)
    
    plt.title("Performance Comparison", fontsize=14)
    plt.ylabel("Returns (%)", fontsize=12)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_price(price_series, path = None):

    plt.figure(figsize=(12, 4))

    plt.plot(price_series.index, price_series, color='black', linestyle = '-')
    plt.ylabel("SPY Price")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_signals(edge, fear_score, path = None):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(edge.index, edge, color='black', linestyle = '-', label='Edge')
    ax1.set_ylabel("Edge")
    ax1.grid(True, alpha=0.3)

    ax2.plot(fear_score.index, fear_score, color='black', linestyle = '-', label='Fear Score')
    ax2.set_ylabel("Fear Score", rotation=270, labelpad=15)
    ax2.grid(True, alpha=0.3)

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.03), frameon=False)

    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()