import pandas as pd
import numpy as np
from itertools import product
from src.wfo import WalkForwardOptimizer
from src.config import *
from src.outputs import *

# Length of training windows (calendar days)
window_train = 550

# Length of validation windows (calendar days)
window_val = 120

# Length of test windows (calendar days)
window_test = 120

# Initial capital allocation
initial_capital = 10_000_000

# Slippage percentage (already paying full spread by buying at ask and selling at bid)
slippage_pct = 0.0

# Fixed trading fee
fixed_fee = 1.40

# Leverage
leverage = 2.5

# Horizon to hold options
horizon = 7

# Portfolio margin requirement
margin_requirement = 0.2

# Minimum number of trades per parameter configuration
minimum_trades = 10

# Number of trading days (effective) to compute rolling mean for ATM IV
tracker_window = 60

# Strategy parameters to optimize
edge_thresholds = np.arange(0.01, 0.07, 0.01)
fear_threshold = np.arange(1.30, 1.80, 0.025)

if __name__ == "__main__":

    # Short Straddle (No Rules)
    strategy_name = 'Short Straddle (No Rule)'
    strategy_parameters = {0: (-np.inf, +np.inf)}
    wfo = WalkForwardOptimizer(
        window_train,
        window_val,
        window_test,
        strategy_parameters,
        initial_capital, 
        slippage_pct,
        fixed_fee,
        leverage,
        horizon,
        margin_requirement,
        minimum_trades,
        tracker_window,
        spx=True
    )
    backtest_metrics_no, equity_curve_no, daily_returns_no, _ = wfo.run_wfo()

    df_test_data = wfo.get_test_data()
    spx_returns = df_test_data['close'].pct_change().dropna()
    print_backtest_metrics(backtest_metrics_no, strategy_name)
    plot_drawdown(equity_curve_no, strategy_name, PATH_PLOT_DRAWDOWN.format("no"))
    plot_monthly_returns(daily_returns_no, strategy_name, PATH_PLOT_RETURNS.format("no"))
    plot_rolling_correlation(daily_returns_no, spx_returns, strategy_name, 60, PATH_PLOT_CORR.format("no"))

    # Short Straddle (Edge Rule)
    strategy_name = 'Short Straddle (Edge Rule)'
    strategy_parameters = {key: (et, +np.inf) for key, et in enumerate(edge_thresholds)}
    wfo = WalkForwardOptimizer(
        window_train,
        window_val,
        window_test,
        strategy_parameters,
        initial_capital, 
        slippage_pct,
        fixed_fee,
        leverage,
        horizon,
        margin_requirement,
        minimum_trades,
        tracker_window,
        spx=True
    )
    backtest_metrics_e, equity_curve_e, daily_returns_e, _ = wfo.run_wfo()
    df_test_data = wfo.get_test_data()
    spx_returns = df_test_data['close'].pct_change().dropna()
    print_backtest_metrics(backtest_metrics_e, strategy_name)
    plot_drawdown(equity_curve_e, strategy_name, PATH_PLOT_DRAWDOWN.format("e"))
    plot_monthly_returns(daily_returns_e, strategy_name, PATH_PLOT_RETURNS.format("e"))
    plot_rolling_correlation(daily_returns_e, spx_returns, strategy_name, 60, PATH_PLOT_CORR.format("e"))

    # Short Straddle (Fear Rule)
    strategy_name = 'Short Straddle (Fear Rule)'
    strategy_parameters = {key: (-np.inf, ft) for key, ft in enumerate(fear_threshold)}
    wfo = WalkForwardOptimizer(
        window_train,
        window_val,
        window_test,
        strategy_parameters,
        initial_capital, 
        slippage_pct,
        fixed_fee,
        leverage,
        horizon,
        margin_requirement,
        minimum_trades,
        tracker_window,
        spx=True
    )
    backtest_metrics_f, equity_curve_f, daily_returns_f, _ = wfo.run_wfo()
    df_test_data = wfo.get_test_data()
    spx_returns = df_test_data['close'].pct_change().dropna()
    print_backtest_metrics(backtest_metrics_f, strategy_name)
    plot_drawdown(equity_curve_f, strategy_name, PATH_PLOT_DRAWDOWN.format("f"))
    plot_monthly_returns(daily_returns_f, strategy_name, PATH_PLOT_RETURNS.format("f"))
    plot_rolling_correlation(daily_returns_f, spx_returns, strategy_name, 60, PATH_PLOT_CORR.format("f"))

    # Short Straddle (Edge+Fear Rules)
    strategy_name = 'Short Straddle (Edge+Fear Rule)'
    strategy_parameters = {key: (et, ft) for key, (et, ft) in enumerate(product(edge_thresholds, fear_threshold))}
    wfo = WalkForwardOptimizer(
        window_train,
        window_val,
        window_test,
        strategy_parameters,
        initial_capital, 
        slippage_pct,
        fixed_fee,
        leverage,
        horizon,
        margin_requirement,
        minimum_trades,
        tracker_window,
        spx=True
    )
    backtest_metrics_ef, equity_curve_ef, daily_returns_ef, _ = wfo.run_wfo()
    print_backtest_metrics(backtest_metrics_ef, strategy_name)
    plot_drawdown(equity_curve_ef, strategy_name, PATH_PLOT_DRAWDOWN.format("ef"))
    plot_monthly_returns(daily_returns_ef, strategy_name, PATH_PLOT_RETURNS.format("ef"))
    plot_rolling_correlation(daily_returns_ef, spx_returns, strategy_name, 60, PATH_PLOT_CORR.format("ef"))

    equity_curves = pd.concat(
        [
            equity_curve_no,
            equity_curve_e,
            equity_curve_f,
            equity_curve_ef
        ],
        axis=1
    )
    equity_curves.columns = ['no_rule', 'edge_rule', 'fear_rule', 'edge_fear_rule']
    plot_equity_curves(equity_curves, initial_capital, PATH_PLOT_EQUITY)

    df_test_data = wfo.get_test_data()
    print_backtest_metrics(get_metrics(df_test_data['close'], df_test_data['r_daily']), strategy = "Long")