import pandas as pd
import numpy as np
import sys
from itertools import product
from src.wfo import WalkForwardOptimizer
from src.config import PATH_DF_VALID_RESULTS, PATH_PLOT_EQUITY, PATH_PLOT_DRAWDOWN
from src.outputs import print_backtest_metrics, plot_equity_curve, plot_drawdown, plot_monthly_returns

# Length of training windows (calendar days)
window_train = 550

# Length of validation windows (calendar days)
window_val = 120

# Length of test windows (calendar days)
window_test = 120

# Initial capital allocation
initial_capital = 10_000_000

# Splippage percentage (already paying full spread by buying at ask and selling at bid)
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
# strategy_parameters = {0: (-np.inf, -np.inf)}

edge_thresholds = np.arange(0.01, 0.07, 0.01)
#strategy_parameters = {key: (et, -np.inf) for key, et in enumerate(edge_thresholds)}

fear_threshold = np.arange(1.30, 1.80, 0.025)
strategy_parameters = {key: (et, ft) for key, (et, ft) in enumerate(product(edge_thresholds, fear_threshold))}


if __name__ == "__main__":

    if len(sys.argv) > 1:
        id = sys.argv[1]
    else:
        id = 0

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

    backtest_metrics, equity_curve, df_valid_results = wfo.run_wfo()

    print_backtest_metrics(backtest_metrics)

    df_valid_results.to_csv(PATH_DF_VALID_RESULTS.format(id), index = False)

    plot_equity_curve(equity_curve, initial_capital, PATH_PLOT_EQUITY.format(id))

    plot_drawdown(equity_curve, 'Conditional Short Volatility', PATH_PLOT_DRAWDOWN.format(id))