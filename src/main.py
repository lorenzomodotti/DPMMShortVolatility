import pandas as pd
import numpy as np
import sys
from itertools import product
from src.wfo import WalkForwardOptimizer
from src.config import PATH_DF_VALID_RESULTS
from src.outputs import print_backtest_metrics, plot_equity_curve, plot_drawdown, plot_monthly_returns

# Length of training windows (calendar days)
window_train = 550

# Length of validation windows (calendar days)
window_val = 180

# Length of test windows (calendar days)
window_test = 120

# Initial capital allocation
initial_capital = 100_000_000

# Splippage percentage
slippage_pct = 0.0

# Fixed trading fee
fixed_fee = 0.0

# Leverage
leverage = 1.0

# Horizon
horizon = 21

# Minimum number of trades per parameter configuration
minimum_trades = 15

# Strategy parameters to optimize
# strategy_parameters = {0: (-np.inf, -np.inf)}

edge_thresholds = np.arange(0.0050, 0.0275, 0.0025)
#strategy_parameters = {key: (et, -np.inf) for key, et in enumerate(edge_thresholds)}

prob_thresholds = np.arange(0.60, 0.80, 0.025)
strategy_parameters = {key: (et, pt) for key, (et, pt) in enumerate(product(edge_thresholds, prob_thresholds))}



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
        minimum_trades
    )

    backtest_metrics, equity_curve, df_valid_results = wfo.run_wfo()

    print_backtest_metrics(backtest_metrics)

    df_valid_results.to_csv(PATH_DF_VALID_RESULTS.format(id), index = False)

    plot_equity_curve(equity_curve, initial_capital)

    plot_drawdown(equity_curve, 'Conditional Short Volatility')