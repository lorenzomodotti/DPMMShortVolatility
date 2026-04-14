# Number of moneyness levels (quantiles)
M = 16

# Number of regimes to identify
K = 4

# Target expiry (days)
target_time_to_expiry = 30

# Forecast horizon (days)
forecast_horizon = 21

# Path to processed 1-minute intraday SPY OHLC data
PATH_SPY_INTRADAY_OHLC = 'data/spy_1min_ohlc.parquet'

# Path to processed treasury daily rate
PATH_TREASURY_DAILY_RATE = 'data/treasury_daily_rate.parquet'

# Path to processed SPY daily realized volatility
PATH_SPY_DAILY_VOL = 'data/spy_daily_realized_volatility.parquet'

# Path to processed SPY daily OHLC
PATH_SPY_OPTION_CHAIN = 'data/spy_option_chain.parquet'

# Path to SPY dividend data
PATH_SPY_DIVIDED = "raw_data/spy_dividend.csv"

# Path to dataframe with WFO validation/optimization results
PATH_DF_VALID_RESULTS = "outputs/wfo_valid_results_{}.csv"

# Path volatility plot
PATH_PLOT_VOLATILITY = "plots/volatility.png"

# Path price plot
PATH_PLOT_PRICE = "plots/price.png"

# Path price signals
PATH_PLOT_SIGNALS = "plots/signals.png"

# Path equity curve plot
PATH_PLOT_EQUITY = "plots/equity_curves.png"

# Path drawdown plot
PATH_PLOT_DRAWDOWN = "plots/drawdown_{}.png"

# Path returns map
PATH_PLOT_RETURNS = "plots/returns_{}.png"

# Path correlation plot
PATH_PLOT_CORR = "plots/correlation_{}.png"