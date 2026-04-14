import warnings
warnings.filterwarnings("ignore")

import random
import os
import pandas as pd
import numpy as np
from src.data_transformers import SplineTransformer, IVSmoother
from src.dpmm import dpmm_train, dpmm_forecast
from src.har import har_train, har_forecast
from src.backtester import Backtester
from src.logger import LOGGER
from src.outputs import *
from src.config import *

def seed_everything(seed=124):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def _get_spread_ratio(options, target_time_to_expiry, w = 0.7, k = 2):

    target_time_to_expiry /= 365.0
    
    # Available expiries for the trading day
    distances_T = (options['time_to_expiry'] - target_time_to_expiry).abs() / target_time_to_expiry
    # Available strikes for the trading day
    distances_K = (options['log_moneyness']).abs()
    # Metric
    distances = (w * distances_T**2 + (1 - w) * distances_K**2)**0.5
    # Best options w.r.t metric 
    top_k_indices = np.argsort(distances)[:k]

    # Weight as normalized distance
    top_distances = distances.iloc[top_k_indices].values
    inverse_distances = 1.0 / (top_distances + 1e-8)
    normalized_distances = inverse_distances / np.sum(inverse_distances)
    
    # Compute spread ratios
    top_spread_ratios = ((options['ask'] - options['bid']) / options['mid']).iloc[top_k_indices].values
    # Compute forward
    top_forward = options['forward'].iloc[top_k_indices].values

    return np.inner(top_spread_ratios, normalized_distances), np.inner(top_forward, normalized_distances)


def get_df_strategy(df_option_chain_test, fear_score, atm_iv_test, har_vol_forecast, spot_price, trading_days):
    # Filter the option chain immediately to only include common trading days
    df_filtered = df_option_chain_test[df_option_chain_test.index.get_level_values('date').isin(trading_days)]
    
    # Group per trading day
    grouped = df_filtered.groupby('date')
    
    # Initialize dictionaries for cleaner Series creation
    sr_dict = {}
    f_dict = {}
    
    # Compute spread ratios and forwards only for the validated dates
    for date, group in grouped:
        sr, f = _get_spread_ratio(group, target_time_to_expiry)
        sr_dict[date] = sr
        f_dict[date] = f

    # Convert results to Series
    spread_ratios = pd.Series(sr_dict, name='spread_ratio')
    forwards = pd.Series(f_dict, name='forward')

    # Concatenate and align everything to the 'trading_days' index
    # We use .loc[trading_days] to ensure order and presence
    df_strategy = pd.concat(
        [
            fear_score.loc[trading_days],
            atm_iv_test.loc[trading_days],
            har_vol_forecast.loc[trading_days],
            spread_ratios.loc[trading_days],
            forwards.loc[trading_days],
            spot_price.loc[trading_days]
        ],
        axis=1
    ).dropna() # Drops any days where a component (like HAR forecast) might be NaN

    # Compute edge
    df_strategy['edge'] = df_strategy['atm_iv'] - df_strategy['vol_forecast'] - (df_strategy['spread_ratio'] * df_strategy['atm_iv'])

    return df_strategy


def get_df_backtest(df_strategy, df_treasury):
    # Use the index of df_strategy as the "source of truth" for valid days

    df_strategy = df_strategy.iloc[1:]

    valid_dates = df_strategy.index
    
    df_backtest = pd.concat(
        [
            df_strategy[['atm_iv', 'close', 'forward', 'spread_ratio']],
            df_treasury.loc[valid_dates, ['r_annual', 'r_daily']]
        ],
        axis=1
    ).dropna()
    
    return df_backtest.reset_index().rename(columns={'index': 'date'})


class ATMIVTracker:

    def __init__(self, window = 252):
        self.window = window
        self.history = pd.Series(dtype='float64')

    def get_historical_rolling_mean(self, new_data):
        start = len(self.history)
        # Append new data to global state
        self.history = pd.concat([self.history, new_data])
        # Compute rolling means
        rolling_means = self.history.rolling(window=self.window).mean().shift(1)
        # Return rolling means for new data
        return rolling_means.iloc[start:].values


class WalkForwardOptimizer:

    def __init__(
            self,
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
            spx = False
        ):
        
        self.window_train = window_train
        self.window_val = window_val
        self.window_test = window_test
        self.strategy_parameters = strategy_parameters
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.fixed_fee = fixed_fee
        self.leverage = leverage
        self.horizon = horizon
        self.margin_requirement = margin_requirement
        self.minimum_trades = minimum_trades
        self._load_data(spx)
        self._initialize_dates()
        self.logged_valid_results = []
        self.atm_iv_tracker = ATMIVTracker(tracker_window)
        self.test_data = []

    def _load_data(self, spx = False):
        # Load option chain data
        self.df_option_chain = pd.read_parquet(PATH_SPY_OPTION_CHAIN)
        # Load realized volatility data
        self.df_realized_vol = pd.read_parquet(PATH_SPY_DAILY_VOL)
        # Load treasury rate data
        self.df_treasury_daily_rate = pd.read_parquet(PATH_TREASURY_DAILY_RATE)
        # Simulate SPX prices instead of SPY
        if spx:
            self.df_realized_vol['close'] *= 10
            self.df_option_chain['strike'] *= 10
            self.df_option_chain['forward'] *= 10

    def _initialize_dates(self):
        # First training day available
        first_day = self.df_option_chain.index[0]
        # Last training day available
        self.last_day = self.df_option_chain.index[-1]
        # Beginning training window 
        self.start_train = pd.to_datetime(first_day)
        # End training window
        self.end_train = self.start_train + pd.Timedelta(days = self.window_train)
        # Beginning validation window 
        self.start_val = self.end_train + pd.Timedelta(days = 1)
        # End validation window
        self.end_val = self.start_val + pd.Timedelta(days = self.window_val)
        # Beginning test window 
        self.start_test = self.end_val + pd.Timedelta(days = 1)
        # End test window
        self.end_test = self.start_test + pd.Timedelta(days = self.window_test)
        # First test day
        self.first_test_day = self.start_test

    def _update_dates(self):
        # Update training, validation, and test split dates
        self.start_train = self.start_train + pd.Timedelta(days = self.window_test + 1)
        self.end_train = self.start_train + pd.Timedelta(days = self.window_train)
        self.start_val = self.end_train + pd.Timedelta(days = 1)
        self.end_val = self.start_val + pd.Timedelta(days = self.window_val)
        self.start_test = self.end_val + pd.Timedelta(days = 1)
        self.end_test = self.start_test + pd.Timedelta(days = self.window_test)

    def run_wfo(self):
        # Set random seeds
        seed_everything()
        # Initialize backtester for all test partitions
        self.backtester = Backtester(self.initial_capital, self.slippage_pct, self.fixed_fee, self.leverage, self.horizon, self.margin_requirement)
        
        # Iterate over partitions
        while self.end_test < self.last_day:
            LOGGER.info(f"Testing Partition: {self.start_test.date()} | {self.end_test.date()}")
            # Train, optimize, test
            best_parameters_key, valid_metrics = self._run_wfo_partition()
            # Log validation results and clean
            self._log_results(best_parameters_key, valid_metrics)
            self._clean_and_update()

        # self._plot()
        
        # Retrieve backtesting metrics over all test partitions
        backtest_metrics = self.backtester.get_metrics(self.df_treasury_daily_rate['r_daily'])
        # Retrieve equity curve over all test partitions
        equity_curve, daily_returns = self.backtester.get_equity_curve()

        return backtest_metrics, equity_curve, daily_returns, pd.DataFrame(self.logged_valid_results)

    def _run_wfo_partition(self):
        # Train models on training partition
        self._train_step()
        # Optimize strategy parameter on validation partition
        best_parameters_key, valid_metrics = self._valid_step()
        # Run trading strategy over test partition
        self._test_step(best_parameters_key)
        return best_parameters_key, valid_metrics

    def _train_step(self):
        # Training data
        df_option_chain = self.df_option_chain.loc[:self.end_train]
        df_realized_vol = self.df_realized_vol.loc[:self.end_train]

        # Quantiles for moneyness
        quantiles = np.linspace(0.01, 0.99, M-1)

        # Create spline transformer
        self.spline_transformer = SplineTransformer(df_option_chain['log_moneyness'], quantiles, degree = 3)

        # Get spline transformer attributes
        D = self.spline_transformer.get_num_basis()
        moneyness_grid = self.spline_transformer.get_moneyness_grid().astype(np.float32)
        spline_basis = self.spline_transformer.get_basis().astype(np.float32)

        # Create IV smoother
        self.iv_smoother = IVSmoother(moneyness_grid, target_time_to_expiry)

        # Get IV and ATM IV series
        iv, atm_iv = self.iv_smoother.get_iv(df_option_chain)
        
        # Initialize historical ATM volatility rolling means
        rolling_atm_iv = self.atm_iv_tracker.get_historical_rolling_mean(atm_iv)

        # Train models
        self.dpmm = dpmm_train(iv, spline_basis, D, K)
        self.har = har_train(df_realized_vol, horizon = forecast_horizon)
    
    def _valid_step(self):
        # Validation data (add last training day to generate signal for first validation day)
        adjusted_start = self.start_val - pd.Timedelta(days = 1)
        df_option_chain = self.df_option_chain.loc[adjusted_start:self.end_val]
        df_realized_vol = self.df_realized_vol.loc[adjusted_start:self.end_val]
        
        # Trading days to align data
        common_dates = sorted(
            set(df_option_chain.index.get_level_values('date').unique()) & 
            set(df_realized_vol.index.unique())
        )
        trading_days = pd.to_datetime(common_dates)

        # Get IV and ATM IV series
        iv, atm_iv = self.iv_smoother.get_iv(df_option_chain)

        # Update historical ATM volatility rolling means
        rolling_atm_iv = self.atm_iv_tracker.get_historical_rolling_mean(atm_iv)

        # Forecast regimes
        fear_score = dpmm_forecast(self.dpmm, iv, self.spline_transformer, rolling_atm_iv, df_option_chain.index.unique())

        # Forecast volatility
        vol_forecast = har_forecast(self.har, df_realized_vol)
        
        # Create dataset for strategy
        df_strategy = get_df_strategy(
            df_option_chain, 
            fear_score, 
            atm_iv, 
            vol_forecast, 
            df_realized_vol['close'], 
            trading_days
        )

        # Generate trading signal for every parameter
        trading_signals = self.generate_trading_signals(df_strategy)
        
        # Create dataset for backtesting
        df_backtest = get_df_backtest(df_strategy, self.df_treasury_daily_rate)

        # Optimize parameters
        best_parameters_key, valid_metrics = self._optimize(df_backtest, trading_signals)

        return best_parameters_key, valid_metrics
    
    def _test_step(self, best_parameters_key):
        # Test data (add last validation day to generate signal for first test day)
        adjusted_start = self.start_test - pd.Timedelta(days = 1)
        df_option_chain = self.df_option_chain.loc[adjusted_start:self.end_test]
        df_realized_vol = self.df_realized_vol.loc[adjusted_start:self.end_test]
        
        # Trading days to align data
        common_dates = sorted(
            set(df_option_chain.index.get_level_values('date').unique()) & 
            set(df_realized_vol.index.unique())
        )
        trading_days = pd.to_datetime(common_dates)

        # Get IV and ATM IV series
        iv, atm_iv = self.iv_smoother.get_iv(df_option_chain)

        # Update historical ATM volatility rolling means
        rolling_atm_iv = self.atm_iv_tracker.get_historical_rolling_mean(atm_iv)

        # Forecast regimes
        fear_score = dpmm_forecast(self.dpmm, iv, self.spline_transformer, rolling_atm_iv, df_option_chain.index.unique())

        # Forecast volatility
        vol_forecast = har_forecast(self.har, df_realized_vol)
        
        # Create dataset for strategy
        df_strategy = get_df_strategy(
            df_option_chain, 
            fear_score, 
            atm_iv, 
            vol_forecast, 
            df_realized_vol['close'], 
            trading_days
        )

        # Generate trading signals
        trading_signals = self.generate_trading_signals(df_strategy, best_parameters_key)
        
        # Create dataset for backtesting
        df_backtest = get_df_backtest(df_strategy, self.df_treasury_daily_rate)

        # Run and update backtester
        self.backtester.run(df_backtest, trading_signals)

        # Accumulate time series data
        self._accumulate(df_realized_vol, df_strategy, trading_signals)

    def generate_trading_signals(self, df_strategy, best_parameters_key = None):

        def get_signals(df_strategy, edge_threshold, fear_threshold):
            # Initialize signal
            df_strategy['signal'] = 0.0
            # Low fear score
            mask_safe = df_strategy['fear_score'] < fear_threshold
            # High edge
            mask_value = df_strategy['edge'] > edge_threshold
            # Price not decreasing (momentum)
            mask_trend = df_strategy['close'] >= df_strategy['close'].ewm(span=forecast_horizon//2, adjust=False).mean()
            # Trading signal
            mask_signal = mask_safe & mask_value & mask_trend
            df_strategy.loc[mask_signal, 'signal'] = 1.0
            # Shift to avoid lookahead bias
            return df_strategy['signal'].shift(1)

        if best_parameters_key is not None:
            # Retrive optimal parameters
            edge_threshold, fear_threshold = self.strategy_parameters[best_parameters_key]
            trading_signals = get_signals(df_strategy, edge_threshold, fear_threshold)
        else:
            # Try every parameter combination
            trading_signals = {}
            for key, (edge_threshold, fear_threshold) in self.strategy_parameters.items():
                trading_signals[key] = get_signals(df_strategy, edge_threshold, fear_threshold)
        
        return trading_signals

    def _optimize(self, df_backtest, trading_signals, metric_to_optimize = 'sharpe_ratio'):

        backtest_metrics = {}

        # Run backtest and collect metrics
        for key in trading_signals.keys():
            backtester = Backtester(self.initial_capital, self.slippage_pct, self.fixed_fee, self.leverage, self.horizon, self.margin_requirement)
            backtester.run(df_backtest, trading_signals[key], liquidate = True)
            backtest_metrics[key] = backtester.get_metrics(self.df_treasury_daily_rate['r_daily'])

        # Choose best parameters with a minimum number of trades
        best_metric = -np.inf
        best_parameters_key = None
        for key, metrics in backtest_metrics.items():
            if (metrics['trade_count'] >= self.minimum_trades) and (metrics[metric_to_optimize] > best_metric):
                best_metric = metrics[metric_to_optimize]
                best_parameters_key = key

        # Choose best parameters if no combination yields the minimum number of trades
        if best_parameters_key is None:
            for key, metrics in backtest_metrics.items():
                if metrics[metric_to_optimize] > best_metric:
                    best_metric = metrics[metric_to_optimize]
                    best_parameters_key = key

        return best_parameters_key, backtest_metrics[best_parameters_key]

    def _clean_and_update(self):
        self._update_dates()
        self.spline_transformer = None
        self.iv_smoother = None
        self.dpmm = None
        self.har = None

    def _log_results(self, best_parameters_key, valid_metrics):
        result_dict = {}
        
        # Log dates
        result_dict['start_train'] = self.start_train
        result_dict['start_val'] = self.start_val
        result_dict['start_test'] = self.start_test
        result_dict['end_test'] = self.end_test

        # Log parameters
        edge_threshold, fear_threshold = self.strategy_parameters[best_parameters_key] 
        result_dict['edge_threshold'] = edge_threshold
        result_dict['fear_threshold'] = fear_threshold

        # Log backtest metrics
        result_dict['total_return'] = valid_metrics['total_return']
        result_dict['max_drawdown'] = valid_metrics['max_drawdown']
        result_dict['sharpe_ratio'] = valid_metrics['sharpe_ratio']
        result_dict['sortino_ratio'] = valid_metrics['sortino_ratio']
        result_dict['turnover'] = valid_metrics['turnover']
        result_dict['trade_count'] = valid_metrics['trade_count']

        self.logged_valid_results.append(result_dict)
    
    def _accumulate(self, df_realized_vol, df_strategy, trading_signals):
        # Indices to align data
        valid_indices = df_strategy.index

        # Align data
        part1 = df_realized_vol.loc[valid_indices, ['close', 'vol']]
        part2 = df_strategy.loc[valid_indices, ['atm_iv', 'vol_forecast', 'edge', 'fear_score']]
        part3 = trading_signals.loc[valid_indices]
        part4 = self.df_treasury_daily_rate.loc[valid_indices, 'r_daily']

        # Concatenate columns
        new_data = pd.concat([part1, part2, part3, part4], axis=1)

        # Store test partition data
        self.test_data.append(new_data)

    def _plot(self):

        df_test_data = pd.concat(self.test_data, axis=0).sort_index()
        df_test_data = df_test_data[~df_test_data.index.duplicated(keep='last')].sort_index()

        # Plot volatility
        realized_21d_future = (
            df_test_data['vol']
            .rolling(window=21)
            .mean()
            .shift(-21)
        )
        realized_21d_future.name = 'vol'
        plot_volatility(
            df_test_data['vol_forecast'],
            df_test_data['atm_iv'],
            realized_21d_future,
            path = PATH_PLOT_VOLATILITY
        )

        # Plot price
        plot_price(
            df_test_data['close'],
            path = PATH_PLOT_PRICE
        )

        # Plot fear and edge
        plot_signals(
            df_test_data['edge'],
            df_test_data['fear_score'],
            path = PATH_PLOT_SIGNALS
        )

    def get_test_data(self):
        df_test_data = pd.concat(self.test_data, axis=0).sort_index()
        df_test_data = df_test_data[~df_test_data.index.duplicated(keep='last')].sort_index()
        return df_test_data