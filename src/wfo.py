import warnings
warnings.filterwarnings("ignore")

import random
import os
import pandas as pd
import numpy as np
from src.data_transformer import SplineTransformer, IVSmoother, VolatilitySmileDataset
from src.dpmm import dpmm_train, dpmm_forecast
from src.har import har_train, har_forecast
from src.backtester import *
from src.logger import LOGGER
from src.outputs import *
from src.config import *

# Number of moneyness levels (quantiles)
M = 16

# Quantiles for moneyness
quantiles = np.linspace(0.01, 0.99, M-1)

# Number of regimes to identify
K = 2

# Target expiry
target_time_to_expiry = 30

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
    normalized_distances = top_distances / np.sum(top_distances)
    
    # Compute spread ratios
    top_spread_ratios = ((options['ask'] - options['bid']) / options['mid']).iloc[top_k_indices].values
    # Compute forward
    top_forward = options['forward'].iloc[top_k_indices].values

    return np.inner(top_spread_ratios, normalized_distances), np.inner(top_forward, normalized_distances)


def get_df_strategy(df_option_chain_test, df_regimes, atm_iv_test, har_vol_forecast, dates_test):
    
    # Group per trading day
    grouped = df_option_chain_test.groupby('date')
    
    # Initialize arrays
    spread_ratios = np.zeros(len(grouped))
    forwards = np.zeros(len(grouped))
    trading_days = []
    
    # Compute spread ratios, forwards per trading day
    for t, (date, group) in enumerate(grouped):
        sr, f = _get_spread_ratio(group, target_time_to_expiry)
        spread_ratios[t] = sr
        forwards[t] = f
        trading_days.append(date)

    # Spread ratio series
    spread_ratios = pd.Series(
        spread_ratios,
        index = trading_days,
        name = 'spread_ratio'
    )

    # Forward series
    forwards = pd.Series(
        forwards,
        index = trading_days,
        name = 'forward'
    )

    # Concatenate and align data
    df_strategy = pd.concat(
        [
            df_regimes.loc[dates_test],
            atm_iv_test.loc[dates_test],
            har_vol_forecast.loc[dates_test],
            spread_ratios.loc[dates_test],
            forwards.loc[dates_test],
        ],
        axis=1
    ).dropna()

    # Compute edge
    df_strategy['edge'] = df_strategy['atm_iv'] - df_strategy['vol_forecast'] - (df_strategy['spread_ratio'] * df_strategy['atm_iv'])

    return df_strategy


def get_df_backtest(df_strategy, spot_price, df_treasury, trading_days):
    
    return pd.concat(
        [
            df_strategy['signal'],
            df_strategy['atm_iv'],
            spot_price.loc[trading_days, 'close'],  
            df_strategy['forward'],
            df_strategy['spread_ratio'],
            df_treasury.loc[trading_days, 'r_annual'],
            df_treasury.loc[trading_days, 'r_daily']
        ],
        axis=1).dropna().reset_index().rename(columns={'index': 'date'})

class WalkForwardOptimizer:

    def __init__(self, window_train, window_val, window_test, strategy_parameters, initial_capital, slippage_pct, fixed_fee, leverage, horizon, minimum_trades):
        self.window_train = window_train
        self.window_val = window_val
        self.window_test = window_test
        self.strategy_parameters = strategy_parameters
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.fixed_fee = fixed_fee
        self.leverage = leverage
        self.horizon = horizon
        self.minimum_trades = minimum_trades
        self._load_data()
        self._initialize_dates()
        self.logged_valid_results = []

    def _load_data(self):
        # Load option chain data
        self.df_option_chain = pd.read_parquet(PATH_SPY_OPTION_CHAIN)
        # Load realized volatility data
        self.df_realized_vol = pd.read_parquet(PATH_SPY_DAILY_VOL)
        # Load treasury rate data
        self.df_treasury_daily_rate = pd.read_parquet(PATH_TREASURY_DAILY_RATE)

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

    def _train_step(self):
        # Training data
        df_option_chain = self.df_option_chain.loc[:self.end_train]
        df_realized_vol = self.df_realized_vol.loc[:self.end_train]

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

        # Train models
        self.dpmm = dpmm_train(iv, spline_basis, D, K)
        self.har = har_train(df_realized_vol, horizon = self.horizon)
    
    def _valid_step(self):
        # Validation data
        df_option_chain = self.df_option_chain.loc[self.start_val:self.end_val]
        df_realized_vol = self.df_realized_vol.loc[self.start_val:self.end_val]
        
        # Trading days to align data
        trading_days = pd.to_datetime(
            np.intersect1d(
                df_option_chain.index.unique().date, 
                df_realized_vol.index.unique().date
            )
        )

        # Get IV and ATM IV series
        iv, atm_iv = self.iv_smoother.get_iv(df_option_chain)

        # Forecast regimes
        df_regimes, index_panic_cluster = dpmm_forecast(self.dpmm, iv, self.spline_transformer, df_option_chain.index.unique())

        # Forecast volatility
        vol_forecast = har_forecast(self.har, df_realized_vol)
        
        # Create dataset for strategy
        df_strategy = get_df_strategy(df_option_chain, df_regimes, atm_iv, vol_forecast, trading_days)

        # Generate trading signal for every parameter
        trading_signals = self.generate_trading_signals(df_strategy, index_panic_cluster)
        
        # Create dataset for backtesting
        df_backtest = get_df_backtest(df_strategy, df_realized_vol, self.df_treasury_daily_rate, trading_days)

        best_parameters_key, valid_metrics = self._optimize(df_backtest, trading_signals)

        return best_parameters_key, valid_metrics

    def generate_trading_signals(self, df_strategy, index_panic_cluster, best_parameters_key = None):

        if best_parameters_key is not None:
            # Retrive optimal parameters
            edge_threshold, prob_threshold = self.strategy_parameters[best_parameters_key]
            # Initialize signal
            df_strategy['signal'] = 0.0
            # High probability of being in normal regime
            mask_safe = df_strategy.iloc[:,index_panic_cluster] > (prob_threshold)
            # High edge
            mask_value = df_strategy['edge'] > edge_threshold
            # Trading signal
            df_strategy.loc[mask_safe & mask_value, 'signal'] = 1.0
            trading_signals = df_strategy['signal'].copy()
        else:
            trading_signals = {}
            for key, (edge_threshold, prob_threshold) in self.strategy_parameters.items():
                # Initialize signal
                df_strategy['signal'] = 0.0
                # High probability of being in normal regime
                mask_safe = df_strategy.iloc[:,index_panic_cluster] > (prob_threshold)
                # High edge
                mask_value = df_strategy['edge'] > edge_threshold
                # Trading signal
                df_strategy.loc[mask_safe & mask_value, 'signal'] = 1.0
                trading_signals[key] = df_strategy['signal'].copy()
        
        return trading_signals

    def _optimize(self, df_backtest, trading_signals):

        backtest_metrics = {}

        # Run backtest and collect metrics
        for key in trading_signals.keys():
            backtester = Backtester(self.initial_capital, self.slippage_pct, self.fixed_fee, self.leverage, self.horizon)
            backtester.run(df_backtest, trading_signals[key], liquidate = True)
            backtest_metrics[key] = backtester.get_metrics(self.df_treasury_daily_rate['r_daily'])

        # Choose best parameters
        best_sortino = -np.inf
        best_parameters_key = None
        for key, metrics in backtest_metrics.items():
            if (metrics['trade_count'] >= self.minimum_trades) and (metrics['sortino_ratio'] > best_sortino):
                best_sortino = metrics['sortino_ratio']
                best_parameters_key = key

        if best_parameters_key is None:
            for key, metrics in backtest_metrics.items():
                if metrics['sortino_ratio'] > best_sortino:
                    best_sortino = metrics['sortino_ratio']
                    best_parameters_key = key

        return best_parameters_key, backtest_metrics[best_parameters_key]

    def _test_step(self, best_parameters_key):
        # Validation data
        df_option_chain = self.df_option_chain.loc[self.start_test:self.end_test]
        df_realized_vol = self.df_realized_vol.loc[self.start_test:self.end_test]
        
        # Trading days to align data
        trading_days = pd.to_datetime(
            np.intersect1d(
                df_option_chain.index.unique().date, 
                df_realized_vol.index.unique().date
            )
        )

        # Get IV and ATM IV series
        iv, atm_iv = self.iv_smoother.get_iv(df_option_chain)

        # Forecast regimes
        df_regimes, index_panic_cluster = dpmm_forecast(self.dpmm, iv, self.spline_transformer, df_option_chain.index.unique())

        # Forecast volatility
        vol_forecast = har_forecast(self.har, df_realized_vol)
        
        # Create dataset for strategy
        df_strategy = get_df_strategy(df_option_chain, df_regimes, atm_iv, vol_forecast, trading_days)

        # Generate trading signals
        trading_signals = self.generate_trading_signals(df_strategy, index_panic_cluster, best_parameters_key)
        
        # Create dataset for backtesting
        df_backtest = get_df_backtest(df_strategy, df_realized_vol, self.df_treasury_daily_rate, trading_days)

        # Update backtester
        self.backtester.run(df_backtest, trading_signals)
        
    
    def _clean_and_update(self):
        self._update_dates()
        self.spline_transformer = None
        self.iv_smoother = None
        self.dpmm = None
        self.har = None

    def _log_results(self, best_parameters_key, test_metrics):
        result_dict = {}
        
        # Log dates
        result_dict['start_train'] = self.start_train
        result_dict['start_val'] = self.start_val
        result_dict['start_test'] = self.start_test
        result_dict['end_test'] = self.end_test

        # Log parameters
        edge_threshold, prob_threshold = self.strategy_parameters[best_parameters_key] 
        result_dict['edge_threshold'] = edge_threshold
        result_dict['prob_threshold'] = prob_threshold

        # Log backtest metrics
        result_dict['total_return'] = test_metrics['total_return']
        result_dict['max_drawdown'] = test_metrics['max_drawdown']
        result_dict['sharpe_ratio'] = test_metrics['sharpe_ratio']
        result_dict['sortino_ratio'] = test_metrics['sortino_ratio']
        result_dict['turnover'] = test_metrics['turnover']
        result_dict['trade_count'] = test_metrics['trade_count']

        self.logged_valid_results.append(result_dict)
    
    def _run_wfo_partition(self):
        #LOGGER.info(f"Starting Training: {self.start_train} | {self.end_train}")
        self._train_step()
        #LOGGER.info(f"Starting Validation: {self.start_val} | {self.end_val}")
        best_parameters_key, valid_metrics = self._valid_step()
        #LOGGER.info(f"Starting Backtest: {self.start_test} | {self.end_test}")
        self._test_step(best_parameters_key)
        return best_parameters_key, valid_metrics

    def run_wfo(self):
        seed_everything()
        self.backtester = Backtester(self.initial_capital, self.slippage_pct, self.fixed_fee, self.leverage, self.horizon)
        while self.end_test < self.last_day:
            LOGGER.info(f"Starting Partition: {self.start_test.date()} | {self.end_test.date()}")
            best_parameters_key, valid_metrics = self._run_wfo_partition()
            self._log_results(best_parameters_key, valid_metrics)
            self._clean_and_update()
        
        backtest_metrics = self.backtester.get_metrics(self.df_treasury_daily_rate['r_daily'])
        equity_curve = self.backtester.get_equity_curve()

        return backtest_metrics, equity_curve, pd.DataFrame(self.logged_valid_results)