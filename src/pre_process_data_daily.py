import pandas as pd
import numpy as np
import dask.dataframe as dd
from src.pricing import *
from src.config import *
from src.logger import LOGGER

def process_treasury_rates():
    """
    Ingest daily treasury rates data and compute annual continuously compounded rate and daily rate
    """

    # Load US treasury rate data
    df = pd.read_csv(
        "raw_data/us_treasury_rates.csv",
        usecols=['date', '1_month'],
        parse_dates=['date'],
        date_format='%Y-%m-%d'
    )
    df.columns = ['date', 'rate_1_month']
    df = df.set_index('date')

    # Annual continuously compounded rate
    df['r_annual'] = df.eval("log(1 + rate_1_month/100)")

    # Daily rate
    df['r_daily'] = df.eval("(1 + rate_1_month/100)**(1/252) - 1")
    
    df.to_parquet(PATH_TREASURY_DAILY_RATE)

    LOGGER.info(f"US Treasury rate data processed. Coverage: {df.index.min()} to {df.index.max()}. Trading days: {len(df)}")


def process_option_chain():

    LOGGER.info("Processing SPY option chain data")

    # Load underlying SPY OHLCV daily data
    df_underlying_price = pd.read_csv(
        "raw_data/spy_ohlcv.csv",
        usecols=['date','open','high','low','close'],
        parse_dates=['date'],
        date_format='%Y-%m-%d'
    )
    df_underlying_price = df_underlying_price.set_index('date')
    
    # Load underlying SPY dividend data
    df_underlying_dividend = pd.read_csv(
        "raw_data/spy_dividend.csv",
        usecols=['ex_date', 'amount'],
        parse_dates=['ex_date'],
        date_format='%Y-%m-%d'
    )
    
    df_risk_free_rate = pd.read_parquet(PATH_TREASURY_DAILY_RATE)

    # Load SPY option chain data
    df_option_chain = pd.read_csv(
        "raw_data/spy_option_chain.csv",
        usecols=['date','expiration','strike','call_put','bid','ask'],
        parse_dates=['date', 'expiration'],
        date_format='%Y-%m-%d'
    )

    # Remove illiquid options
    df_option_chain = df_option_chain.query("bid > 0 and ask >= bid")

    # Transform Saturdays to Fridays
    df_option_chain['date'] = df_option_chain['date'].where(df_option_chain['date'].dt.weekday != 5, df_option_chain['date'] - pd.Timedelta(days=1))

    # Add spot price
    df = df_option_chain.merge(df_underlying_price['close'], how='inner', left_on='date', right_index=True)

    # Add annual continuously compounded risk-free rate
    df['risk_free_rate'] = df['date'].map(df_risk_free_rate['r_annual'])

    # Add dividend date and amount
    df = pd.merge_asof(
        df, 
        df_underlying_dividend, 
        left_on='date', 
        right_on='ex_date', 
        direction='forward'
    )

    # Mask when to adjust price for dividends (date < dividend date <= expiration)
    mask_dividend = (df['ex_date'] > df['date']) & (df['ex_date'] <= df['expiration'])

    # Compute time to dividend
    df['time_to_dividend'] = np.where(mask_dividend, (df['ex_date'] - df['date']).dt.days, 0) / 365.0

    # Compute dividend amount
    df['amount'] = np.where(mask_dividend, df['amount'], 0)

    # Compute time to expiry
    df['time_to_expiry'] = (df['expiration'] - df['date']).dt.days / 365.0

    # Comput forward price: F = (S - PV_div) * exp(r * T)
    df['forward'] = df.eval("(close - amount * exp(- risk_free_rate * time_to_dividend)) * exp(risk_free_rate * time_to_expiry)")

    # Compute log-moneyness of options
    df['log_moneyness'] = df.eval("log(forward / strike)")

    # Keep Out-of-the-Money (OTM) and At-the-Money (ATM) options
    df = df.query("(call_put == 'Call' and log_moneyness <= 0) or (call_put == 'Put' and log_moneyness >= 0)")

    # Compute option mid price
    df['mid'] = df.eval("(ask+bid)/2")

    # Compute implied volatility
    LOGGER.info("Computing Implied Volatility inverting Black-Scholes formula")
    df['iv'] = df.apply(lambda row: bs_iv(row), axis=1)

    # Keep only relevant columns
    df = df[['date', 'expiration', 'time_to_expiry', 'strike', 'call_put', 'bid', 'ask', 'mid', 'forward', 'log_moneyness', 'iv']]

    df = df.set_index('date')

    df.to_parquet(PATH_SPY_OPTION_CHAIN)

    LOGGER.info(f"SPY option chain data processed. Coverage: {df.index.min()} to {df.index.max()}. Number of options: {len(df)}. Number of trading days: {df.index.nunique()}.")


def compute_daily_realized_volatility():

    # Load 1-minute intraday data from partitioned parquet
    ddf = dd.read_parquet(PATH_SPY_INTRADAY_OHLC, columns=['close'], calculate_divisions=True)

    # Load dividend data
    dividends = pd.read_csv(
        PATH_SPY_DIVIDED,
        parse_dates=['ex_date'],
        usecols=['ex_date','amount']
    ).set_index('ex_date')

    # Intraday Variance

    # Resample to 5-minutes to avoid bid-ask bounce
    df_5min = ddf['close'].resample('5min').median().compute()

    # Compute intraday variance
    intraday_var = np.log(df_5min).diff().pow(2).resample('D').sum()

    # Overnight Variance

    # Compute open and close for each day
    df_daily = ddf['close'].resample('D').ohlc().compute() 

    # Remove weekends introduced by resampling
    mask_weekends = df_daily['close'].isna()
    intraday_var = intraday_var[~mask_weekends]
    df_daily = df_daily[~mask_weekends]

    # Merge dividends to daily data
    df_daily = df_daily.merge(dividends, how='left', left_index=True, right_index=True)

    # Fill NaNs (no dividend on that day) with 0.0
    df_daily['amount'] = df_daily['amount'].fillna(0.0)

    # Compute overnight variance
    overnight_var = np.log((df_daily['open'] + df_daily['amount']) / df_daily['close'].shift(1)).pow(2)

    # Total Variance
    df = (intraday_var + overnight_var).to_frame()
    df.columns = ['var']

    # Add close price
    df['close'] = df.index.map(df_daily['close'])

    # Impute corrupted data
    mask_corrupted_data = df['var'] < 3.5e-09
    imputed_var = df['var'].mask(mask_corrupted_data).rolling(window=5, min_periods=1).mean().shift(1)
    df['var'] = df['var'].where(~mask_corrupted_data, imputed_var).ffill()

    # Annualized Volatility
    df['vol'] = df.eval("sqrt(var * 252)")

    df = df.dropna()
    df.to_parquet(PATH_SPY_DAILY_VOL)

    LOGGER.info(f"SPY daily realized volatility computed. Coverage: {df.index.min()} to {df.index.max()}. Number of trading days: {len(df)}.")


if __name__ == "__main__":

    process_treasury_rates()

    process_option_chain()

    compute_daily_realized_volatility()