import mysql.connector
import pandas as pd
from src.pricing import *
from src.utils import LOGGER
from src.config import PATH_OPTION_CHAIN_QUERY, PATH_ASSET_QUERY, PATH_INTRADAY_SPY_DATA, PATH_RISK_FREE_RATE_DATA

def get_mysql_connection():
    """
    Get connection to MySQL server
    """
    mysql_connection = None
    try:
        mysql_connection = mysql.connector.connect(
            host="127.0.0.1",
            port=3306,
            user="root",
            password=""
        )
        LOGGER.info("Connected to MySQL server")
    except Exception as e:
        LOGGER.error(f"Unable to connect to MySQL server: {e}", exc_info=True)
        mysql_connection = None
    return mysql_connection

def read_query_from_file(file_path):
    """
    Read SQL query from file
    """
    try:
        with open(file_path, 'r') as file:
            query_string = file.read()
            return query_string
    except FileNotFoundError:
        LOGGER.error(f"Unable to find SQL file at: {file_path}")
        return []

def query_mysql_db(connection, query, params = None, name = ""):
    """
    Query MySQL database
    """
    df = []
    try:
        df = pd.read_sql(query, connection, params=params)
        LOGGER.info(f"Retrieved {name} data from MySQL server")
    except Exception as e:
        LOGGER.error(f"Unable to retrieve {name} data from MySQL server: {e}", exc_info=True)
        df = []
    return df

def ingest_options_data(ticker = 'SPY', r = 0.02):
    """
    Retrieve and transform option chain data for a specific asset
    """
    # Connect to MySQL server
    mysql_connection = get_mysql_connection()
    # Retrieve query
    query = read_query_from_file(PATH_OPTION_CHAIN_QUERY)
    # Get options data
    df = query_mysql_db(
        mysql_connection,
        query,
        params = {'ticker': ticker},
        name = "daily SPY option chain"
        )
    # Convert date and expiration to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['expiration'] = pd.to_datetime(df['expiration'], format='%Y-%m-%d')
    # Comput time to expiry
    df['time_to_expiry'] = (df['expiration'] - df['date']).dt.days
    # Compute moneyness of options
    df['moneyness'] = df.eval("strike / underlying_price")

    # Keep Out-of-the-Money (OTM) options
    df_otm = (
        df
        .query("(call_put == 'Call' and moneyness > 1) or (call_put == 'Put' and moneyness < 1)")
        .drop(columns=['act_symbol'])
    )
    # Keep At-of-the-Money (ATM) options with mean volatility
    df_atm = (
        df
        .query("(call_put == 'Call' and moneyness == 1) or (call_put == 'Put' and moneyness == 1)")
        .groupby(['date', 'act_symbol', 'call_put', 'underlying_price', 'option_price', 'expiration', 'time_to_expiry', 'strike'])
        .agg(vol = ('vol', 'mean'))
        .reset_index()
        .drop(columns=['act_symbol'])
    )
    df_atm['moneyness'] = 1.0
    df_concat = pd.concat([df_otm, df_atm])
    # Compute log-moneyness
    df_concat['log_moneyness'] = np.log(df_concat['moneyness'])

    # If series of risk free rate is provided match dates
    if hasattr(r, "__len__"):
        # Explicitly merge and KEEP the result
        df_concat = pd.merge(df_concat, r, how='inner', on='date')
        # We don't need a separate 'r' variable anymore, it's in the columns
        r_arg = None # Signal to use internal column
    else:
        # Constant rate
        r_arg = r
    # Validate market price data
    df_concat = validate_pricing_data(df_concat, r=r_arg)
    # Recompute IV for non valid market data
    df_fix = update_iv(df_concat, error_tolerance=2.0, r_fallback=r_arg)
    # Remove outliers
    return df_fix.query('vol < 1').sort_values(by='date')

def ingest_daily_prices(ticker = 'SPY'):
    """
    Retrieve and transform daily price data for a specific asset
    """
    # Connect to MySQL server
    mysql_connection = get_mysql_connection()
    # Retrieve query
    query = read_query_from_file(PATH_ASSET_QUERY)
    # Get options data
    df = query_mysql_db(
        mysql_connection,
        query,
        params = {'ticker': ticker},
        name = "daily SPY price"
        )
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    # Set date as index
    df.set_index('date', inplace=True)
    # Compute returns
    df['return'] = df['close'].pct_change()
    # Compute log-returns
    df['log_return'] = np.log(df['close']).diff()
    # Compute Parkinson volatility
    df['parkinson_vol'] = df.eval("sqrt(252 / (4*log(2)) * (log(high/low))**2)")
    return df.sort_values(by='date')

def query_local_data(path, name = ""):
    """
    Query local CSV dataframe
    """
    try:
        df = pd.read_csv(path)
        LOGGER.info(f"Retrieved {name} data")
    except Exception as e:
        LOGGER.error(f"Unable to retrieve {name} data: {e}", exc_info=True)
        df = []
    return df

def ingest_intraday_data(ticker = 'SPY'):
    """
    Retrieve and transform intraday price data for a specific asset
    """
    # Get intraday data
    df = query_local_data(PATH_INTRADAY_SPY_DATA, "intraday SPY price")
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Set date as index
    df.set_index('date', inplace=True)
    # Subsample to remove microstructure noise 
    df_sub = df.resample('5min').agg(
        {
            'open':  'first',  # The open is the first open of the period
            'high':  'max',    # The high is the maximum of the highs
            'low':   'min',    # The low is the minimum of the lows
            'close': 'last',   # The close is the last close
            'volume': 'sum',   # Total volume
            'barCount': 'sum'  # Total bar count
        }
    )
    # Remove NAs introduced by subsampling (no trading days)
    df_sub.dropna(inplace=True)
    # Compute log-returns
    df_sub['log_return'] = np.log(df_sub['close']).diff()
    return df_sub.sort_values(by='date')

def ingest_risk_free_rate_data(ticker = 'DTB4WK'):
    """
    Retrieve and transform risk free rate data
    """
    # Get intraday data
    df = query_local_data(PATH_RISK_FREE_RATE_DATA, "risk free rate")
    df.columns = ['date', 'percentage_rate']
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    # Fill NAs
    df.ffill(inplace=True)
    # Compute risk free rate
    df['r'] = df.eval("log(1 + percentage_rate/100)")
    return df