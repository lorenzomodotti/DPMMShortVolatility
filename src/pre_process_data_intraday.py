import glob
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from src.logger import LOGGER
from src.config import *

def print_stats(ddf):
    
    total_rows = len(ddf)
    LOGGER.info(f"Total 1-minute intervals: {total_rows:,}")
    
    start_date = ddf.index.min().compute()
    end_date = ddf.index.max().compute()
    LOGGER.info(f"Coverage: {start_date} to {end_date}")
    
    nulls = ddf.isnull().sum().compute()
    LOGGER.info("\nMissing Values per Column:")
    LOGGER.info(nulls)
    
    dupes = (ddf.index.value_counts() > 1).sum().compute()
    LOGGER.info(f"\nDuplicate timestamps found: {dupes}")

    LOGGER.info(f"Number of partitions: {ddf.npartitions}")


def process_1m_intraday_data(path):

    LOGGER.info("Processing 1-minute intraday data")

    # Load 1-minute intraday data
    df_1m = dd.read_csv(
        path,
        parse_dates = ['timestamp']
    )

    # Align timestamps
    df_1m['timestamp'] = df_1m['timestamp'] + pd.Timedelta(hours=2, minutes=1)
    df_1m = df_1m.set_index('timestamp', sorted=True)

    # Filter data to avoid overlaps and drop volume
    df_1m = df_1m.loc[:'2020-02-23'].drop(columns=['volume'])

    return df_1m


def process_3s_intraday_data(path):

    LOGGER.info("Processing 3-second intraday data")

    files = glob.glob(path)
    files.sort()

    # Load 3-second intraday data
    df_3s = dd.read_csv(
        files,
        usecols=['TimeStamp', 'SPY'],
        parse_dates=['TimeStamp'],
        date_format='%Y-%m-%d %H:%M:%S.%f'
    )
    df_3s.columns = ['timestamp', 'price']
    df_3s = df_3s.set_index('timestamp', sorted=True)

    # Filter data to keep only trading hours
    df_3s = df_3s.map_partitions(
        lambda partition: partition.between_time('09:30:00', '16:00:00', inclusive='right')
    )

    # Resample to 1-minute with standard convention for time intervals
    df_3s_resampled = df_3s.map_partitions(
        lambda partition: partition['price'].resample('1min', label='right', closed='right').ohlc()
    )
    
    # Remove NaNs introduced by resample to fill the timeline
    return df_3s_resampled.dropna()


if __name__ == "__main__":

    # Load and process 1-minute intraday data from 2008-01-22 to 2021-05-06
    path_1m_data = "raw_data/spy_1m_2008_2021.csv"
    
    df_1m = process_1m_intraday_data(path_1m_data)

    # Load and process 3-second intraday data from 2020-01-27 to 2026-04-03
    path_3s_data = "raw_data/market_data_intraday_3s/data_*/week_*.csv"
    df_3s_resampled = process_3s_intraday_data(path_3s_data)

    # Stack data
    ddf = dd.concat([df_1m, df_3s_resampled])
    ddf = ddf.set_index(ddf.index, sorted=True)

    print_stats(ddf)

    # Write to partitioned parquet file
    with ProgressBar():
        ddf.to_parquet(
            PATH_SPY_INTRADAY_OHLC,
            engine='pyarrow',
            compression='snappy',
            write_index=True
        )