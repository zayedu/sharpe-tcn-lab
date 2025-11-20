import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional

def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads OHLCV data from Yahoo Finance.
    
    Args:
        ticker: Ticker symbol (e.g., "SPY").
        start_date: Start date in "YYYY-MM-DD" format.
        end_date: End date in "YYYY-MM-DD" format.
        
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume.
        Index is DatetimeIndex.
    """
    print(f"Downloading {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
        
    # Handle MultiIndex columns if present (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Keep only required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols].copy()
    
    # Forward fill missing values then drop remaining NaNs
    df = df.ffill().dropna()
    
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    
    print(f"Downloaded {len(df)} rows.")
    return df

if __name__ == "__main__":
    # Simple test
    try:
        data = download_data("SPY", "2020-01-01", "2020-01-10")
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")
