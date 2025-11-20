import numpy as np
import pandas as pd
from typing import Tuple

class FeatureEngineer:
    def __init__(self, window_size: int = 64):
        self.window_size = window_size
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates technical features from OHLCV data.
        
        Features:
        1. Log Returns
        2. Volatility (20-day rolling std of log returns)
        3. RSI (14-day)
        4. MACD (12, 26, 9) - MACD line
        5. MACD Signal
        6. MACD Hist
        7. VWAP Distance (Close - VWAP) / Close
        8. Volume Z-Score (20-day)
        9. High-Low Range (Log)
        10. Close-Open Range (Log)
        11. Momentum (10-day)
        12. Momentum (5-day)
        """
        df = df.copy()
        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']
        open_ = df['Open']
        
        # 1. Log Returns
        df['log_ret'] = np.log(close / close.shift(1))
        
        # 2. Volatility (20-day)
        df['volatility'] = df['log_ret'].rolling(window=20).std()
        
        # 3. RSI (14-day)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 4-6. MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 7. VWAP Distance
        vwap = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['vwap_dist'] = (close - vwap) / close
        
        # 8. Volume Z-Score
        vol_mean = volume.rolling(window=20).mean()
        vol_std = volume.rolling(window=20).std()
        df['vol_z'] = (volume - vol_mean) / vol_std
        
        # 9. High-Low Range
        df['hl_range'] = np.log(high / low)
        
        # 10. Close-Open Range
        df['co_range'] = np.log(close / open_)
        
        # 11. Momentum 10
        df['mom10'] = close / close.shift(10) - 1
        
        # 12. Momentum 5
        df['mom5'] = close / close.shift(5) - 1
        
        # 13. ATR (14-day)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # 14. ADX (14-day)
        up = high - high.shift(1)
        down = low.shift(1) - low
        pos_dm = np.where((up > down) & (up > 0), up, 0.0)
        neg_dm = np.where((down > up) & (down > 0), down, 0.0)
        pos_dm = pd.Series(pos_dm, index=df.index).rolling(window=14).mean()
        neg_dm = pd.Series(neg_dm, index=df.index).rolling(window=14).mean()
        # Avoid division by zero
        atr = df['atr'].replace(0, 1e-8)
        pos_di = 100 * pos_dm / atr
        neg_di = 100 * neg_dm / atr
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-8)
        df['adx'] = dx.rolling(window=14).mean()

        # Drop NaNs created by rolling windows
        df = df.dropna()
        
        # Select only feature columns
        feature_cols = [
            'log_ret', 'volatility', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'vwap_dist', 'vol_z', 'hl_range', 'co_range', 'mom10', 'mom5',
            'atr', 'adx'
        ]
        
        return df[feature_cols], df

    def create_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Creates dataset for TCN with Sharpe Loss.
        Returns:
        X: (N, window_size, num_features)
        y: (N,) - Binary target (kept for reference/metrics)
        returns: (N,) - Next day return (for Sharpe Loss)
        volatility: (N,) - Current volatility (for Position Sizing)
        timestamps: (N,)
        """
        features_df, full_df = self.create_features(df)
        
        # Targets and Returns
        # At index i (time t), we want to predict return at t+1
        # returns[i] should be return at t+1
        next_ret = features_df['log_ret'].shift(-1)
        targets = (next_ret > 0).astype(int)
        
        # Align
        valid_indices = features_df.index[:-1]
        features_df = features_df.loc[valid_indices]
        targets = targets.loc[valid_indices]
        next_ret = next_ret.loc[valid_indices]
        
        data = features_df.values
        timestamps = features_df.index
        returns_arr = next_ret.values
        vol_arr = features_df['volatility'].values
        
        X_list = []
        y_list = []
        ret_list = []
        vol_list = []
        valid_timestamps = []
        
        for i in range(self.window_size, len(data)):
            # Window [t-w+1 ... t]
            window = data[i-self.window_size+1 : i+1]
            
            # Target/Return for t+1
            target = targets.iloc[i]
            ret = returns_arr[i]
            vol = vol_arr[i]
            
            # No Rolling Normalization (Global Normalization will be applied later)
            X_list.append(window)
            y_list.append(target)
            ret_list.append(ret)
            vol_list.append(vol)
            valid_timestamps.append(timestamps[i])
            
        return (np.array(X_list), np.array(y_list), np.array(ret_list), 
                np.array(vol_list), pd.DatetimeIndex(valid_timestamps))

if __name__ == "__main__":
    # Simple test
    dates = pd.date_range(start="2020-01-01", periods=200)
    df = pd.DataFrame({
        'Open': np.random.rand(200) * 100,
        'High': np.random.rand(200) * 110,
        'Low': np.random.rand(200) * 90,
        'Close': np.random.rand(200) * 100,
        'Volume': np.random.rand(200) * 1000000
    }, index=dates)
    
    fe = FeatureEngineer(window_size=64)
    X, y, ts = fe.create_dataset(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
