# analytics module, used for statistical analysis of time price data

import pandas as pd
import numpy as np

# --- Module 2.5: Analytics module: statistical analysis  ---
class AnalyticsModule:
    """
    Calculates returns, log returns, and their rolling statistics.
    Returns new DataFrames with calculated analytics.
    """
    def __init__(self):
        pass

    def calculate_returns_and_volatility(self, data: pd.DataFrame, period1: int = 10, period2: int = 20, period3: int = 30) -> pd.DataFrame:
        """
        Calculates daily returns, daily log returns, and their rolling means and standard deviations.

        Args:
            data (pd.DataFrame): The input DataFrame with a 'close' column and 'date' as index.
            period1: used for rolling windows
            period2: used for rolling windows
            period3: used for rolling windows

        Returns:
            pd.DataFrame: A new DataFrame with calculated return mean and volatility metrics.
        """
        if 'close' not in data.columns:
            print("Error: 'close' column not found in data for returns calculation.")
            return pd.DataFrame()

        analytics_df = pd.DataFrame(index=data.index)

        # daily returns, rolling mean, rolling vol (std dev)
        analytics_df['daily_returns'] = data['close'].pct_change()
        analytics_df['daily_returns_squared'] = data['close'].pct_change() ** 2
        analytics_df['rolling_mean_returns_1'] = analytics_df['daily_returns'].rolling(window=period1).mean()
        analytics_df['rolling_mean_returns_2'] = analytics_df['daily_returns'].rolling(window=period2).mean()
        analytics_df['rolling_mean_returns_3'] = analytics_df['daily_returns'].rolling(window=period3).mean()
        analytics_df['vol_1'] = analytics_df['daily_returns'].rolling(window=period1).std()
        analytics_df['vol_2'] = analytics_df['daily_returns'].rolling(window=period2).std()
        analytics_df['vol_3'] = analytics_df['daily_returns'].rolling(window=period3).std()

        # Daily Log Returns and std dev (log vol)
        analytics_df['daily_log_returns'] = np.log(data['close'] / data['close'].shift(1))
        analytics_df['daily_log_returns_squared'] = np.log(data['close'] / data['close'].shift(1)) ** 2
        analytics_df['rolling_mean_log_returns_1'] = analytics_df['daily_log_returns'].rolling(window=period1).mean()
        analytics_df['rolling_mean_log_returns_2'] = analytics_df['daily_log_returns'].rolling(window=period2).mean()
        analytics_df['rolling_mean_log_returns_3'] = analytics_df['daily_log_returns'].rolling(window=period3).mean()
        analytics_df['vol_log_1'] = analytics_df['daily_log_returns'].rolling(window=period1).std()
        analytics_df['vol_log_2'] = analytics_df['daily_log_returns'].rolling(window=period2).std()
        analytics_df['vol_log_3'] = analytics_df['daily_log_returns'].rolling(window=period3).std()

        # Drop any NaN rows that result from rolling calculations at the beginning
        return analytics_df.dropna()