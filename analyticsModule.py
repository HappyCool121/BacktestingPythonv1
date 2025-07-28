# analytics module, used for statistical analysis of time price data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, adfuller, kpss

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

    def rolling_coefficient_pearson (self, series_1: pd.Series, series_2: pd.Series, symbol1: str = "", symbol2: str ="", window: int = 7) -> dict:
        """
        Calculates and plots the rolling Pearson correlation coefficient between two assets' log returns.

        Args:
            series_1 (pd.Series): first data series (can be any series, doesnt have to be log returns)
            series_2 (pd.Series): second data series (to compare to)
            symbol1 (str): The ticker symbol of the first asset (for documentation, actual data depends on input series)
            symbol2 (str): The ticker symbol of the second asset (for documentation, actual data depends on input series)
            window (int): The rolling window size (number of periods) for correlation calculation.

        returns:
            dict containing:
                df of all rolling correlation values up to the most current date
                statistics regarding rolling coefficient:
                    series.mean(): Calculates the arithmetic mean.
                    series.std(): Calculates the standard deviation.
                    series.median(): Calculates the median (50th percentile).
                    series.min(): Returns the minimum value.
                    series.max(): Returns the maximum value.
                    series.quantile(q): Returns the value at the specified quantile q (e.g., 0.25 for the 25th percentile).
                    series.count(): Returns the number of non-null observations.
                    series.var(): Calculates the variance.
                    series.skew(): Calculates the skewness. A positive skew means the tail on the right side is longer or fatter. A negative skew means the tail on the left side is longer or fatter.
                    series.kurt(): Calculates the kurtosis. Positive kurtosis (leptokurtic) means more extreme outliers than a normal distribution. Negative kurtosis (platykurtic) means fewer extreme outliers.
        """

        mydict = {} # empty dict to return when any error is encountered
            # Calculate rolling coefficient based on given window
        rolling_corr_pair = series_1.rolling(window=window).corr(series_2).dropna()

        if rolling_corr_pair.empty:
            print(
                f"No sufficient data to calculate {window}-day rolling correlation for {symbol1} and {symbol2}. Check window size or data range.")
            return mydict

        if not rolling_corr_pair.empty: # stats regarding rolling correlation
            print(f"\n--- Statistics for Rolling {window}-Day Correlation ({symbol1} vs {symbol2}) ---")
            # Mean
            print(f"Mean: {rolling_corr_pair.mean():.4f}")
            # Standard Deviation
            print(f"Standard Deviation: {rolling_corr_pair.std():.4f}")
            # Median
            print(f"Median: {rolling_corr_pair.median():.4f}")
            # Minimum
            print(f"Minimum: {rolling_corr_pair.min():.4f}")
            # Maximum
            print(f"Maximum: {rolling_corr_pair.max():.4f}")
            # Quantiles (e.g., 25th, 50th, 75th percentiles)
            print(f"25th Percentile (Q1): {rolling_corr_pair.quantile(0.25):.4f}")
            print(f"75th Percentile (Q3): {rolling_corr_pair.quantile(0.75):.4f}")
            # Count of non-NA/null values
            print(f"Count: {rolling_corr_pair.count()}")
            # Variance
            print(f"Variance: {rolling_corr_pair.var():.4f}")
            # Skewness (measure of asymmetry of the distribution)
            print(f"Skewness: {rolling_corr_pair.skew():.4f}")
            # Kurtosis (measure of 'tailedness' of the distribution)
            print(f"Kurtosis: {rolling_corr_pair.kurt():.4f}")
            # --- The describe() method for a quick summary ---
            print(f"\n--- Full Statistics Summary (.describe()) for Rolling {window}-Day Correlation ---")
            print(rolling_corr_pair.describe())

        rolling_coefficient_results = {
            'rolling_corr_df': rolling_corr_pair,
            'mean': rolling_corr_pair.mean(),
            'std': rolling_corr_pair.std(),
            'median': rolling_corr_pair.median(),
            'min': rolling_corr_pair.min(),
            'max': rolling_corr_pair.max(),
            'quantile_25': rolling_corr_pair.quantile(0.25),
            'quantile_75': rolling_corr_pair.quantile(0.75),
            'count': rolling_corr_pair.count(),
            'variance': rolling_corr_pair.var(),
            'skewness': rolling_corr_pair.skew(),
            'kurtosis': rolling_corr_pair.kurt()
        }

        # Plotting the correlation values
        plt.figure(figsize=(12, 6))
        rolling_corr_pair.plot(title=f'Rolling {window}-Day Pearson Correlation: {symbol1} vs {symbol2}')
        plt.xlabel('Date')
        plt.ylabel('Correlation Coefficient')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        return rolling_coefficient_results

    def get_autocorrelation_values(self, data_series: pd.Series, nlags: int = None, qstat: bool = False):
        """
        Calculates the Autocorrelation Function (ACF) values and optionally Q-statistic.

        Args:
            data_series (pd.Series): The time series data. Must not contain NaNs.
            nlags (int, optional): The number of lags to return. If None, statsmodels chooses automatically.
            qstat (bool, optional): If True, returns the Ljung-Box Q-statistic and p-value.

        Returns:
            np.ndarray: Array of ACF values.
            tuple (optional): Q-statistic and p-value if qstat is True.
        """
        if data_series.isnull().any():
            print("Warning: Input data_series contains NaNs. ACF calculation might be affected or fail. Dropping NaNs.")
            data_series = data_series.dropna()
            if data_series.empty:
                print("Error: data_series became empty after dropping NaNs. Cannot calculate ACF values.")
                return None if not qstat else (None, None, None)

        if qstat:
            # Returns acf_values, q_statistic, p_values
            return acf(x=data_series, nlags=nlags, qstat=True)
        else:
            return acf(x=data_series, nlags=nlags)

    def plot_autocorrelation(self, data_series: pd.Series, title: str = "Autocorrelation Function", lags: int = None,
                             alpha: float = 0.05):
        """
        Plots the Autocorrelation Function (ACF) for a given time series.

        Args:
            data_series (pd.Series): The time series data to analyze (e.g., daily log returns, rolling correlation).
                                     Must not contain NaNs.
            title (str): Title for the plot.
            lags (int, optional): The number of lags to plot. If None, statsmodels chooses automatically.
            alpha (float, optional): Significance level for the confidence intervals (e.g., 0.05 for 95%).
        """
        if data_series.isnull().any():
            print(
                "Warning: Input data_series contains NaNs. ACF calculation might be affected or fail. Dropping NaNs for plot.")
            data_series = data_series.dropna()
            if data_series.empty:
                print("Error: data_series became empty after dropping NaNs. Cannot plot ACF.")
                return

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_acf(x=data_series, lags=lags, alpha=alpha, ax=ax, title=(title + f" acf plot"),
                 fft=True)  # Use fft=True for potentially faster computation
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


"""
outline of how backtesting this hypothesis will go:

will first obtain data using data handler as per usual (get all currency pairs required)

from now on, everything else will be executed in "strategyModule"

will first run calculate_returns_and_volatility on both the target currency pairs

will then run rolling_coefficient_pearson on these two pairs to obtain the rolling coefficient series

with this series, we will calculate rolling mean and std dev (will have a relatively long lookback window)

whenever value goes below rolling std dev, a signal will be generated (for both currency pairs)

we also have to calculate SL/TP values, as well as position sizes (for both currency pairs)


what do we have to do before all this?

- implement a way to backtest two different assets at once, since we are testing an arbitrage strategy
    - will have to sort out portfolio management, backtestengine. for strategymodule, we can generate signals
        separately anyway 
    - will have to modify portfolio management such that position sizing depends on the data provided by the strategy 
        module. Currently, position sizing depends on SL as well as current equity (cash). 
        need to put trailing stop on hold
    - will need a new tag for trades class, which is the symbol the current trade is in, since more than one asset is 
        involved
    - ideally, datahandler class will output a dict containing dataframes for all symbols
    - ignore indicator module, monte carlo module and stats module for now (may need to revisit stats module in the future
        
"""