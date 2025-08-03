# analytics module, used for statistical analysis of time price data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import cumsum, log, polyfit, sqrt, std, subtract, var, log10


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

    def perform_adf_test(self, series: pd.Series, significance_level: float = 0.05, regression: str = 'c', ):
        """
        Performs the Augmented Dickey-Fuller (ADF) test on a time series to check for stationarity.

        The null hypothesis of the test is that the series has a unit root (it is non-stationary).
        If the p-value is below the significance level, we reject the null hypothesis.

        Args:
            series (pd.Series): The time series data to test (e.g., a column of closing prices).
            significance_level (float): The threshold for the p-value to reject the null hypothesis.
        """
        if not isinstance(series, pd.Series):
            print("Error: Input must be a pandas Series.")
            return

        # Drop NaN values which can cause errors in the test
        series_cleaned = series.dropna()

        print(f"\n--- Augmented Dickey-Fuller Test Results for '{series_cleaned.name}' ---")

        # Perform the ADF test
        # The autolag='AIC' parameter automatically selects the optimal number of lags
        adf_result = adfuller(series_cleaned, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)

        # Extract and print the results in a readable format
        print(f'ADF Statistic    : {adf_result[0]:.20f}')
        print(f'p-value          : {adf_result[1]:.20f}')
        print(f'# Lags Used      : {adf_result[2]}')
        print(f'# Observations   : {adf_result[3]}')

        print('\nCritical Values:')
        for key, value in adf_result[4].items():
            print(f'\t{key}: {value:.4f}')

        # --- Interpretation of the results ---
        print("\n--- Interpretation ---")
        if adf_result[1] <= significance_level:
            print(f"Conclusion: p-value ({adf_result[1]:.4f}) is less than or equal to {significance_level}.")
            print(">> We reject the null hypothesis.")
            print(">> The series is likely STATIONARY (does not have a unit root).")
        else:
            print(f"Conclusion: p-value ({adf_result[1]:.4f}) is greater than {significance_level}.")
            print(">> We fail to reject the null hypothesis.")
            print(">> The series is likely NON-STATIONARY (has a unit root).")
        print("------------------------")

    def calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 100):
        """
        Calculates the Hurst Exponent of a time series using Rescaled Range (R/S) analysis.

        Args:
            series (pd.Series): The time series data to analyze.
            max_lag (int): The maximum number of lags to use for the calculation.
        """
        if not isinstance(series, pd.Series):
            print("Error: Input must be a pandas Series.")
            return 0

        series_cleaned = series.dropna()
        lags = range(2, max_lag)

        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(series_cleaned[lag:], series_cleaned[:-lag]))) for lag in lags]

        # Use a polyfit to plot the log of lags vs the log of tau
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        # The Hurst Exponent is the slope of the line
        hurst_exponent = poly[0] * 2.0

        print(f"\n--- Hurst Exponent Calculation for '{series_cleaned.name}' ---")
        print(f"Hurst Exponent   : {hurst_exponent:.4f}")

        # --- Interpretation of the results ---
        print("\n--- Interpretation ---")
        if hurst_exponent < 0.5:
            print(">> H < 0.5: The series is likely MEAN-REVERTING (anti-persistent).")
            print("   A high value is likely to be followed by a low value and vice-versa.")
        elif hurst_exponent > 0.5:
            print(">> H > 0.5: The series is likely TRENDING (persistent).")
            print("   A high value is likely to be followed by another high value.")
        else:
            print(">> H = 0.5: The series is likely a RANDOM WALK.")
            print("   The movements are unpredictable.")
        print("------------------------")

        return hurst_exponent

    def get_hurst_exponent(self, pd: pd.Series, max_lag=1000):
        """Returns the Hurst Exponent of the time series"""

        lags = range(2, max_lag)

        time_series = pd.dropna()

        # variances of the lagged differences
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

        # calculate the slope of the log plot -> the Hurst Exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)

        return reg[0]


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

    def plot_single_column(self, df: pd.DataFrame, column_name: str, title: str = 'Single Column Plot', xlabel: str = 'Index',
                           ylabel: str = None):
        """
        Plots a single specified column from a pandas DataFrame against its index.

        Args:
            df (pd.DataFrame): The DataFrame containing the data. Its index will be the x-axis.
            column_name (str): The name of the column to plot on the y-axis.
            title (str): The title for the chart.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis. Defaults to the column_name if not provided.
        """
        # --- Input Validation ---
        if not isinstance(df, pd.DataFrame):
            print("Error: Input must be a pandas DataFrame.")
            return

        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the DataFrame.")
            print(f"Available columns are: {list(df.columns)}")
            return

        if df.empty:
            print("Warning: The provided DataFrame is empty. Nothing to plot.")
            return

        # If ylabel is not provided, use the column name
        if ylabel is None:
            ylabel = column_name

        # --- Plotting Setup ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 8))

        # --- Plot the Selected Column ---
        # This command uses the DataFrame's index for the x-axis and the specified column for the y-axis.
        df[column_name].plot(ax=ax, color='royalblue', lw=2)

        # --- Formatting ---
        ax.set_title(f'Plot of {column_name}', fontsize=16)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def plot_all_columns_on_shared_axis(df: pd.DataFrame, title: str = 'DataFrame Plot', xlabel: str = 'Index',
                                        ylabel: str = 'Values'):
        """
        Plots every column of a pandas DataFrame as a separate line on a single chart.
        All lines share the DataFrame's index as their common x-axis.

        Args:
            df (pd.DataFrame): The DataFrame to plot. Its index will be the x-axis.
            title (str): The title for the chart.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
        """
        if not isinstance(df, pd.DataFrame):
            print("Error: Input must be a pandas DataFrame.")
            return

        if df.empty:
            print("Warning: The provided DataFrame is empty. Nothing to plot.")
            return

        # --- Plotting Setup ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 8))

        # --- Plot Each Column ---
        # This single command tells pandas to use the DataFrame's index for the x-axis
        # and to create a separate line (series) for each column on the y-axis.
        # It automatically assigns different colors and uses column names for the legend.
        df.plot(ax=ax)

        # --- Formatting ---
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # Place the legend outside of the plot area to avoid covering data
        ax.legend(title='Columns', bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout(rect=(0, 0, 0.9, 1))  # Adjust layout to make space for the legend
        plt.grid(True)
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