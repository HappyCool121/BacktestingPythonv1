# analytics module, used for statistical analysis of time price data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from numpy import cumsum, log, polyfit, sqrt, std, subtract, var, log10


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
            series_1 (pd.Series): first data series (can be any series, doesn't have to be log returns)
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

    def perform_comprehensive_stationarity_test(self, series: pd.Series, significance_level: float = 0.05,
                                                adf_regression: str = 'c', kpss_regression: str = "c",
                                                kpss_nlags: str = "auto"):
        """
        Performs both ADF and KPSS tests to provide a comprehensive stationarity analysis.

        This method combines two complementary tests:
        - ADF Test: H0 = non-stationary, H1 = stationary
        - KPSS Test: H0 = stationary, H1 = non-stationary

        By using both tests together, we can classify results into four categories:
        1. Both reject H0 → Conflicting results (further investigation needed)
        2. ADF rejects, KPSS fails to reject → Strong evidence for stationarity
        3. ADF fails to reject, KPSS rejects → Strong evidence for non-stationarity
        4. Both fail to reject H0 → Inconclusive results

        Args:
            series (pd.Series): The time series data to test
            significance_level (float): Significance level for both tests
            adf_regression (str): Regression type for ADF test ('c', 'ct', 'ctt', 'nc')
            kpss_regression (str): Regression type for KPSS test ('c' or 'ct')
            kpss_nlags (str or int): Lag selection for KPSS test

        Returns:
            dict: Comprehensive results from both tests with combined interpretation
        """
        if not isinstance(series, pd.Series):
            print("Error: Input must be a pandas Series.")
            return {}

        # Ensure we have sufficient data
        series_cleaned = series.dropna()
        if len(series_cleaned) < 10:
            print("Error: Insufficient data points for stationarity tests. Need at least 10 observations.")
            return {}

        print(f"\n{'=' * 80}")
        print(f"COMPREHENSIVE STATIONARITY ANALYSIS FOR '{series_cleaned.name}'")
        print(f"{'=' * 80}")
        print(f"Sample size: {len(series_cleaned)} observations")
        print(f"Significance level: {significance_level}")

        # Initialize results dictionary
        comprehensive_results = {
            'series_name': series_cleaned.name,
            'sample_size': len(series_cleaned),
            'significance_level': significance_level,
            'adf': {},
            'kpss': {}
        }

        # Perform ADF Test
        print(f"\n{'-' * 40} ADF TEST {'-' * 40}")
        try:
            adf_result = adfuller(series_cleaned, maxlag=None, regression=adf_regression,
                                  autolag='AIC', store=False, regresults=False)

            adf_statistic = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_lags = adf_result[2]
            adf_nobs = adf_result[3]
            adf_critical_values = adf_result[4]

            # ADF Test results display
            print(f'ADF Statistic    : {adf_statistic:.6f}')
            print(f'ADF p-value      : {adf_pvalue:.6f}')
            print(f'Lags Used        : {adf_lags}')
            print(f'Observations     : {adf_nobs}')

            adf_rejects_h0 = adf_pvalue <= significance_level
            adf_conclusion = "STATIONARY" if adf_rejects_h0 else "NON-STATIONARY"
            print(f'ADF Conclusion   : {adf_conclusion} (H0: non-stationary)')

            comprehensive_results['adf'] = {
                'statistic': adf_statistic,
                'p_value': adf_pvalue,
                'lags_used': adf_lags,
                'critical_values': adf_critical_values,
                'rejects_h0': adf_rejects_h0,
                'conclusion': adf_conclusion
            }

        except Exception as e:
            print(f"ADF Test failed: {str(e)}")
            comprehensive_results['adf'] = {'error': str(e)}
            return comprehensive_results

        # Perform KPSS Test
        print(f"\n{'-' * 40} KPSS TEST {'-' * 39}")
        try:
            kpss_result = kpss(series_cleaned, regression= "c", nlags= "auto")

            kpss_statistic = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_lags = kpss_result[2]
            kpss_critical_values = kpss_result[3]

            # KPSS Test results display
            print(f'KPSS Statistic   : {kpss_statistic:.6f}')
            print(f'KPSS p-value     : {kpss_pvalue:.6f}')
            print(f'Lags Used        : {kpss_lags}')
            print(f'Observations     : {len(series_cleaned)}')

            kpss_rejects_h0 = kpss_pvalue <= significance_level
            kpss_conclusion = "NON-STATIONARY" if kpss_rejects_h0 else "STATIONARY"
            print(f'KPSS Conclusion  : {kpss_conclusion} (H0: stationary)')

            comprehensive_results['kpss'] = {
                'statistic': kpss_statistic,
                'p_value': kpss_pvalue,
                'lags_used': kpss_lags,
                'critical_values': kpss_critical_values,
                'rejects_h0': kpss_rejects_h0,
                'conclusion': kpss_conclusion
            }

        except Exception as e:
            print(f"KPSS Test failed: {str(e)}")
            comprehensive_results['kpss'] = {'error': str(e)}
            return comprehensive_results

        # Combined Analysis and Interpretation
        print(f"\n{'-' * 35} COMBINED ANALYSIS {'-' * 35}")
        print("Test Summary:")
        print(f"  ADF Test:  {adf_conclusion} (p-value: {adf_pvalue:.6f})")
        print(f"  KPSS Test: {kpss_conclusion} (p-value: {kpss_pvalue:.6f})")

        # Determine combined conclusion based on test results
        if adf_rejects_h0 and not kpss_rejects_h0:
            # ADF says stationary, KPSS says stationary → Strong evidence for stationarity
            final_conclusion = "STATIONARY"
            confidence_level = "HIGH"
            interpretation = ("Both tests agree: Strong evidence that the series is STATIONARY.\n"
                              "ADF rejects non-stationarity, KPSS accepts stationarity.")

        elif not adf_rejects_h0 and kpss_rejects_h0:
            # ADF says non-stationary, KPSS says non-stationary → Strong evidence for non-stationarity
            final_conclusion = "NON-STATIONARY"
            confidence_level = "HIGH"
            interpretation = ("Both tests agree: Strong evidence that the series is NON-STATIONARY.\n"
                              "ADF accepts non-stationarity, KPSS rejects stationarity.")

        elif adf_rejects_h0 and kpss_rejects_h0:
            # ADF says stationary, KPSS says non-stationary → Conflicting results
            final_conclusion = "CONFLICTING"
            confidence_level = "LOW"
            interpretation = ("Conflicting results: ADF suggests stationarity while KPSS suggests non-stationarity.\n"
                              "This may indicate the presence of structural breaks or other complications.\n"
                              "Consider: (1) Different lag selections, (2) Structural break tests, (3) Visual inspection.")

        else:
            # Both fail to reject their respective null hypotheses → Inconclusive
            final_conclusion = "INCONCLUSIVE"
            confidence_level = "LOW"
            interpretation = ("Inconclusive results: Neither test provides strong evidence.\n"
                              "ADF cannot reject non-stationarity, KPSS cannot reject stationarity.\n"
                              "Consider: (1) Larger sample size, (2) Different significance levels, (3) Visual analysis.")

        print(f"\nFINAL ASSESSMENT:")
        print(f"  Conclusion: {final_conclusion}")
        print(f"  Confidence: {confidence_level}")
        print(f"\nInterpretation:")
        for line in interpretation.split('\n'):
            if line.strip():
                print(f"  {line}")

        # Add combined results to return dictionary
        comprehensive_results['combined_analysis'] = {
            'final_conclusion': final_conclusion,
            'confidence_level': confidence_level,
            'interpretation': interpretation,
            'adf_rejects_h0': adf_rejects_h0,
            'kpss_rejects_h0': kpss_rejects_h0
        }

        print(f"\n{'=' * 80}")

        return comprehensive_results

    def calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 100):
        """
        Calculates the Hurst Exponent of a time series using Rescaled Range (R/S) analysis.

        Args:
            series (pd.Series): The time series data to analyze.
            max_lag (int): The maximum number of lags to use for the calculation.
        """
        if not isinstance(series, pd.Series):
            print("Error: Input must be a pandas Series.")
            return

        series_cleaned = series.dropna()

        if len(series_cleaned) < max_lag:
            print(
                f"Error: Series length ({len(series_cleaned)}) is shorter than max_lag ({max_lag}). Cannot calculate Hurst Exponent.")
            return np.nan

        lags = range(2, max_lag)

        print("Index diagnostics:")
        print(f"Index type: {type(series_cleaned.index)}")
        print(f"Index min: {series_cleaned.index.min()}")
        print(f"Index max: {series_cleaned.index.max()}")
        print(f"Index is continuous: {series_cleaned.index.is_monotonic_increasing}")
        print(f"First 10 index values: {series_cleaned.index[:10].tolist()}")
        print(f"Last 10 index values: {series_cleaned.index[-10:].tolist()}")
        print(f"Any duplicate indices: {series_cleaned.index.duplicated().any()}")

        # Let's also check what happens when we manually align the arrays
        lag = 2
        array1 = series_cleaned.iloc[lag:]  # Use iloc for position-based indexing
        array2 = series_cleaned.iloc[:-lag]
        manual_diff = array1.values - array2.values  # Convert to numpy arrays to avoid index alignment
        print(f"\nManual difference calculation (first 20 values): {manual_diff[:20]}")
        print(f"Manual diff std: {np.std(manual_diff)}")

        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(series_cleaned.iloc[lag:].values - series_cleaned.iloc[:-lag].values)) for lag in lags]
        print("Calculated Tau values:", tau)

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

    def calculate_hurst_exponent_corrected(self, series: pd.Series,
                                           min_window: int = 10,
                                           max_window: int = None,
                                           n_windows: int = 20) -> tuple[float, dict]:
        """
        Calculates the Hurst Exponent using the proper Rescaled Range (R/S) methodology.

        This implementation fixes the fundamental issues with the previous approach by:
        1. Using cumulative deviations from the mean (not direct differences)
        2. Calculating true rescaled range statistics (R/S)
        3. Ensuring mathematically consistent results between 0 and 1

        Args:
            series: Time series data to analyze
            min_window: Minimum window size for analysis
            max_window: Maximum window size (defaults to series_length // 4)
            n_windows: Number of different window sizes to test

        Returns:
            Tuple of (hurst_exponent, diagnostic_info)
        """

        # Step 1: Clean and validate the data (at least 50 data points)
        clean_series = series.dropna()
        if len(clean_series) < 50:
            raise ValueError(f"Need at least 50 data points, got {len(clean_series)}")

        # Step 2: Set reasonable window size range
        if max_window is None:
            max_window = len(clean_series) // 4
            print(f'max window: {max_window}, min window: {min_window}, length of series: {len(clean_series)}')

        if max_window <= min_window:
            raise ValueError(f'max_window ({max_window}) must be greater than min_window {min_window}')

        # Step 3: Generate window sizes to test
        # Using logarithmic spacing gives better coverage across scales
        window_sizes = np.unique(
            np.logspace(np.log10(min_window), np.log10(max_window), n_windows).astype(int)
        )

        print(f"Analyzing {len(window_sizes)} window sizes from {window_sizes[0]} to {window_sizes[-1]}")



        # Step 4: Calculate R/S statistics for each window size
        rs_statistics = []

        for window_size in window_sizes:

            print(f'calculating RS for window size {window_size}...')
            # Calculate how many non-overlapping windows we can fit
            n_segments = len(clean_series) // window_size

            if n_segments < 2:  # Need at least 2 segments for meaningful analysis
                continue

            segment_rs_values = []

            # Step 5: Analyze each segment separately
            for segment_idx in range(n_segments):
                # Extract the segment data
                start_idx = segment_idx * window_size
                end_idx = start_idx + window_size
                segment_data = clean_series.iloc[start_idx:end_idx].values

                # Step 6: Calculate mean and deviations for this segment
                segment_mean = np.mean(segment_data)
                deviations = segment_data - segment_mean

                # Step 7: Calculate cumulative sum of deviations
                # This is the key step that captures long-term memory effects
                cumulative_deviations = np.cumsum(deviations)

                # Step 8: Calculate the Range (R)
                # Range is the difference between max and min cumulative deviation
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

                # Step 9: Calculate the Standard deviation (S)
                # Use sample standard deviation (ddof=1)
                S = np.std(segment_data, ddof=1)

                # Step 10: Calculate R/S ratio (the rescaled range)
                # This normalizes the range by the variability in the original data
                if S > 0:  # Avoid division by zero
                    segment_rs_values.append(R / S)

            # Step 11: Average R/S across all segments for this window size
            if segment_rs_values:
                mean_rs = np.mean(segment_rs_values)
                rs_statistics.append(mean_rs)
            else:
                rs_statistics.append(np.nan)

        # Step 12: Remove any invalid values
        valid_data = [(ws, rs) for ws, rs in zip(window_sizes, rs_statistics)
                      if not np.isnan(rs) and rs > 0]

        if len(valid_data) < 5:
            raise ValueError("Not enough valid data points for reliable Hurst estimation")

        valid_windows, valid_rs = zip(*valid_data)
        valid_windows = np.array(valid_windows)
        valid_rs = np.array(valid_rs)

        # Step 13: Fit the power law relationship
        # The Hurst exponent is the slope of log(window_size) vs log(R/S)
        log_windows = np.log10(valid_windows)
        log_rs = np.log10(valid_rs)

        # Linear regression to find the slope
        coefficients = np.polyfit(log_windows, log_rs, 1)
        hurst_exponent = coefficients[0]  # The slope IS the Hurst exponent
        intercept = coefficients[1]

        # Step 14: Calculate goodness-of-fit measure
        predicted_log_rs = hurst_exponent * log_windows + intercept
        ss_residual = np.sum((log_rs - predicted_log_rs) ** 2)
        ss_total = np.sum((log_rs - np.mean(log_rs)) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

        # Step 15: Create diagnostic information
        diagnostics = {
            'window_sizes': valid_windows,
            'rs_values': valid_rs,
            'r_squared': r_squared,
            'intercept': intercept,
            'n_data_points': len(clean_series),
            'n_window_sizes': len(valid_windows),
            'log_windows': log_windows,
            'log_rs': log_rs
        }

        # Step 16: Validate the result
        if hurst_exponent < 0 or hurst_exponent > 1:
            print(f"WARNING: Hurst exponent {hurst_exponent:.4f} is outside valid range [0,1]")
            print("This suggests either data quality issues or computational problems")

        # Step 17: Display results with interpretation
        print(f"\n" + "=" * 60)
        print(f"RESCALED RANGE ANALYSIS RESULTS")
        print(f"=" * 60)
        print(f"Series: {series.name if series.name else 'Unnamed'}")
        print(f"Data points analyzed: {len(clean_series)}")
        print(f"Window sizes tested: {len(valid_windows)}")
        print(f"Hurst Exponent: {hurst_exponent:.4f}")
        print(f"R-squared (fit quality): {r_squared:.4f}")

        print(f"\n" + "-" * 40 + " INTERPRETATION " + "-" * 40)

        if 0 <= hurst_exponent < 0.5:
            strength = "Strong" if hurst_exponent < 0.4 else "Moderate" if hurst_exponent < 0.45 else "Weak"
            print(f">> {strength} MEAN-REVERTING behavior (H = {hurst_exponent:.3f})")
            print(f"   • The series tends to return toward its mean over time")
            print(f"   • High values are more likely to be followed by lower values")
            print(f"   • This suggests potential opportunities in mean-reversion strategies")

        elif hurst_exponent > 0.5:
            strength = "Strong" if hurst_exponent > 0.6 else "Moderate" if hurst_exponent > 0.55 else "Weak"
            print(f">> {strength} TRENDING behavior (H = {hurst_exponent:.3f})")
            print(f"   • The series shows persistence - trends tend to continue")
            print(f"   • High values are more likely to be followed by higher values")
            print(f"   • This suggests potential opportunities in momentum strategies")

        else:  # Very close to 0.5
            print(f">> RANDOM WALK behavior (H ≈ 0.5)")
            print(f"   • The series shows no significant long-term memory")
            print(f"   • Future movements are largely independent of past movements")
            print(f"   • This is consistent with efficient market hypothesis")

        # Step 18: Quality warnings
        if r_squared < 0.8:
            print(f"\n⚠️  WARNING: Low R-squared ({r_squared:.3f}) indicates poor fit")
            print(f"   Consider using more data or checking for non-stationarity")

        if len(valid_windows) < 10:
            print(f"\n⚠️  WARNING: Only {len(valid_windows)} window sizes analyzed")
            print(f"   Results may be less reliable with fewer data points")

        print("=" * 60)

        return float(hurst_exponent), diagnostics

    def plot_hurst_analysis(self, hurst_exponent: float, diagnostics: dict,
                            series_name: str = "Time Series"):
        """
        Creates a visualization of the Hurst exponent analysis to help understand the results.

        This plot shows the log-log relationship that defines the Hurst exponent,
        helping you see how well the power law fits your data.
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: The fundamental R/S relationship
        log_windows = diagnostics['log_windows']
        log_rs = diagnostics['log_rs']

        # Scatter plot of actual data points
        ax1.scatter(log_windows, log_rs, alpha=0.7, s=50, color='blue',
                    label='Calculated R/S values')

        # Fitted line
        fitted_line = hurst_exponent * log_windows + diagnostics['intercept']
        ax1.plot(log_windows, fitted_line, 'r-', linewidth=2,
                 label=f'Fitted line (H = {hurst_exponent:.3f})')

        # Reference lines for comparison
        theoretical_05 = 0.5 * log_windows + diagnostics['intercept']
        ax1.plot(log_windows, theoretical_05, 'g--', alpha=0.5,
                 label='H = 0.5 (Random Walk)')

        ax1.set_xlabel('log₁₀(Window Size)', fontsize=12)
        ax1.set_ylabel('log₁₀(R/S)', fontsize=12)
        ax1.set_title(f'Rescaled Range Analysis\nR² = {diagnostics["r_squared"]:.3f}',
                      fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: R/S values vs window size (linear scale)
        windows = diagnostics['window_sizes']
        rs_values = diagnostics['rs_values']

        ax2.loglog(windows, rs_values, 'bo-', alpha=0.7, markersize=6,
                   label='R/S Statistics')

        # Theoretical lines for comparison
        theoretical_rs_05 = (windows / windows[0]) ** 0.5 * rs_values[0]
        theoretical_rs_h = (windows / windows[0]) ** hurst_exponent * rs_values[0]

        ax2.loglog(windows, theoretical_rs_05, 'g--', alpha=0.5,
                   label='H = 0.5 Expected')
        ax2.loglog(windows, theoretical_rs_h, 'r-', linewidth=2,
                   label=f'H = {hurst_exponent:.3f} Fitted')

        ax2.set_xlabel('Window Size', fontsize=12)
        ax2.set_ylabel('R/S Statistic', fontsize=12)
        ax2.set_title(f'R/S Scaling Behavior\n{series_name}', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Example usage with your data:
    def analyze_your_returns(self, daily_returns_series):
        """
        Analyzes your daily returns using the corrected Hurst exponent method.

        This function demonstrates how to use the corrected implementation
        and interpret the results for financial time series data.
        """
        try:
            # Calculate the Hurst exponent using the proper method
            hurst, diagnostics = self.calculate_hurst_exponent_corrected(
                daily_returns_series,
                min_window=10,  # Start with 10-day windows
                max_window=None,  # Let the function choose based on data length
                n_windows=25  # Test 25 different window sizes
            )

            # Create visualization to help understand the results
            self.plot_hurst_analysis(hurst, diagnostics, daily_returns_series.name or "Daily Returns")

            return hurst, diagnostics

        except Exception as e:
            print(f"Error in Hurst analysis: {e}")
            return None, None

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