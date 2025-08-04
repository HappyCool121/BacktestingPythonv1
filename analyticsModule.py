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

    def plot_autocorrelation_with_interpretation(self, data_series: pd.Series, title: str = "Autocorrelation Function",
                                                 lags: int = None, alpha: float = 0.05, interpret_results: bool = True):
        """
        Plots the Autocorrelation Function (ACF) for a given time series with optional interpretation.

        The confidence intervals represent the bounds within which autocorrelations would fall
        if the series were pure white noise (random). Values outside these bounds suggest
        genuine patterns or structure in the data.

        Args:
            data_series (pd.Series): The time series data to analyze
            title (str): Title for the plot
            lags (int, optional): Number of lags to plot (default: min(10*log10(len(series)), len(series)-1))
            alpha (float): Significance level for confidence intervals (0.05 = 95% confidence)
            interpret_results (bool): Whether to print interpretation of results
        """

        if data_series.isnull().any():
            print("Warning: Input data_series contains NaNs. Dropping NaNs for analysis.")
            data_series = data_series.dropna()
            if data_series.empty:
                print("Error: data_series became empty after dropping NaNs. Cannot plot ACF.")
                return

        # Calculate ACF values for interpretation
        n_obs = len(data_series)
        max_lags = min(lags if lags else int(10 * np.log10(n_obs)), n_obs - 1)

        # Get ACF values for numerical analysis
        acf_values = acf(data_series, nlags=max_lags)

        # Calculate confidence bounds (approximate for large samples)
        confidence_bound = 1.96 / np.sqrt(n_obs)  # For 95% confidence

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_acf(x=data_series, lags=max_lags, alpha=alpha, ax=ax,
                 title=f"{title} (n={n_obs}, {100 * (1 - alpha):.0f}% Confidence)", fft=True)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        significant_lags = []
        if interpret_results:
            # Provide interpretation
            print(f"\n--- Autocorrelation Analysis for '{data_series.name}' ---")
            print(f"Sample size: {n_obs} observations")
            print(f"Confidence level: {100 * (1 - alpha):.0f}%")
            print(f"Confidence bounds: ±{confidence_bound:.4f}")

            # Count significant autocorrelations (excluding lag 0 which is always 1)
            for i in range(1, len(acf_values)):
                if abs(acf_values[i]) > confidence_bound:
                    significant_lags.append((i, acf_values[i]))

            if significant_lags:
                print(f"\nSignificant autocorrelations found at {len(significant_lags)} lags:")
                for lag, value in significant_lags[:5]:  # Show first 5
                    print(f"  Lag {lag}: {value:.4f}")
                if len(significant_lags) > 5:
                    print(f"  ... and {len(significant_lags) - 5} more")

                # Provide context based on data type
                if True:
                    print("\nInterpretation:")
                    print("- Significant autocorrelations suggest the series has memory")
                    print("- Past values provide information about future values")
                    print("- Consider this when building forecasting models")
            else:
                print(f"\nNo significant autocorrelations detected.")
                print("The series appears consistent with white noise (random process).")
                print("This suggests past values don't help predict future values.")

            # Warning about multiple testing
            expected_false_positives = max_lags * alpha
            if expected_false_positives >= 1:
                print(f"\nNote: With {max_lags} lags tested at {alpha:.0%} significance,")
                print(f"expect ~{expected_false_positives:.1f} false positives due to multiple testing.")

            print("---" + "-" * 50)

        return {
            'acf_values': acf_values,
            'confidence_bound': confidence_bound,
            'significant_lags': significant_lags if interpret_results else None,
            'n_observations': n_obs
        }

    class HurstAnalyzer: # Wrapped in a class for context

        def calculate_rolling_hurst_exponent(self, series: pd.Series, window_size: int = 252,
                                             min_periods: int = None, method: str = 'rs',
                                             plot_results: bool = True, interpret_results: bool = True):
            """
            Calculates the rolling Hurst exponent to identify time-varying mean reversion and persistence patterns.

            The Hurst exponent helps identify different market regimes:
            - H < 0.5: Mean-reverting behavior (anti-persistent)
            - H = 0.5: Random walk behavior (no memory)
            - H > 0.5: Trending/persistent behavior (momentum)

            This method reveals how these characteristics change over time, which is crucial for
            understanding when mean reversion strategies might be most effective.

            Args:
                series (pd.Series): Time series data (typically log returns)
                window_size (int): Rolling window size (default: 252 for ~1 year of daily data)
                min_periods (int): Minimum observations required (default: window_size // 2)
                method (str): Method for Hurst calculation ('rs' for R/S analysis, 'dfa' for detrended fluctuation)
                plot_results (bool): Whether to create visualization
                interpret_results (bool): Whether to provide detailed interpretation

            Returns:
                pd.Series: Rolling Hurst exponent values with same index as input series
            """

            if not isinstance(series, pd.Series):
                print("Error: Input must be a pandas Series.")
                return pd.Series()

            # Clean the data
            series_clean = series.dropna()
            if len(series_clean) < window_size:
                raise ValueError(f"Error: Series length ({len(series_clean)}) is less than window size ({window_size})")

            # Set minimum periods if not specified
            if min_periods is None:
                min_periods = max(50, window_size // 2)  # Ensure sufficient data for stable estimation

            print(f"Calculating rolling Hurst exponent for '{series_clean.name}'")
            print(f"Window size: {window_size}, Method: {method.upper()}")
            print(f"This may take a moment for large datasets...")

            def hurst_rs_method(data):
                """
                Calculate Hurst exponent using Rescaled Range (R/S) analysis.
                This is the classic method that examines how the range of cumulative deviations
                scales with different time horizons.
                """
                try:
                    # Remove any remaining NaNs within the window
                    clean_data = np.array(data.dropna())
                    if len(clean_data) < 10:  # Need minimum data for meaningful calculation
                        return np.nan

                    N = len(clean_data)
                    if N < 2:
                        return np.nan

                    # Calculate mean
                    mean_val = np.mean(clean_data)

                    # Calculate cumulative deviations from mean
                    cumulative_deviations = np.cumsum(clean_data - mean_val)

                    # Calculate range of cumulative deviations
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

                    # Calculate standard deviation
                    S = np.std(clean_data, ddof=1)

                    # Avoid division by zero
                    if S == 0 or R == 0:
                        return np.nan

                    # For a single window, we approximate Hurst by comparing R/S to expected value
                    # This is a simplified approach; full R/S analysis uses multiple time scales
                    rs_ratio = float(R) / float(S)

                    # Theoretical expectation for random walk: R/S ≈ sqrt(π*N/2)
                    expected_rs = np.sqrt(np.pi * N / 2)

                    # Hurst exponent approximation
                    # If actual R/S > expected, H > 0.5 (persistent)
                    # If actual R/S < expected, H < 0.5 (anti-persistent)
                    hurst_approx = 0.5 + 0.5 * np.log(rs_ratio / expected_rs) / np.log(2)

                    # Bound the result to reasonable range
                    return np.clip(hurst_approx, 0.0, 1.0)

                except Exception as e:
                    return np.nan

            def hurst_dfa_method(data):
                """
                Calculate Hurst exponent using Detrended Fluctuation Analysis (DFA).
                This method is often more robust and handles non-stationarities better.
                """
                try:
                    clean_data = np.array(data.dropna())
                    if len(clean_data) < 20:  # DFA needs more data points
                        return np.nan

                    N = len(clean_data)

                    # Step 1: Create integrated series (cumulative sum)
                    integrated_series = np.cumsum(clean_data - np.mean(clean_data))

                    # Step 2: Divide into boxes of different sizes
                    # Use a smaller range of box sizes for rolling window
                    max_box_size = min(N // 4, 50)  # Limit for computational efficiency
                    min_box_size = max(4, N // 20)

                    if max_box_size <= min_box_size:
                        return np.nan

                    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size),
                                            num=min(10, max_box_size - min_box_size + 1), dtype=int)
                    box_sizes = np.unique(box_sizes)  # Remove duplicates

                    if len(box_sizes) < 3:  # Need at least 3 points for regression
                        return np.nan

                    fluctuations = []

                    for box_size in box_sizes:
                        # Step 3: Detrend each box and calculate fluctuation
                        n_boxes = N // box_size
                        box_fluctuations = []

                        for i in range(n_boxes):
                            start_idx = i * box_size
                            end_idx = start_idx + box_size
                            box_data = integrated_series[start_idx:end_idx]

                            # Fit linear trend to this box
                            x = np.arange(len(box_data))
                            if len(box_data) > 1:
                                slope, intercept = np.polyfit(x, box_data, 1)
                                trend = slope * x + intercept
                                detrended = box_data - trend
                                fluctuation = np.sqrt(np.mean(detrended ** 2))
                                box_fluctuations.append(fluctuation)

                        if box_fluctuations:
                            avg_fluctuation = np.mean(box_fluctuations)
                            fluctuations.append(avg_fluctuation)
                        else:
                            fluctuations.append(np.nan)

                    # Remove any NaN fluctuations
                    valid_indices = ~np.isnan(fluctuations)
                    if np.sum(valid_indices) < 3:
                        return np.nan

                    valid_box_sizes = box_sizes[valid_indices]
                    valid_fluctuations = np.array(fluctuations)[valid_indices]

                    # Step 4: Calculate Hurst exponent from log-log slope
                    # F(n) ~ n^H, so log(F(n)) ~ H * log(n)
                    log_box_sizes = np.log(valid_box_sizes)
                    log_fluctuations = np.log(valid_fluctuations)

                    # Remove any infinite values
                    finite_mask = np.isfinite(log_box_sizes) & np.isfinite(log_fluctuations)
                    if np.sum(finite_mask) < 3:
                        return np.nan

                    hurst_exponent = np.polyfit(log_box_sizes[finite_mask],
                                                log_fluctuations[finite_mask], 1)[0]

                    # Bound the result
                    return np.clip(hurst_exponent, 0.0, 1.0)

                except Exception as e:
                    return np.nan

            # Choose the calculation method
            if method.lower() == 'rs':
                hurst_func = hurst_rs_method
            elif method.lower() == 'dfa':
                hurst_func = hurst_dfa_method
            else:
                print(f"Unknown method '{method}'. Using R/S method.")
                hurst_func = hurst_rs_method

            # Calculate rolling Hurst exponent
            rolling_hurst = series_clean.rolling(window=window_size, min_periods=min_periods).apply(
                hurst_func, raw=False
            )

            # Create results with proper naming
            rolling_hurst.name = f'{series_clean.name}_hurst_{method}_{window_size}d'

            if interpret_results:
                print(f"\n--- Rolling Hurst Exponent Analysis ---")
                print(f"Method: {method.upper()}")
                print(f"Window size: {window_size} periods")
                print(f"Valid calculations: {rolling_hurst.notna().sum()}/{len(rolling_hurst)}")

                if rolling_hurst.notna().sum() > 0:
                    valid_hurst = rolling_hurst.dropna()
                    print(f"\nDescriptive Statistics:")
                    print(f"  Mean Hurst: {valid_hurst.mean():.3f}")
                    print(f"  Std Dev: {valid_hurst.std():.3f}")
                    print(f"  Min: {valid_hurst.min():.3f}")
                    print(f"  Max: {valid_hurst.max():.3f}")
                    print(f"  Median: {valid_hurst.median():.3f}")

                    # Classify periods
                    mean_reverting = (valid_hurst < 0.5).sum()
                    random_walk = ((valid_hurst >= 0.45) & (valid_hurst <= 0.55)).sum()  # Buffer around 0.5
                    trending = (valid_hurst > 0.5).sum()

                    total_periods = len(valid_hurst)
                    print(f"\nMarket Regime Classification:")
                    print(
                        f"  Mean-reverting periods (H < 0.5): {mean_reverting} ({100 * mean_reverting / total_periods:.1f}%)")
                    print(f"  Random walk periods (H ≈ 0.5): {random_walk} ({100 * random_walk / total_periods:.1f}%)")
                    print(f"  Trending periods (H > 0.5): {trending} ({100 * trending / total_periods:.1f}%)")

                    # Identify most extreme periods
                    most_mean_reverting = valid_hurst.idxmin()
                    most_trending = valid_hurst.idxmax()
                    print(f"\nExtreme Periods:")
                    print(f"  Most mean-reverting: {most_mean_reverting} (H = {valid_hurst.min():.3f})")
                    print(f"  Most trending: {most_trending} (H = {valid_hurst.max():.3f})")

            if plot_results and rolling_hurst.notna().sum() > 0:
                # Create comprehensive visualization
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

                # Plot 1: Original series
                ax1.plot(series_clean.index, series_clean.values, alpha=0.7, linewidth=0.8)
                ax1.set_title(f'Original Series: {series_clean.name}')
                ax1.set_ylabel('Value')
                ax1.grid(True, alpha=0.3)

                # Plot 2: Rolling Hurst exponent
                valid_hurst_for_plot = rolling_hurst.dropna()
                ax2.plot(valid_hurst_for_plot.index, valid_hurst_for_plot.values,
                         linewidth=1.5, color='darkblue', label=f'Rolling Hurst ({window_size}d)')

                # Add reference lines
                ax2.axhline(y=0.5, color='blue', linestyle='--', alpha=0.7, label='Random Walk (H=0.5)')
                ax2.axhline(y=0.4, color='green', linestyle=':', alpha=0.7, label='Mean Reverting (H<0.5)')
                ax2.axhline(y=0.6, color='red', linestyle=':', alpha=0.7, label='Trending (H>0.5)')

                ax2.fill_between(valid_hurst_for_plot.index, 0.4, 0.6, alpha=0.1, color='gray',
                                 label='Neutral zone')

                ax2.set_title(f'Rolling Hurst Exponent ({method.upper()} method), window: {window_size}')
                ax2.set_ylabel('Hurst Exponent')
                ax2.set_ylim(0, 1)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Plot 3: Market regime classification
                regime_colors = []
                regime_values = []
                for h in valid_hurst_for_plot.values:
                    if h < 0.45:
                        regime_colors.append('green')
                        regime_values.append(1)  # Mean reverting
                    elif h > 0.55:
                        regime_colors.append('red')
                        regime_values.append(3)  # Trending
                    else:
                        regime_colors.append('blue')
                        regime_values.append(2)  # Random walk

                ax3.scatter(valid_hurst_for_plot.index, regime_values, c=regime_colors, alpha=0.6, s=10)
                ax3.set_yticks([1, 2, 3])
                ax3.set_yticklabels(
                    ['Mean Reverting\n(H < 0.45)', 'Random Walk\n(0.45 ≤ H ≤ 0.55)', 'Trending\n(H > 0.55)'])
                ax3.set_title('Market Regime Classification Over Time')
                ax3.set_xlabel('Date')
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

            return rolling_hurst

        def analyze_hurst_regimes(self, series: pd.Series, rolling_hurst: pd.Series,
                                  window_size: int = 252, regime_thresholds: tuple = (0.45, 0.55),
                                  min_regime_duration: int = 20):
            """
            Analyzes the rolling Hurst exponent results to identify distinct market regimes
            and their characteristics. This method helps validate and interpret the results
            from calculate_rolling_hurst_exponent.

            Args:
                series (pd.Series): Original time series data (e.g., daily log returns)
                rolling_hurst (pd.Series): Rolling Hurst exponent values
                window_size (int): Window size used for Hurst calculation
                regime_thresholds (tuple): (lower, upper) thresholds for regime classification
                min_regime_duration (int): Minimum periods for a regime to be considered significant

            Returns:
                dict: Comprehensive analysis results including regime periods, statistics, and validation
            """

            if len(rolling_hurst.dropna()) == 0:
                print("Error: No valid Hurst exponent values found.")
                return {}

            # Clean inputs and align indices
            valid_hurst = rolling_hurst.dropna()
            aligned_series = series.reindex(valid_hurst.index).dropna()

            if len(aligned_series) == 0:
                print("Error: No overlapping data between series and Hurst values.")
                return {}

            lower_threshold, upper_threshold = regime_thresholds

            print(f"\n--- Hurst Exponent Regime Analysis ---")
            print(
                f"Analysis period: {valid_hurst.index[0]} to {valid_hurst.index[-1]}")
            print(f"Regime thresholds: Mean-reverting < {lower_threshold}, Trending > {upper_threshold}")
            print(f"Minimum regime duration: {min_regime_duration} periods")

            # Classify each period into regimes
            def classify_regime(h_value):
                if h_value < lower_threshold:
                    return 'Mean-Reverting'
                elif h_value > upper_threshold:
                    return 'Trending'
                else:
                    return 'Random Walk'

            # Apply classification
            regime_classification = valid_hurst.apply(classify_regime)

            # Identify regime changes and continuous periods
            regime_changes = regime_classification.ne(regime_classification.shift())
            regime_periods = []

            current_regime = None
            regime_start_date = None

            for date, is_change in regime_changes.items():
                if is_change:
                    # End previous regime if it exists and meets criteria
                    if current_regime is not None and regime_start_date is not None:
                        end_date = regime_classification.index[regime_classification.index.get_loc(date) - 1]
                        period_slice = valid_hurst[regime_start_date:end_date]
                        duration = len(period_slice)

                        if duration >= min_regime_duration:
                            regime_periods.append({
                                'regime': current_regime,
                                'start_date': regime_start_date,
                                'end_date': end_date,
                                'duration': duration,
                                'avg_hurst': period_slice.mean(),
                                'std_hurst': period_slice.std()
                            })

                    # Start new regime
                    current_regime = regime_classification[date]
                    regime_start_date = date

            # Handle the last regime
            if current_regime is not None and regime_start_date is not None:
                period_slice = valid_hurst[regime_start_date:]
                duration = len(period_slice)
                if duration >= min_regime_duration:
                    regime_periods.append({
                        'regime': current_regime,
                        'start_date': regime_start_date,
                        'end_date': regime_classification.index[-1],
                        'duration': duration,
                        'avg_hurst': period_slice.mean(),
                        'std_hurst': period_slice.std()
                    })

            # Calculate regime statistics
            regime_stats = {}
            for regime_type in ['Mean-Reverting', 'Random Walk', 'Trending']:
                mask = regime_classification == regime_type
                if mask.sum() > 0:
                    regime_data = aligned_series[mask]
                    hurst_data = valid_hurst[mask]

                    # Ensure std is not zero before calculating Sharpe ratio
                    volatility = regime_data.std()
                    sharpe = (regime_data.mean() / volatility) * np.sqrt(252) if volatility > 0 else 0

                    regime_stats[regime_type] = {
                        'periods': mask.sum(),
                        'percentage': 100 * mask.sum() / len(regime_classification),
                        'avg_hurst': hurst_data.mean(),
                        'std_hurst': hurst_data.std(),
                        'avg_return': regime_data.mean(),
                        'volatility': volatility,
                        'max_drawdown': self._calculate_max_drawdown(regime_data),
                        'annualized_sharpe_ratio': sharpe
                    }

            # Print detailed analysis
            print(f"\nRegime Distribution:")
            for regime_type, stats in regime_stats.items():
                print(f"  {regime_type}: {stats['periods']} periods ({stats['percentage']:.1f}%)")
                print(f"    Average Hurst: {stats['avg_hurst']:.3f} ± {stats['std_hurst']:.3f}")
                print(
                    f"    Performance: Avg Daily Return {stats['avg_return']:.5f}, Daily Vol {stats['volatility']:.5f}, Sharpe {stats['annualized_sharpe_ratio']:.3f}")

            print(f"\nSignificant Regime Periods (≥{min_regime_duration} periods, showing last 10):")
            for i, period in enumerate(regime_periods[-10:], 1):
                print(
                    f"  {i}. {period['regime']}: {period['start_date']} to {period['end_date']}")
                print(f"     Duration: {period['duration']} periods, Avg Hurst: {period['avg_hurst']:.3f}")

            # Validate results with additional metrics
            validation_results = self._validate_hurst_regimes(aligned_series, regime_classification, valid_hurst)

            print(f"\n--- Validation Results ---")
            print(f"Regime persistence (autocorr of regime types): {validation_results['regime_persistence']:.3f}")
            print(f"Mean H in Trending vs. Mean-Reverting: {validation_results['cross_regime_difference']:.3f}")
            print(f"Correlation(Regime, Next Day's Return): {validation_results['regime_return_correlation']:.3f}")

            # Package results
            analysis_results = {
                'regime_classification': regime_classification,
                'regime_periods': regime_periods,
                'regime_statistics': regime_stats,
                'validation_results': validation_results,
                'parameters': {
                    'window_size': window_size,
                    'regime_thresholds': regime_thresholds,
                    'min_regime_duration': min_regime_duration,
                    'analysis_start': valid_hurst.index[0],
                    'analysis_end': valid_hurst.index[-1]
                }
            }

            return analysis_results

        def _calculate_max_drawdown(self, series: pd.Series):
            """Helper method to calculate maximum drawdown for a return series."""
            try:
                # Add 1 to daily returns to get growth factors for cumulative product
                cumulative = (1 + series).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return drawdown.min()
            except Exception:
                return np.nan

        def _validate_hurst_regimes(self, series: pd.Series, regime_classification: pd.Series,
                                    hurst_values: pd.Series):
            """Helper method to validate the regime classification results."""
            results = {
                'regime_persistence': 0.0,
                'cross_regime_difference': 0.0,
                'regime_return_correlation': 0.0
            }
            try:
                # Test 1: Regime persistence (do regimes tend to persist?)
                regime_numeric = regime_classification.map({
                    'Mean-Reverting': -1, 'Random Walk': 0, 'Trending': 1
                })
                regime_persistence = regime_numeric.autocorr(lag=1)
                results['regime_persistence'] = regime_persistence if pd.notna(regime_persistence) else 0.0

                # Test 2: Cross-regime Hurst differences
                mean_reverting_hurst = hurst_values[regime_classification == 'Mean-Reverting'].mean()
                trending_hurst = hurst_values[regime_classification == 'Trending'].mean()
                if pd.notna(trending_hurst) and pd.notna(mean_reverting_hurst):
                    results['cross_regime_difference'] = trending_hurst - mean_reverting_hurst

                # Test 3: Relationship between regime type and subsequent return's sign
                # A negative correlation suggests mean-reverting regimes have positive future returns (rebound)
                # and trending regimes have negative future returns (reversal of trend).
                future_returns = series.shift(-1)
                # Aligning data before calculating correlation
                aligned_data = pd.concat([regime_numeric, future_returns], axis=1).dropna()
                if not aligned_data.empty and len(aligned_data) > 1:
                    regime_return_corr = aligned_data.corr().iloc[0, 1]
                    results['regime_return_correlation'] = regime_return_corr if pd.notna(regime_return_corr) else 0.0

                return results

            except Exception:

                return results

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