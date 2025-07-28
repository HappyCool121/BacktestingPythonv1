# indicator module, which creates a pandas dataframe which includes all the indicators calculated

import pandas as pd
import numpy as np

# --- Module 2: Basic Indicator Calculation ---
class IndicatorModule:
    """
    Calculates and adds technical indicators to the DataFrame.
    This will add another column directly to the price data with values calculated by the indicator.
    This module is designed to be easily extendable.
    """
    def __init__(self):
        pass

    def add_ema(self, data: pd.DataFrame, fast_period: int = 10, slow_period: int = 30) -> pd.DataFrame:
        """
        Calculates and adds Fast and Slow Exponential Moving Averages (EMAs).

        Args:
            data (pd.DataFrame): The input DataFrame with a 'close' column.
            fast_period (int): The window for the fast EMA.
            slow_period (int): The window for the slow EMA.

        Returns:
            pd.DataFrame: The DataFrame with 'fast_ema' and 'slow_ema' columns added.
        """
        data['fast_ema'] = data['close'].ewm(span=fast_period, adjust=False).mean()
        data['slow_ema'] = data['close'].ewm(span=slow_period, adjust=False).mean()
        return data

    def add_ema1(self, data: pd.DataFrame, period: int = 30) -> pd.DataFrame:

        data['ema'] = data['close'].ewm(span=period, adjust=False).mean()
        return data

    def add_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculates and adds the Relative Strength Index (RSI).

        Args:
            data (pd.DataFrame): The input DataFrame with a 'close' column.
            period (int): The window for the RSI calculation.

        Returns:
            pd.DataFrame: The DataFrame with the 'rsi' column added.
        """
        delta = data['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        return data

    def add_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculates and adds the Average True Range (ATR).

        Args:
            data (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.
            period (int): The window for the ATR calculation (usually 14).

        Returns:
            pd.DataFrame: The DataFrame with the 'tr' (True Range) and 'atr' columns added.
        """
        # Calculate True Range (TR)
        # 1. Current High - Current Low
        tr1 = data['high'] - data['low']
        # 2. Absolute(Current High - Previous Close)
        tr2 = abs(data['high'] - data['close'].shift(1))
        # 3. Absolute(Current Low - Previous Close)
        tr3 = abs(data['low'] - data['close'].shift(1))

        # Take the maximum of the three True Range components
        # We use .max(axis=1) because tr1, tr2, tr3 are Series, and we want
        # the max for each row.
        data['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Average True Range (ATR)
        # Wilder's original ATR uses a smoothed moving average, similar to EMA.
        # The 'adjust=False' is important for matching standard ATR calculations.
        data['atr'] = data['tr'].ewm(span=period, adjust=False).mean()

        return data

    def daily_returns(self, data: pd.DataFrame, period = 30, period2 = 120) -> pd.DataFrame:
        data['daily returns'] = data['close'].pct_change()
        data['rolling_mean_short'] = data['daily returns'].rolling(period).mean()
        data['rolling_mean_long'] = data['daily returns'].rolling(period2).mean()

        data['vol_short'] = data['daily returns'].rolling(period).std()
        data['vol_long'] = data['daily returns'].rolling(period2).std()

        return data

    def daily_log_returns(self, data: pd.DataFrame, period = 30, period2 = 120) -> pd.DataFrame:
        data['daily_log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['rolling_log_short'] = data['daily_log_returns'].rolling(period).mean()
        data['rolling_log_long'] = data['daily_log_returns'].rolling(period2).mean()

        data['log_vol_short'] = data['daily_log_returns'].rolling(period).std()
        data['log_vol_long'] = data['daily_log_returns'].rolling(period2).std()

        return data

    def add_supertrend2bands(self, data: pd.DataFrame, period: int = 14, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Calculates ATR with given period, and calculates upper and lower bands of super trend
        given a multiplier. Super trend logic: Lower band will keep increasing if higher lows
        are created. When there is a lower low, as long as lowest price stays above the latest
        lower band, the lower band will remain the same. But when prices do end up being lower
        than the latest lowest band, new lower band will be updated.

        Args:
            data (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.
            period (int): The window for the ATR calculation (usually 14).
            multiplier (float): The multiplier for the ATR.
        Returns:
            pd.DataFrame: The DataFrame with 'atr', 'upper_band' and 'lower_band' columns added.
        """

        # Calculate True Range (TR)
        # 1. Current High - Current Low
        tr1 = data['high'] - data['low']
        # 2. Absolute(Current High - Previous Close)
        tr2 = abs(data['high'] - data['close'].shift(1))
        # 3. Absolute(Current Low - Previous Close)
        tr3 = abs(data['low'] - data['close'].shift(1))

        # Take the maximum of the three True Range components
        # We use .max(axis=1) because tr1, tr2, tr3 are Series, and we want
        # the max for each row.
        data['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Average True Range (ATR)
        # Wilder's original ATR uses a smoothed moving average, similar to EMA.
        # The 'adjust=False' is important for matching standard ATR calculations.
        data['atr'] = data['tr'].ewm(span=period, adjust=False).mean() #average true range

        #calculate the upper and lower bands depending on ATR and multiplier
        #data['upper_band'] = (data['high'] - data['low'])/2 + multiplier*data['atr']
        #data["lower_band"] = (data['high'] - data['low'])/2 - multiplier*data['atr']

        data['midpoint'] = (data['high'] + data['low'])/2
        data['upper_band'] = data['midpoint'] + (data['atr'] * multiplier)
        data["lower_band"] = data['midpoint'] - (data['atr'] * multiplier)

        data = data.fillna(0)
        upperbandLatest = 1000000000.0 #max value
        lowerbandLatest = 0.0 #minimum value

        for index, row in data.iterrows():
            upperband = row['upper_band']
            priceHigh = row['high']

            if upperband < upperbandLatest: #which at first is 1000000.0
                #update the upper band to move lower
                upperbandLatest = upperband
                data.at[index, 'super_trend_upper_band'] = upperband

            elif upperband > upperbandLatest: #when upper band increases above the previous upper band
                #check if prices break above the latest upper band

                if priceHigh < upperbandLatest:
                    #prices are still below the latest upper band
                    data.at[index, 'super_trend_upper_band'] = upperbandLatest #remains the same

                elif priceHigh > upperbandLatest:
                    #prices have broken the latest upper band
                    upperbandLatest = upperband
                    data.at[index, 'super_trend_upper_band'] = upperband

        for index, row in data.iterrows():
            lowerband = row['lower_band']
            priceLow = row['low']

            if lowerband > lowerbandLatest:
                #new lows created
                lowerbandLatest = lowerband
                data.at[index, 'super_trend_lower_band'] = lowerband

            elif lowerband < lowerbandLatest:
                #check if prices broke through latest lower band

                if priceLow > lowerbandLatest:
                    #prices still above lower band, same latest lower band
                    data.at[index, 'super_trend_lower_band'] = lowerbandLatest

                elif priceLow < lowerbandLatest:
                    #prices now below lower band, update lower band
                    lowerbandLatest = lowerband
                    data.at[index, 'super_trend_lower_band'] = lowerbandLatest

        # SIGNAL GENERATION LINE
        # data['signal'] = np.where(data['super_trend_upper_band'] > data['super_trend_upper_band'].shift(1), 1, 0)
        # signal generation is done in strategy class, even though the signals themselves are in the same dataframe

        return data