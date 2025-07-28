# strategy module, calculates signals, SL, TP and position sizes (to be added soon)

import pandas as pd
import numpy as np

# --- Module 3: Strategy Logic ---
class StrategyModule:
    """
    Generates trading signals based on the indicators.
    This module is designed to be easily extendable.
    Simply assigns 1, -1, and 0 to each row of the dataframe based on some indicators
    new addition:
    will now decide the stop loss and take profit prices/percentage
    first implementation: super trend indicator and strategy:
    will use atr for stop loss and take profit levels

    outputs: dataframe with new columns:
    data['signal']
    data['stop_loss_level']
    data['take_profit_level']
    """

    def __init__(self):
        pass

    def generate_ema_crossover_signals(self, data: pd.DataFrame, tp_mul: float, sl_mul: float) -> pd.DataFrame:
        """
        Generates signals for an EMA crossover strategy.
        Signal:
         -  1: Go Long (Fast EMA crosses above Slow EMA)
         - -1: Go Short/Sell (Fast EMA crosses below Slow EMA)
         -  0: Hold
        """
        # A long signal is generated when the fast EMA is above the slow EMA.
        # A short signal is generated when the fast EMA is below the slow EMA.
        data['signal'] = np.where(data['fast_ema'] > data['slow_ema'], 1, -1)

        data['signal'] = data['signal'].shift(1).fillna(0)  # signals to move forward by one
        signal_counts = data['signal'].value_counts()
        print(signal_counts)

        is_long_signal = data['signal'] == 1
        is_short_signal = data['signal'] == -1

        tp_multiplier = tp_mul
        sl_multiplier = sl_mul

        data['stop_loss_level'] = np.where(is_long_signal, (data['open'] - data['atr'] * sl_multiplier),
                                           np.where(is_short_signal, (data['open'] + data['atr'] * sl_multiplier), 0))

        data['take_profit_level'] = np.where(is_long_signal, (data['open'] + data['atr'] * tp_multiplier),
                                             np.where(is_short_signal, (data['open'] - data['atr'] * tp_multiplier), 0))

        print(data.head())

        return data

    def generate_rsi_oversold_overbought_signals(self, data: pd.DataFrame, rsi_low: int = 30,
                                                 rsi_high: int = 70) -> pd.DataFrame:
        """
        Generates signals based on RSI oversold/overbought levels.
        Signal:
         -  1: Go Long (RSI crosses above the low threshold)
         - -1: Go Short/Sell (RSI crosses below the high threshold)
         -  0: Hold
        """
        data['signal'] = np.where(data['rsi'] < rsi_low, 1, 0)
        data['signal'] = np.where(data['rsi'] > rsi_high, -1, data['signal'])
        return data

    def generateSupertrendSignals(self, data: pd.DataFrame, tp_mul: float, sl_mul: float) -> pd.DataFrame:
        """
        Generates signals based on super trend signals, along with stop loss and take profit levels, which are
        determined by the SL and TP multiplier.
        Signal:
            - 1: go long when current upper band value is higher than previous
            - 0: hold: when current upper band is higher than/equal to stop loss
            - -1: sell: when current position is 1, and current price is lower than stop loss
        """

        # Signal generation:
        # Define the condition for a long signal
        long_condition = data['super_trend_upper_band'] > data['super_trend_upper_band'].shift(1)
        # Define the condition for a short signal
        short_condition = data['super_trend_lower_band'] < data['super_trend_lower_band'].shift(1)
        # Apply the nested logic
        data['signal'] = np.where(long_condition, 1,
                                  np.where(short_condition, -1, 0))  # no short condition
        # move signals by one day to ensure no lookahead bias (previously done in backtest.run)
        data['signal'] = data['signal'].shift(1).fillna(0)  # signals to move forward by one
        signal_counts = data['signal'].value_counts()
        print(signal_counts)

        """
        look at rows 
        when there is a long signal, 
        use open price and atr value to calculate SL and TP values for trade
        add these SL and TP columns to the side

        when there is a short signal, 
        do the same, but ensure that stop loss and take profit values are logical 
        same as long, add these SL and TP columns to the side
        """
        is_long_signal = data['signal'] == 1
        is_short_signal = data['signal'] == -1

        tp_multiplier = tp_mul
        sl_multiplier = sl_mul

        data['stop_loss_level'] = np.where(is_long_signal, (data['open'] - data['atr'] * sl_multiplier),
                                           np.where(is_short_signal, (data['open'] + data['atr'] * sl_multiplier), 0))

        data['take_profit_level'] = np.where(is_long_signal, (data['open'] + data['atr'] * tp_multiplier),
                                             np.where(is_short_signal, (data['open'] - data['atr'] * tp_multiplier), 0))

        print(data.head())

        return data
