# main file that runs backtest, all parameters are to be declared in this file

import pandas as pd

from analyticsModule import AnalyticsModule
from portfolioManagement import PortfolioManagement
from backtestingEngine import BacktestEngine
from monteCarlo import monteCarloSimulation
from datahandler import DataHandler
from indicatorModule import IndicatorModule
from strategyModule import StrategyModule
from analyticsModule import TimeSeriesAnalyticsModule

# _______________________________MAIN_______________________________
if __name__ == '__main__':
    # 1. CONFIGURE BACKTEST FOR A SINGLE ASSET
    symbols = ["GC=F"]
    symbol_to_test = "GC=F"
    CONFIG = {
        "symbols": symbols, # Pass the symbol as a list
        "start_date": "2015-01-01",
        "end_date": "2025-06-25",
        "interval": "1d",
        "initial_capital": 10000.0,
        "commission_pct": 0.0001,
        "pct_capital_risk": 0.01,
        "leverage": 1.0 # No leverage for now
    }

    pd.set_option('display.max_columns', None)

    # 2. PREPARE DATA AND SIGNALS
    # DataHandler fetches data and returns a dictionary
    data_handler = DataHandler(
        symbols=CONFIG["symbols"],
        start_date=CONFIG["start_date"],
        end_date=CONFIG["end_date"],
        interval=CONFIG["interval"]
    )
    price_data_dict = data_handler.get_data()

    # Isolate the DataFrame for our target symbol
    df = price_data_dict[symbol_to_test]

#BACKTESTING STRATEGY

    # Apply indicators
    indicator_module = IndicatorModule()
    df = indicator_module.add_supertrend2bands(df, period=14, multiplier=3.0)
    # df = indicator_module.add_ema(df, fast_period=10, slow_period=20) # Example

    # Apply strategy to generate signals and SL/TP levels
    strategy_module = StrategyModule()
    df_with_signals = strategy_module.generateSupertrendSignals(df, tp_mul=2.0, sl_mul=1.0)

    # --- Prepare data for the multi-asset engine ---
    # Put the processed DataFrame back into a dictionary
    final_data_dict = {symbol_to_test: df_with_signals}

    # 3. INITIALIZE COMPONENTS
    if not df_with_signals.empty:
        portfolio = PortfolioManagement(
            initial_capital=CONFIG["initial_capital"],
            commission_pct=CONFIG["commission_pct"],
        )

        # The BacktestEngine now takes the list of symbols and the dictionary of data
        backtester = BacktestEngine(
            symbols=CONFIG["symbols"],
            data_dict=final_data_dict,
            portfolio=portfolio
        )

        # 4. RUN AND GET RESULTS
        backtester.run()
        results = backtester.generate_results(symbol_to_test)

        # You can now access the detailed results if needed
        print("\n--- Final Trade Log ---")
        print(results["trades"].head())


