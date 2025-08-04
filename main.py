# main file that runs backtest, all parameters are to be declared in this file

import pandas as pd

from analyticsModule import AnalyticsModule
from portfolioManagement import PortfolioManagement
from backtestingEngine import BacktestEngine
from monteCarlo import monteCarloSimulation
from datahandler import DataHandler
from indicatorModule import IndicatorModule
from strategyModule import StrategyModule

# _______________________________MAIN_______________________________
if __name__ == '__main__':
    # 1. CONFIGURE BACKTEST FOR A SINGLE ASSET
    symbols = ["BTC-USD", "AUDUSD=X", "GC=F"]
    symbol_to_test = "BTC-USD"
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

    analytics_module = AnalyticsModule()

    df_with_analytics = analytics_module.calculate_returns_and_volatility(df, 7, 30, 365)
    print(df_with_analytics.head())

    # result = analytics_module.plot_single_column(df_with_analytics, "daily_log_returns", "plot of daily log returns")
    #
    plot_prices = analytics_module.plot_single_column(df, 'close', f'plot of daily close for {symbol_to_test}')
    #
    # hurst, diagnostics = analytics_module.analyze_your_returns(df_with_analytics['daily_log_returns'])
    #
    # hurstvol, diagnosticsvol = analytics_module.analyze_your_returns(df_with_analytics['daily_log_returns_squared'])
    #
    # stationary_test = analytics_module.perform_comprehensive_stationarity_test(df_with_analytics['daily_log_returns'])
    #
    # autocorrelation_test2 = analytics_module.plot_autocorrelation_with_interpretation(df_with_analytics['daily_log_returns'])
    #
    # autocorrelation_testvol = analytics_module.plot_autocorrelation_with_interpretation(df_with_analytics['daily_log_returns_squared'])

    hurst_class = analytics_module.HurstAnalyzer()

    rolling_hurst = hurst_class.calculate_rolling_hurst_exponent(df_with_analytics['daily_log_returns'])

    analyze_regime = hurst_class.analyze_hurst_regimes(df_with_analytics['daily_log_returns'], rolling_hurst)

#BACKTESTING STRATEGY

    # # Apply indicators
    # indicator_module = IndicatorModule()
    # df = indicator_module.add_supertrend2bands(df, period=14, multiplier=3.0)
    # # df = indicator_module.add_ema(df, fast_period=10, slow_period=20) # Example
    #
    # # Apply strategy to generate signals and SL/TP levels
    # strategy_module = StrategyModule()
    # df_with_signals = strategy_module.generateSupertrendSignals(df, tp_mul=2.0, sl_mul=1.0)
    #
    # # --- Prepare data for the multi-asset engine ---
    # # Put the processed DataFrame back into a dictionary
    # final_data_dict = {symbol_to_test: df_with_signals}
    #
    # # 3. INITIALIZE COMPONENTS
    # if not df_with_signals.empty:
    #     portfolio = PortfolioManagement(
    #         initial_capital=CONFIG["initial_capital"],
    #         commission_pct=CONFIG["commission_pct"],
    #     )
    #
    #     # The BacktestEngine now takes the list of symbols and the dictionary of data
    #     backtester = BacktestEngine(
    #         symbols=CONFIG["symbols"],
    #         data_dict=final_data_dict,
    #         portfolio=portfolio
    #     )
    #
    #     # 4. RUN AND GET RESULTS
    #     backtester.run()
    #     results = backtester.generate_results(symbol_to_test)
    #
    #     # You can now access the detailed results if needed
    #     print("\n--- Final Trade Log ---")
    #     print(results["trades"].head())
    #

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

