# arbitrary run file

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
        "start_date": "2020-01-01",
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

    timeseries_analytics_module = TimeSeriesAnalyticsModule()
    analytics_module = AnalyticsModule()

    df_with_analytics = analytics_module.calculate_returns_and_volatility(df, 7, 30, 365)
    # print(df_with_analytics.head())

    time_series_results_dict = timeseries_analytics_module.seasonaldecompose(df_with_analytics['daily_log_returns'])

    observed_df = time_series_results_dict['trend']

    analytics_module.plot_single_column(df_with_analytics, "daily_log_returns")
    analytics_module.plot_single_column(observed_df, "trend")



    # result = analytics_module.plot_single_column(df_with_analytics, "daily_log_returns", "plot of daily log returns")
    # #
    # plot_prices = analytics_module.plot_single_column(df, 'close', f'plot of daily close for {symbol_to_test}')
    # #
    # hurst, diagnostics = analytics_module.analyze_your_returns(df_with_analytics['daily_log_returns'])
    # #
    # # hurstvol, diagnosticsvol = analytics_module.analyze_your_returns(df_with_analytics['daily_log_returns_squared'])
    # #
    # # stationary_test = analytics_module.perform_comprehensive_stationarity_test(df_with_analytics['daily_log_returns'])
    # #
    # # autocorrelation_test2 = analytics_module.plot_autocorrelation_with_interpretation(df_with_analytics['daily_log_returns'])
    # #
    # # autocorrelation_testvol = analytics_module.plot_autocorrelation_with_interpretation(df_with_analytics['daily_log_returns_squared'])
    #
    # hurst_class = analytics_module.HurstAnalyzer()
    #
    # rolling_hurst = hurst_class.calculate_rolling_hurst_exponent(df_with_analytics['daily_log_returns'])
    #
    # analyze_regime = hurst_class.analyze_hurst_regimes(df_with_analytics['daily_log_returns'], rolling_hurst)
    #
    # ou_results = analytics_module.estimate_ou_parameters_for_regimes(df_with_analytics['daily_log_returns'], analyze_regime['regime_classification'], analyze_regime)
    #
    #
