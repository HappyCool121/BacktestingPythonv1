# main file that runs backtest, all parameters are to be declared in this file

from portfolioManagement import portfolio_management
from backtestingEngine import BacktestEngine
from monteCarlo import monteCarloSimulation
from datahandler import DataHandler
from indicatorModule import IndicatorModule
from strategyModule import StrategyModule

# _______________________________MAIN_______________________________
if __name__ == '__main__':
    # Using the original modules provided by the user for data and signal generation
    # from __main__ import DataHandler, IndicatorModule, StrategyModule

    # 1. CONFIGURE BACKTEST
    CONFIG = {
        "symbol": ["EURUSD=X", "AUDUSD=X", "GBPUSD=X", "NZDUSD=X", "CADUSD=X"],
        "start_date": "2020-01-01",
        "end_date": "2024-06-25",
        "interval": "1d",
        "initial_capital": 1000000.0,
        "commission_pct": 0.0001,  # 0.1% commission
    }

    data_handler = DataHandler(symbols=CONFIG["symbol"], start_date=CONFIG["start_date"],
                               end_date=CONFIG["end_date"], interval=CONFIG["interval"])

    price_data_dict = data_handler.get_data()

    indicator_module = IndicatorModule()

    price_data = indicator_module.add_supertrend2bands(price_data_dict["AUDUSD=X"], period=14, multiplier=3.0)
    price_data = indicator_module.add_ema(price_data, fast_period=10, slow_period=20)

    strategy_module = StrategyModule()
    price_data_with_signals = strategy_module.generateSupertrendSignals(price_data, 100, 1)
    # price_data_with_signals = strategy_module.generate_ema_crossover_signals(price_data, 2, 1)

    if not price_data_with_signals.empty:
        # 3. INITIALIZE COMPONENTS
        # The Portfolio is created and configured
        portfolio = portfolio_management(
            initial_capital=CONFIG["initial_capital"],
            commission_pct=CONFIG["commission_pct"]
        )

        # The BacktestEngine is given the data and the portfolio to manage
        backtester = BacktestEngine(
            data=price_data_with_signals,
            portfolio=portfolio,
            profit_factor=2
        )

        # 4. RUN AND GET RESULTS
        run = backtester.run()

        results = backtester.generate_results()

        # You can now access the detailed results if needed
        print("\n--- Final Trade Log ---")
        print(results["metrics"])

        monte_carlo_module = monteCarloSimulation(results["returns_curve"], 1000, 10000.0)
        monte_carlo_results = monte_carlo_module.run()
        monte_carlo_plot = monte_carlo_module.plot_results(100)

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

