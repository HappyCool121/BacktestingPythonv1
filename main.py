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
        "initial_capital": 1000000.0,
        "commission_pct": 0.0001,  # 0.1% commission
    }

    data_handler = DataHandler(symbol=CONFIG["symbol"][0], start_date=CONFIG["start_date"],
                               end_date=CONFIG["end_date"])
    price_data = data_handler.get_data()

    indicator_module = IndicatorModule()

    price_data = indicator_module.add_supertrend2bands(price_data, period=14, multiplier=3.0)
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

