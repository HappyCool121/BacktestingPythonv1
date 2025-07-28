# runs the backtest by iterating through the time series data

import pandas as pd
from statisticsModule import StatisticsModule
from portfolioManagement import portfolio_management

# --- Module 7: Running the backtest ---
class BacktestEngine:
    """
    Simulates a trading strategy on a bar-by-bar basis, managing a portfolio
    of cash and open positions with advanced trade management features.
    """
    def __init__(self,
         data: pd.DataFrame, portfolio: portfolio_management, profit_factor: float):

        """
        Initializes the iterative backtesting engine. Basic parameters are in arguments can be expanded later on.
        As of right now, there is data (time price data), initial capital, and transaction costs.

        Args:
            data (pd.DataFrame): DataFrame with price data and a 'signal' column.
            initial_capital (float): The starting capital for the portfolio.
            commissions_pct (float): percentage of commissions/fees for each trade
        """

        self.data = data.copy()
        self.portfolio = portfolio
        self.profit_factor = profit_factor


        # portfolio class inputted already has the following properties:
            # self.initial_capital = initial_capital
            # self.commission_pct = commission_pct

            # self.cash = initial_capital  # current liquid cash (excludes equity from open trades)
            # self.open_trades = []  # contains trades objects
            # self.closed_trades = []  # contains dicts (of closed trades)
            # self.equity_history = []  # contains date and equity history

            # self.SL_hit = 0
            # self.TP_hit = 0
            # self.end_of_backtest_hit = 0

    def run(self):
        """
        Executes the backtest, iterating through the entire dataframe. At each iteration,
        1. go through trades that are already open to see if any are supposed to be closed
        2. create new trades if there is a signal detected for that particular row
        3. find total equity (liquid cash + pnl of open trades) for the current date

        At the end of the loop, method should
        1. close all current open trades with "END OF BACKTEST"
        2. generate relevant results

        Args: none
        """

        print('Starting backtest...')
        trade_ID_counter = 0

        # move signal forward by one bar (DONE IN STRATEGY MODULE)
        # self.data['signal'] = self.data['signal'].shift(1).fillna(0)  # signals to move forward by one

        for index, row in self.data.iterrows(): #base iteration; through the whole pd

            #1. check for open trades that have hit SL/TP
            #2. those that didn't get stopped out, check if their stop loss needs to be updated
            for trade in self.portfolio.open_trades:

                if trade.direction == "LONG":

                    if row['low'] <= trade.stop_loss:
                        self.portfolio.close_trade(trade, row['date'], trade.stop_loss, "STOP LOSS")

                    elif row['high'] >= trade.take_profit:
                        self.portfolio.close_trade(trade, row['date'], trade.take_profit, "TAKE PROFIT")

                    elif trade.trade_type != "SIMPLE":
                        self.portfolio.update_SL(trade, row['open'])

                elif trade.direction == "SHORT":

                    if row['high'] >= trade.stop_loss:
                        self.portfolio.close_trade(trade, row['date'], exit_price=trade.stop_loss, exit_reason="STOP LOSS")

                    elif row['low'] <= trade.take_profit:
                        self.portfolio.close_trade(trade, row['date'], exit_price=trade.take_profit, exit_reason="TAKE PROFIT")

                    elif trade.trade_type != "SIMPLE":
                        self.portfolio.update_SL(trade, row['open'])

            if row['signal'] == 1:

                stop_loss_price = row['stop_loss_level']
                take_profit_price = row['take_profit_level']

                direction = "LONG"
                entry_price = row['open']
                trade_type = "TRAILING_SL_FIXED_TP"

                self.portfolio.open_trade(
                    entry_date=row['date'],
                    entry_price=entry_price,
                    trade_ID=trade_ID_counter,
                    trade_type=trade_type,
                    direction=direction,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                )

                trade_ID_counter += 1

            elif row['signal'] == -1:

                stop_loss_price = row['stop_loss_level']
                take_profit_price = row['take_profit_level']

                direction = "SHORT"
                entry_price = row['open']
                trade_type = "TRAILING_SL_FIXED_TP" #"SIMPLE"

                self.portfolio.open_trade(
                    entry_date=row['date'],
                    entry_price=entry_price,
                    trade_ID=trade_ID_counter,
                    trade_type=trade_type,
                    direction=direction,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                )

                trade_ID_counter += 1

            self.portfolio.record_equity(row['date'], row['close'])

        # end of loop
        if self.portfolio.open_trades:
            last_price = self.data['close'].iloc[-1]
            last_date = self.data['date'].iloc[-1]
            for trade in self.portfolio.open_trades[:]:
                self.portfolio.close_trade(trade, last_date, last_price, "End of Backtest")

        return 0

    # uses stats module
    def generate_results(self):
        """Generates and displays the final report and plot."""
        equity_df = pd.DataFrame(self.portfolio.equity_history)
        stats_module = StatisticsModule(equity_df, self.portfolio.closed_trades)

        metrics = stats_module.calculate_metrics()
        stats_module.display_report(metrics)
        stats_module.plot_equity(self.data, self.portfolio.initial_capital)

        return {
            "metrics": metrics,
            "trades": stats_module.trades,
            "equity_curve": stats_module.equity_curve,
            "returns_curve": stats_module.returns
        }
