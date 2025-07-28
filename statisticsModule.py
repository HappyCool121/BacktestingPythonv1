# calculates backtesting results and benchmarks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Module 6: Backtesting data and results ---
class StatisticsModule:
    """
    Calculates and presents key performance metrics, and plots relevant graphs from data gathered in the backtest
    Takes the equity history as well as closed trades list from backtest class.
    Plots equity curve (more can be added)

    Args: equity_curve: df (from backtest), trades: list[dicts] (list of closed trade dicts), risk free rate

    :return:
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{annualized_return:.2%}",
        "Annualized Volatility": f"{annualized_volatility:.2%}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Calmar Ratio": f"{calmar_ratio:.2f}",
        "Total Trades": len(self.trades),
        "Win Rate": f"{len(profit_trades) / len(self.trades):.2%}" if not self.trades.empty else "0.00%",
        "Profit Factor": f"{profit_trades['pnl'].sum() / abs(loss_trades['pnl'].sum()):.2f}" if abs(loss_trades['pnl'].sum()) > 0 else "inf",
        "Average Win": f"{profit_trades['pnl'].mean():.2f}",
        "Average Loss": f"{loss_trades['pnl'].mean():.2f}"

    """

    def __init__(self, equity_curve: pd.DataFrame, trades: list[dict], risk_free_rate: float = 0.02):
        self.equity_curve = equity_curve.set_index('date')['equity']
        # relevant portfolio_management class method:
        # def record_equity(self, date, current_price: float):
        #     """Records the total value of the portfolio at a point in time."""
        #     open_positions_value = sum(trade['quantity'] * current_price for trade in self.open_trades)
        #     total_equity = self.cash + open_positions_value
        #     self.equity_history.append({'date': date, 'equity': total_equity})
        #
        #

        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        # checks if the dict is empty
        # converts the list of dicts into a dataframe, dict keys being the column names
        self.risk_free_rate = risk_free_rate
        self.returns = self.equity_curve.pct_change().dropna()
        # .pct_change simply calculates pct change between current element and a prior element (ie daily returns)
        self.days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        # the index is the date, we simply take the difference between the first and last date
        # this will give us a Timedelta type, where we can use .days to extract the number of days

    def _get_empty_metrics(self) -> dict: # used by calculate_metrics method when empty args are inputted
        return {k: "N/A" for k in
                ["Total Return", "Annualized Return", "Annualized Volatility", "Max Drawdown", "Sharpe Ratio",
                 "Calmar Ratio", "Total Trades", "Win Rate", "Profit Factor", "Average Win", "Average Loss"]}

    def calculate_metrics(self) -> dict:
        """Calculates a dictionary of all key performance metrics."""
        if self.returns.empty or self.trades.empty:
            return self._get_empty_metrics()  # if either returns or trades doesn't exist

        # Overall Performance
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (365.25 / self.days) - 1

        # Risk Metrics
        annualized_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratio = (
                                   annualized_return - self.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0

        # Drawdown
        cumulative_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # Trade Statistics
        profit_trades = self.trades[self.trades['PNL'] > 0]
        loss_trades = self.trades[self.trades['PNL'] <= 0]

        return {
            "Total Return": f"{total_return:.2%}",
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Volatility": f"{annualized_volatility:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Total Trades": len(self.trades),
            "Win Rate": f"{len(profit_trades) / len(self.trades):.2%}" if not self.trades.empty else "0.00%",
            "Profit Factor": f"{profit_trades['PNL'].sum() / abs(loss_trades['PNL'].sum()):.2f}" if abs(
                loss_trades['PNL'].sum()) > 0 else "inf",
            "Average Win": f"{profit_trades['PNL'].mean():.2f}",
            "Average Loss": f"{loss_trades['PNL'].mean():.2f}",
        }

    def display_report(self, metrics: dict):
        """Prints a formatted report of the performance metrics."""
        print("\n--- Quantitative Performance Report ---")
        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
        print("---------------------------------------")

    def plot_equity(self, data: pd.DataFrame, initial_capital: float):
        """Plots the strategy equity curve against a Buy & Hold benchmark."""
        # Calculate Buy and Hold
        buy_hold_equity = (data['close'] / data['close'].iloc[0]) * initial_capital

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 8))

        ax.plot(self.equity_curve.index, self.equity_curve, label='Strategy Equity', color='royalblue', lw=2)
        ax.plot(data['date'], buy_hold_equity, label='Buy & Hold', color='grey', linestyle='--', lw=2)

        # Formatting
        ax.set_title('Portfolio Equity Curve vs. Buy & Hold', fontsize=16)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='upper left')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.tight_layout()
        plt.show()
