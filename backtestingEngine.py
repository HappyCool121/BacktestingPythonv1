# runs the backtest by iterating through the time series data

import pandas as pd
from statisticsModule import StatisticsModule
from portfolioManagement import PortfolioManagement

# --- Module 7: Running the backtest ---
class BacktestEngine:
    """
    Simulates a trading strategy on a bar-by-bar basis, managing a portfolio
    of cash and open positions with advanced trade management features.
    """
    def __init__(self,
         symbols: list[str], data_dict: dict[str: pd.DataFrame], portfolio: PortfolioManagement):

        """
        Initializes the iterative backtesting engine. Basic parameters are in arguments can be expanded later on.
        As of right now, there is data (time price data), initial capital, and transaction costs.

        Args:
            symbols (list[str]): list of symbols to trade on
            data_dict (dict[str: pd.DataFrame]): dict containing data for all symbols
            portfolio (PortfolioManagement): portfolio management object
        """

        self.symbols = symbols
        self.portfolio = portfolio
        self.trade_ID_counter = 0

        # --- Data Pre-processing Step ---
        # Merge the dictionary of DataFrames into a single, wide DataFrame for easy iteration.
        self.data = self._prepare_data(data_dict)

    def _prepare_data(self, data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges a dictionary of DataFrames into a single time-aligned DataFrame.
        """
        print("Preparing and merging multi-asset data...")
        processed_dfs = []
        for symbol, df in data_dict.items():
            # Set date as index to align all dataframes
            df_indexed = df.set_index('date')
            # Add the symbol as a suffix to each column (e.g., 'open' -> 'open_EURUSD=X')
            df_suffixed = df_indexed.add_suffix(f'_{symbol}')
            processed_dfs.append(df_suffixed)

        # Join all dataframes into one. 'outer' join handles missing dates for any symbol.
        merged_df = pd.concat(processed_dfs, axis=1, join='outer')

        # Forward-fill missing values that can occur from non-trading days in some assets
        merged_df.fillna(method='ffill', inplace=True)
        # Drop any remaining NaN rows at the beginning
        merged_df.dropna(inplace=True)

        # Reset the index to have 'date' as a column again for the loop
        merged_df.reset_index(inplace=True)
        print("Data preparation complete. Merged data:")
        print(merged_df.head())
        return merged_df

    def run(self):
        """
        Executes the multi-asset backtest, iterating through the merged dataframe.
        """
        print('Starting multi-asset backtest...')

        rowcounter = 0

        for index, row in self.data.iterrows():
            current_date = row['date']
            rowcounter += 1

            # --- 1. Check and manage existing open trades ---
            for trade in self.portfolio.open_trades[:]:
                symbol = trade.symbol
                current_low = row.get(f'low_{symbol}')
                current_high = row.get(f'high_{symbol}')
                #print(f'Current low: {current_low}, current high: {current_high}')

                # Skip if price data for this symbol isn't available on this date
                if pd.isna(current_low) or pd.isna(current_high):
                    continue

                if trade.direction == "LONG":
                    if current_low <= trade.stop_loss:
                        self.portfolio.close_trade(trade, current_date, trade.stop_loss, "STOP LOSS")
                        print(f'long STOP LOSS: low: {current_low} <= SL: {trade.stop_loss}')
                    elif current_high > trade.take_profit:
                        self.portfolio.close_trade(trade, current_date, trade.take_profit, "TAKE PROFIT")
                        print(f'long TAKE PROFIT: high {current_high} > TP: {trade.take_profit}')

                elif trade.direction == "SHORT":
                    if current_high >= trade.stop_loss:
                        self.portfolio.close_trade(trade, current_date, trade.stop_loss, "STOP LOSS")
                        print(f'short STOP LOSS: high {current_high} >= SL: {trade.stop_loss}')
                    elif current_low <= trade.take_profit:
                        self.portfolio.close_trade(trade, current_date, trade.take_profit, "TAKE PROFIT")
                        print(f'short STOP LOSS: low {current_high} <= TP: {trade.take_profit}')

            # --- 2. Check for new signals and open trades for each symbol ---
            for symbol in self.symbols:
                signal_value = row.get(f'signal_{symbol}', 0)

                if signal_value != 0:
                    # print(f'signal generated {rowcounter}')
                    # Ensure all required data for a trade exists on this row
                    required_cols = [f'open_{symbol}', f'stop_loss_level_{symbol}', f'take_profit_level_{symbol}']
                    if not all(col in row and pd.notna(row[col]) for col in required_cols):
                        print('continued because of missing required columns')
                        continue  # Skip if any data is missing for this signal

                    self.portfolio.open_trade(
                        entry_date=current_date,
                        symbol=symbol,
                        entry_price=row[f'open_{symbol}'],
                        signal_value=signal_value,
                        trade_ID=self.trade_ID_counter,
                        trade_type="SIMPLE",
                        stop_loss=row[f'stop_loss_level_{symbol}'],
                        take_profit=row[f'take_profit_level_{symbol}'],
                    )
                    self.trade_ID_counter += 1

            # --- 3. Record portfolio equity for the day ---
            market_prices = {s: row.get(f'close_{s}') for s in self.symbols}
            #print(market_prices)
            self.portfolio.record_equity(current_date, market_prices)

        # --- 4. End of backtest: Close any remaining open positions ---
        if self.portfolio.open_trades:
            print('backtest complete, closing all remaining open trades')
            last_row = self.data.iloc[-1]
            last_date = last_row['date']
            for trade in self.portfolio.open_trades[:]:
                last_price = last_row.get(f'close_{trade.symbol}')
                if pd.notna(last_price):
                    self.portfolio.close_trade(trade, last_date, last_price, "End of Backtest")
                    self.portfolio.record_equity(last_date, {s: last_row.get(f'close_{s}') for s in self.symbols})

        print("Backtest complete.")

    def generate_results(self, symbol: str):
        """Generates and displays the final report and plot."""
        equity_df = pd.DataFrame(self.portfolio.equity_history)

        primary_symbol = symbol
        primary_asset_data = self.data[['date', f'close_{primary_symbol}']].rename(
            columns={f'close_{primary_symbol}': 'close'})

        # You need to pass the actual StatisticsModule class here
        stats_module = StatisticsModule(equity_df, self.portfolio.closed_trades)
        metrics = stats_module.calculate_metrics()
        stats_module.display_report(metrics)
        stats_module.plot_equity(primary_asset_data, self.portfolio.initial_capital)

        # For now, just returning the raw data
        return {
            # "metrics": metrics,
            "trades": pd.DataFrame(self.portfolio.closed_trades),
            "equity_curve": pd.DataFrame(self.portfolio.equity_history)
        }