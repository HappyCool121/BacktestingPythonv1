# data collection and cleaning module

import pandas as pd
import yfinance as yf
from typing import List, Dict

# --- Module 1: Data Collection from yfinance ---
class DataHandler:
    """
    Handles fetching and preparing financial data for multiple symbols, returns a dict containing df of all signals
    """

    def __init__(self, symbols: List[str], start_date: str, end_date: str, interval: str = "1d"):
        """
        Initializes the DataHandler.
        Args:
            symbols (List[str]): A list of ticker symbols to download (e.g., ["EURUSD=X", "AUDUSD=X"]).
            start_date (str): The start date for the data in "YYYY-MM-DD" format.
            end_date (str): The end date for the data in "YYYY-MM-DD" format.
            interval (str): The data interval (e.g., "1d", "1h").
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def get_data(self) -> Dict[str, pd.DataFrame]:
        """
        Downloads data from yfinance for a list of symbols and returns a dictionary of DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are the symbol strings and values
                                     are the corresponding DataFrames with cleaned OHLCV data.
                                     Returns an empty dictionary if the input list is empty.
        """
        if not self.symbols:
            print("Error: No symbols provided in the list.")
            return {}

        all_data = {}

        for symbol in self.symbols:
            print(f"Fetching data for {symbol} from {self.start_date} to {self.end_date}...")

            # Download data from yfinance
            df = yf.download(symbol, start=self.start_date, end=self.end_date, interval=self.interval)

            if df.empty:
                print(f"Warning: No data found for symbol {symbol}. Skipping.")
                continue  # Skip to the next symbol

            # 1. Reset the index to make the 'Date' index a regular column
            df.reset_index(inplace=True)

            # 2. Select and rename columns to the desired format
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

            # 3. Add the cleaned DataFrame to our dictionary
            all_data[symbol] = df

            print(f"Data for {symbol} processed successfully.")

        return all_data